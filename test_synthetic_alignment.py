#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_synthetic_alignment.py

Test de récupération synthétique pour le pipeline Procrustes figé
(local_jnd_bridge_calibration v1.1.0).

Ce test N'EST PAS une validation psychophysique. Il vérifie que, sur
une géométrie contrôlée

    X_obs = s * X_ref @ R.T + t + bruit

l'estimation Procrustes sur des ponts d'entraînement, suivie de
l'application figée par apply_procrustes_transform, permet de récupérer
la géométrie de référence sur des points tenus à l'écart.

Modes comparés :
    centering_mode="common"        (référence principale)
    centering_mode="per_condition" (analyse de sensibilité)

Erreurs rapportées séparément :
    1. erreur point-à-point sur les ponts tenus à l'écart ;
    2. erreur des distances inter-points ;
    3. erreur relative du volume estimé par condition
       (métrique locale = identité → volume euclidien de l'enveloppe).

API utilisée (v1.1.0) :

    from local_jnd_bridge_calibration import (
        procrustes_align_bridges,
        apply_procrustes_transform,
        local_volume_in_region,
    )

    result = procrustes_align_bridges(
        X_src, X_dst,
        centering_mode="common" | "per_condition",
        condition_labels=<ndarray>,   # obligatoire en per_condition
    )
    transform = result["transform"]   # ou result entier

    X_aligned = apply_procrustes_transform(
        X, transform,
        condition_labels=<ndarray>,   # obligatoire en per_condition
    )

    V = local_volume_in_region(
        region_samples, anchor_points, anchor_metrics,
        bandwidth=..., region_base_volume=...,
    )
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, "/home/workdir/artifacts")

from local_jnd_bridge_calibration import (
    apply_procrustes_transform,
    local_volume_in_region,
    procrustes_align_bridges,
)

__all__ = [
    "run_case",
    "test_common_and_per_condition",
    "test_per_condition_absent_raises",
]


# ---------- générateurs ----------


def _random_rotation(rng: np.random.Generator, d: int) -> np.ndarray:
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


# ---------- métriques ----------


def _point_errors(X_ref: np.ndarray, X_est: np.ndarray) -> tuple[float, float]:
    err = np.linalg.norm(X_est - X_ref, axis=1)
    return float(np.median(err)), float(np.percentile(err, 95))


def _distance_errors(X_ref: np.ndarray, X_est: np.ndarray) -> tuple[float, float]:
    D_ref = np.linalg.norm(X_ref[:, None, :] - X_ref[None, :, :], axis=-1)
    D_est = np.linalg.norm(X_est[:, None, :] - X_est[None, :, :], axis=-1)
    mask = \~np.eye(len(X_ref), dtype=bool)
    rel = np.abs(D_est - D_ref)[mask] / (D_ref[mask] + 1e-12)
    return float(np.median(rel)), float(np.percentile(rel, 95))


def _euclid_volume_proxy(
    points: np.ndarray,
    bandwidth: float = 2.0,
) -> float:
    """
    Volume local sous métrique identité (densité = 1).

    On place une ancre au barycentre avec G = I, puis on évalue
    local_volume_in_region sur les points fournis. Avec G = I partout,
    estimated_volume ≈ region_base_volume * 1.
    region_base_volume est pris égal au volume de la boîte englobante
    des points (proxy euclidien simple).
    """
    if len(points) < 2:
        return 0.0

    d = points.shape[1]
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    side = np.maximum(hi - lo, 1e-6)
    base_vol = float(np.prod(side))

    center = points.mean(axis=0, keepdims=True)
    G_eye = np.eye(d)[None, :, :]

    out = local_volume_in_region(
        region_samples=points,
        anchor_points=center,
        anchor_metrics=G_eye,
        bandwidth=bandwidth,
        region_base_volume=base_vol,
        interpolation="linear",
    )
    return float(out["estimated_volume"])


# ---------- cas principal ----------


def run_case(mode: str, seed: int = 0) -> Dict[str, Any]:
    """
    Exécute un cas synthétique complet pour un mode de centrage donné.

    Retourne un dictionnaire des erreurs (médiane, p95) sur les
    ponts tenus à l'écart, plus l'erreur relative de volume par condition.
    """
    rng = np.random.default_rng(seed)
    d = 3
    n_conditions = 3
    n_anchors_train = 5
    n_anchors_held = 3
    n_unique = 6
    s_true = 1.7
    noise_std = 0.02

    anchors = rng.standard_normal((n_anchors_train + n_anchors_held, d)) * 0.5
    centers = rng.uniform(-3.0, 3.0, size=(n_conditions, d))

    X_ref_list: List[np.ndarray] = []
    cond_list: List[int] = []
    kind_list: List[str] = []
    idx_list: List[int] = []

    for c in range(n_conditions):
        for i, a in enumerate(anchors):
            X_ref_list.append(a + centers[c])
            cond_list.append(c)
            kind_list.append("anchor")
            idx_list.append(i)
        uniques = rng.standard_normal((n_unique, d)) * 0.5
        for j, u in enumerate(uniques):
            X_ref_list.append(u + centers[c])
            cond_list.append(c)
            kind_list.append("unique")
            idx_list.append(j)

    X_ref = np.asarray(X_ref_list, dtype=float)
    cond = np.asarray(cond_list, dtype=int)
    kind = np.asarray(kind_list)
    idx = np.asarray(idx_list, dtype=int)

    R_true = _random_rotation(rng, d)
    t_true = rng.uniform(-2.0, 2.0, size=d)
    X_obs = s_true * (X_ref @ R_true.T) + t_true
    X_obs = X_obs + rng.standard_normal(X_obs.shape) * noise_std

    train_mask = (kind == "anchor") & (idx < n_anchors_train)
    held_mask = (kind == "anchor") & (idx >= n_anchors_train)

    if mode == "common":
        result = procrustes_align_bridges(
            X_obs[train_mask],
            X_ref[train_mask],
            allow_reflection=True,
            centering_mode="common",
        )
        X_aligned = apply_procrustes_transform(X_obs, result)
    elif mode == "per_condition":
        result = procrustes_align_bridges(
            X_obs[train_mask],
            X_ref[train_mask],
            allow_reflection=True,
            centering_mode="per_condition",
            condition_labels=cond[train_mask],
        )
        X_aligned = apply_procrustes_transform(
            X_obs,
            result,
            condition_labels=cond,
        )
    else:
        raise ValueError(f"mode inconnu : {mode!r}")

    p_med, p_p95 = _point_errors(X_ref[held_mask], X_aligned[held_mask])
    d_med, d_p95 = _distance_errors(X_ref[held_mask], X_aligned[held_mask])

    vol_errs: List[float] = []
    for c in range(n_conditions):
        m = (cond == c) & (kind == "unique")
        V_ref = _euclid_volume_proxy(X_ref[m])
        V_est = _euclid_volume_proxy(X_aligned[m])
        if V_ref > 1e-12:
            vol_errs.append(abs(V_est - V_ref) / V_ref)

    return {
        "mode": mode,
        "point_median": p_med,
        "point_p95": p_p95,
        "dist_median": d_med,
        "dist_p95": d_p95,
        "vol_median": float(np.median(vol_errs)) if vol_errs else float("nan"),
        "vol_max": float(np.max(vol_errs)) if vol_errs else float("nan"),
        "disparity_train": float(result["disparity"]),
        "scale": float(result["scale"]),
    }


# ---------- tests ----------


def test_common_and_per_condition() -> None:
    """
    Exécute les deux modes et vérifie que common atteint les seuils.

    Seuils absolus calibrés sur le bruit synthétique (noise_std=0.02,
    échelle de référence \~0.5). Le mode common doit les satisfaire.
    per_condition est rapporté sans assertion de supériorité : il
    quantifie l'effet du choix de centrage.
    """
    r_common = run_case("common", seed=0)
    r_pc = run_case("per_condition", seed=0)

    for r in (r_common, r_pc):
        print(
            f"[{r['mode']:14s}] "
            f"point_med={r['point_median']:.4f}  "
            f"dist_med={r['dist_median']:.4f}  "
            f"vol_med={r['vol_median']:.4f}  "
            f"scale={r['scale']:.4f}  "
            f"disp={r['disparity_train']:.2e}"
        )

    assert r_common["point_median"] < 0.10, (
        f"point_median common trop élevé : {r_common['point_median']:.4f}"
    )
    assert r_common["dist_median"] < 0.05, (
        f"dist_median common trop élevé : {r_common['dist_median']:.4f}"
    )
    assert r_common["vol_median"] < 0.15, (
        f"vol_median common trop élevé : {r_common['vol_median']:.4f}"
    )


def test_per_condition_absent_raises() -> None:
    """per_condition doit refuser une condition absente de l'entraînement."""
    rng = np.random.default_rng(1)
    d = 3
    X_ref = rng.standard_normal((15, d)) * 0.5
    cond = np.repeat([0, 1, 2], 5)
    R = _random_rotation(rng, d)
    X_obs = 1.3 * (X_ref @ R.T) + rng.uniform(-1.0, 1.0, size=d)

    train_mask = cond != 2
    result = procrustes_align_bridges(
        X_obs[train_mask],
        X_ref[train_mask],
        centering_mode="per_condition",
        condition_labels=cond[train_mask],
    )

    cond_with_unseen = cond.copy()
    cond_with_unseen[cond_with_unseen == 2] = 99

    raised = False
    try:
        apply_procrustes_transform(
            X_obs,
            result,
            condition_labels=cond_with_unseen,
        )
    except ValueError:
        raised = True

    if not raised:
        raise AssertionError(
            "per_condition aurait dû refuser une condition absente "
            "de l'entraînement"
        )


if __name__ == "__main__":
    test_common_and_per_condition()
    test_per_condition_absent_raises()
    print("OK")
