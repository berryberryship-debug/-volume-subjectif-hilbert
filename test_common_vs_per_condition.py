#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test synthétique à géométrie connue : centrage COMMUN vs PAR CONDITION.

Principe
--------
1. Géométrie cible connue Z (plusieurs conditions, champ métrique connu).
2. Source = échelle * rotation * Z + translation (+ décalage propre à chaque
   condition dans le scénario B) + bruit.
3. La transformation est estimée UNIQUEMENT sur les ponts d'entraînement,
   puis appliquée figée (apply_procrustes_transform) aux ponts tenus à l'écart.
4. Erreurs mesurées sur les ponts tenus à l'écart :
     - erreur de coordonnées (RMSE) ;
     - erreur relative de distances euclidiennes par paires ;
     - erreur relative de distance métrique intégrée ;
     - erreur relative de volume local.

Scénario A : la vraie transformation est commune à toutes les conditions.
Scénario B : chaque condition porte en plus un décalage source propre.

Usage : python test_common_vs_per_condition.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from local_jnd_bridge_calibration import (
    apply_procrustes_transform,
    local_volume_in_region,
    path_integrated_distance,
    procrustes_align_bridges,
)

N_COND, N_PER, N_TRAIN_PER, DIM = 3, 12, 8, 3
BANDWIDTH = 1.5


def random_rotation(rng, d):
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def make_world(rng, scenario, noise):
    centers = rng.normal(scale=1.5, size=(N_COND, DIM))
    Z = np.vstack([c + 0.6 * rng.normal(size=(N_PER, DIM)) for c in centers])
    labels = np.repeat(np.arange(N_COND), N_PER)

    Q = random_rotation(rng, DIM)
    s = rng.uniform(0.5, 2.5)
    t = rng.normal(scale=3.0, size=DIM)

    offsets = np.zeros_like(Z)
    if scenario == "B":
        offsets = rng.normal(scale=1.0, size=(N_COND, DIM))[labels]

    S = s * (Z @ Q.T) + t + offsets + noise * s * rng.normal(size=Z.shape)

    metrics = []
    for _ in range(N_COND):
        B = 0.5 * rng.normal(size=(DIM, DIM))
        metrics.append(B @ B.T + 0.5 * np.eye(DIM))

    return {"Z": Z, "S": S, "labels": labels, "anchors": centers,
            "metrics": np.stack(metrics), "s": s}


def split(rng, labels):
    train, held = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        train.extend(idx[:N_TRAIN_PER])
        held.extend(idx[N_TRAIN_PER:])
    return np.array(train), np.array(held)


def evaluate(world, tr, ho, mode, pairs):
    lab = world["labels"]
    per = mode == "per_condition"

    fit = procrustes_align_bridges(
        world["S"][tr], world["Z"][tr],
        allow_reflection=False,
        centering_mode=mode,
        condition_labels=lab[tr] if per else None,
    )
    # Transformation figée appliquée aux ponts tenus à l'écart.
    Zhat = apply_procrustes_transform(
        world["S"][ho], fit, lab[ho] if per else None
    )
    Ztrue = world["Z"][ho]

    coord_rmse = float(np.sqrt(np.mean(np.sum((Zhat - Ztrue) ** 2, axis=1))))

    d_hat, d_true = pdist(Zhat), pdist(Ztrue)
    dist_err = float(np.mean(np.abs(d_hat - d_true)) / np.mean(d_true))

    A, M = world["anchors"], world["metrics"]
    ml_hat, ml_true = [], []
    for i, j in pairs:
        ml_hat.append(path_integrated_distance(
            np.vstack([Zhat[i], Zhat[j]]), A, M, BANDWIDTH, n_substeps=10))
        ml_true.append(path_integrated_distance(
            np.vstack([Ztrue[i], Ztrue[j]]), A, M, BANDWIDTH, n_substeps=10))
    ml_hat, ml_true = np.array(ml_hat), np.array(ml_true)
    metric_err = float(np.mean(np.abs(ml_hat - ml_true)) / np.mean(ml_true))

    v_hat = local_volume_in_region(Zhat, A, M, BANDWIDTH, 1.0)["estimated_volume"]
    v_true = local_volume_in_region(Ztrue, A, M, BANDWIDTH, 1.0)["estimated_volume"]
    vol_err = float(abs(v_hat - v_true) / v_true)

    return {"coord_rmse": coord_rmse, "dist_err": dist_err,
            "metric_dist_err": metric_err, "vol_err": vol_err}


# ----------------------------------------------------------------------------
# Contrôles exacts (sans bruit)
# ----------------------------------------------------------------------------

def exactness_checks():
    rng = np.random.default_rng(0)
    n_ho = N_COND * (N_PER - N_TRAIN_PER)
    pairs = [(i, (i + 1) % n_ho) for i in range(n_ho)]

    # A sans bruit : les deux modes retrouvent la géométrie.
    w = make_world(rng, "A", 0.0)
    tr, ho = split(rng, w["labels"])
    for mode in ("common", "per_condition"):
        r = evaluate(w, tr, ho, mode, pairs)
        assert r["coord_rmse"] < 1e-9, (mode, r)

    # B sans bruit : seul le centrage par condition retrouve la géométrie.
    w = make_world(rng, "B", 0.0)
    tr, ho = split(rng, w["labels"])
    r_per = evaluate(w, tr, ho, "per_condition", pairs)
    r_com = evaluate(w, tr, ho, "common", pairs)
    assert r_per["coord_rmse"] < 1e-9, r_per
    assert r_com["coord_rmse"] > 1e-2, r_com

    # La transformation est figée : l'appliquer à l'entraînement reproduit
    # exactement aligned_coords, et modifier les données de test ne change
    # pas la transformation.
    fit = procrustes_align_bridges(w["S"][tr], w["Z"][tr], centering_mode="per_condition",
                                   condition_labels=w["labels"][tr], allow_reflection=False)
    again = apply_procrustes_transform(w["S"][tr], fit, w["labels"][tr])
    assert np.allclose(again, fit["aligned_coords"])

    # Condition inconnue : erreur explicite, pas de repli silencieux.
    try:
        apply_procrustes_transform(w["S"][ho][:2], fit, np.array([99, 99]))
        raise AssertionError("condition inconnue non détectée")
    except ValueError:
        pass

    # Étiquettes manquantes en mode per_condition : erreur explicite.
    try:
        procrustes_align_bridges(w["S"][tr], w["Z"][tr], centering_mode="per_condition")
        raise AssertionError("labels manquants non détectés")
    except ValueError:
        pass

    print("Contrôles exacts (sans bruit) : OK")


# ----------------------------------------------------------------------------
# Simulation répétée
# ----------------------------------------------------------------------------

def simulate(scenario, noise, n_rep=150, seed=123):
    rng = np.random.default_rng(seed)
    n_ho = N_COND * (N_PER - N_TRAIN_PER)
    pairs = [(i, j) for i in range(n_ho) for j in range(i + 1, n_ho)]
    pairs = [pairs[k] for k in rng.choice(len(pairs), size=20, replace=False)]

    out = {"common": [], "per_condition": []}
    for _ in range(n_rep):
        w = make_world(rng, scenario, noise)
        tr, ho = split(rng, w["labels"])
        for mode in out:
            out[mode].append(evaluate(w, tr, ho, mode, pairs))
    return out


def report(scenario, noise, out):
    keys = ["coord_rmse", "dist_err", "metric_dist_err", "vol_err"]
    print(f"\nScénario {scenario}, bruit = {noise}")
    print(f"{'métrique':<18}{'médiane commun':>16}{'médiane par cond.':>20}"
          f"{'P(par cond. < commun)':>24}")
    for k in keys:
        c = np.array([r[k] for r in out["common"]])
        p = np.array([r[k] for r in out["per_condition"]])
        print(f"{k:<18}{np.median(c):>16.4f}{np.median(p):>20.4f}"
              f"{np.mean(p < c):>24.2f}")


if __name__ == "__main__":
    exactness_checks()
    for scenario in ("A", "B"):
        for noise in (0.05, 0.20):
            report(scenario, noise, simulate(scenario, noise))
