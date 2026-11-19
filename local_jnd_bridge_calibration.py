#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration indépendante des stimuli-ponts (version corrigée).

Fonctions principales :
- alignement Procrustes (translation, rotation, échelle) ;
- estimation d'une métrique locale SPD G(x) avec shrinkage ;
- interpolation vectorisée du champ métrique local ;
- distances intégrées et volumes locaux ;
- calibration linéaire ou monotone des distances ;
- validation sur paires de stimuli-ponts.

Les données de calibration doivent rester indépendantes des données utilisées
pour ajuster la métrique principale W_theta.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

__all__ = [
    "apply_linear_scale",
    "apply_monotone_binned",
    "evaluate_calibration_split",
    "fit_linear_scale",
    "fit_local_metric_at_anchor",
    "fit_monotone_binned",
    "interpolate_local_metric",
    "interpolate_local_metric_batch",
    "path_integrated_distance",
    "local_volume_in_region",
    "procrustes_align_bridges",
    "calibrate_and_validate_bridges",
]


def _as_finite_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contient des valeurs non finies.")
    return arr


def _ensure_spd(matrix: np.ndarray, eig_floor: float = 1e-10) -> np.ndarray:
    """Symétrise une matrice (ou un lot de matrices) et borne ses valeurs propres."""
    G = _as_finite_array(matrix, "matrix")
    if G.ndim < 2 or G.shape[-1] != G.shape[-2]:
        raise ValueError("La matrice doit être carrée.")
    G = 0.5 * (G + np.swapaxes(G, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, float(eig_floor))
    G_spd = (eigvecs * eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    return 0.5 * (G_spd + np.swapaxes(G_spd, -1, -2))


def fit_local_metric_at_anchor(
    displacements: np.ndarray,
    jnd_thresholds: np.ndarray,
    G_prior: Optional[np.ndarray] = None,
    shrink_lambda: float = 0.25,
    reg: float = 1e-3,
    max_iter: int = 2000,
) -> np.ndarray:
    """
    Estime une métrique locale SPD G autour d'une ancre.

    Pour chaque direction v_i et seuil u_i : u_i^2 * v_i.T @ G @ v_i ≈ 1.
    Seule la direction des déplacements est utilisée.

    Procédure (le shrinkage n'est appliqué qu'UNE fois) :
      1. G_local minimise l'écart aux données + reg * ||G_local - G_prior||^2 ;
      2. G_final = (1 - shrink_lambda) * G_local + shrink_lambda * G_prior.
    """
    dx = _as_finite_array(displacements, "displacements")
    u = _as_finite_array(jnd_thresholds, "jnd_thresholds")

    if dx.ndim != 2:
        raise ValueError("displacements doit être une matrice 2D.")
    if u.ndim != 1:
        raise ValueError("jnd_thresholds doit être un vecteur 1D.")
    if len(dx) != len(u):
        raise ValueError(
            "displacements et jnd_thresholds doivent avoir le même nombre de lignes."
        )
    if len(dx) < 2:
        raise ValueError("Au moins deux sondes directionnelles sont nécessaires.")
    if np.any(u <= 0.0):
        raise ValueError("Tous les seuils JND doivent être strictement positifs.")

    d = dx.shape[1]

    if G_prior is None:
        G_prior_arr = np.eye(d, dtype=float)
    else:
        G_prior_arr = _ensure_spd(G_prior, eig_floor=1e-10)
        if G_prior_arr.shape != (d, d):
            raise ValueError(
                "G_prior doit avoir la même dimension que les déplacements."
            )

    lam = float(np.clip(shrink_lambda, 0.0, 1.0))
    reg = max(float(reg), 0.0)

    norms = np.linalg.norm(dx, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Un déplacement nul ne définit pas une direction.")
    directions = dx / norms

    target_quadratic = 1.0 / np.maximum(u ** 2, 1e-12)
    tril_i, tril_j = np.tril_indices(d)

    def unpack(theta: np.ndarray) -> np.ndarray:
        A = np.zeros((d, d), dtype=float)
        A[tril_i, tril_j] = theta
        G = A @ A.T + 1e-10 * np.eye(d)
        return 0.5 * (G + G.T)

    def objective(theta: np.ndarray) -> float:
        G_local = unpack(theta)
        quad = np.einsum("ni,ij,nj->n", directions, G_local, directions)
        data_loss = float(np.mean((quad - target_quadratic) ** 2))
        prior_loss = float(reg * np.sum((G_local - G_prior_arr) ** 2))
        return data_loss + prior_loss

    theta0 = np.zeros(len(tril_i), dtype=float)
    theta0[tril_i == tril_j] = 1.0

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )

    # Un arrêt sans « success » (ex. perte de précision en ligne de recherche)
    # donne souvent une solution utilisable : on avertit au lieu d'échouer,
    # et on ne lève une erreur que si la solution est inutilisable.
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Échec de l'estimation de G locale : {result.message}"
        )
    if not result.success:
        warnings.warn(
            f"L-BFGS-B n'a pas déclaré de convergence : {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    G_local_hat = unpack(result.x)
    G_final = (1.0 - lam) * G_local_hat + lam * G_prior_arr
    return _ensure_spd(G_final, eig_floor=1e-10)


def interpolate_local_metric_batch(
    x_query: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float = 1.0,
    eig_floor: float = 1e-10,
) -> np.ndarray:
    """
    Interpole G(x) pour un lot de points, de forme (m, d) -> (m, d, d).

    Poids gaussiens normalisés par soustraction du maximum (log-sum-exp) :
    pas de repli arbitraire vers des poids uniformes quand tous les poids
    sont minuscules, l'ancre la plus proche domine simplement.
    """
    xq = _as_finite_array(x_query, "x_query")
    anchors = _as_finite_array(anchor_points, "anchor_points")
    metrics = _as_finite_array(anchor_metrics, "anchor_metrics")

    if xq.ndim != 2:
        raise ValueError("x_query doit être une matrice 2D (m, d).")
    if anchors.ndim != 2:
        raise ValueError("anchor_points doit être une matrice 2D.")
    if metrics.ndim != 3:
        raise ValueError("anchor_metrics doit être un tableau 3D.")

    n_anchors, dim = anchors.shape
    if len(metrics) != n_anchors:
        raise ValueError("Le nombre d'ancres et de métriques doit coïncider.")
    if xq.shape[1] != dim:
        raise ValueError("x_query et anchor_points doivent avoir la même dimension.")
    if metrics.shape[1:] != (dim, dim):
        raise ValueError(
            "Chaque métrique doit être carrée et compatible avec les ancres."
        )

    bandwidth = float(bandwidth)
    if bandwidth <= 0.0:
        raise ValueError("bandwidth doit être strictement positif.")

    d2 = np.sum((xq[:, None, :] - anchors[None, :, :]) ** 2, axis=2)  # (m, n)
    logw = -0.5 * d2 / (bandwidth ** 2)
    logw -= np.max(logw, axis=1, keepdims=True)
    weights = np.exp(logw)
    weights /= np.sum(weights, axis=1, keepdims=True)

    G = np.einsum("ma,aij->mij", weights, metrics)
    return _ensure_spd(G, eig_floor=eig_floor)


def interpolate_local_metric(
    x_query: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float = 1.0,
    eig_floor: float = 1e-10,
) -> np.ndarray:
    """Interpole G(x) en un seul point x (vecteur 1D) -> matrice (d, d)."""
    xq = _as_finite_array(x_query, "x_query")
    if xq.ndim != 1:
        raise ValueError("x_query doit être un vecteur 1D.")
    return interpolate_local_metric_batch(
        xq[None, :], anchor_points, anchor_metrics, bandwidth, eig_floor
    )[0]


def path_integrated_distance(
    waypoints: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    n_substeps: int = 20,
    bandwidth: float = 1.0,
) -> float:
    """
    Longueur métrique d'un chemin polygonal (vectorisée).

    Pour chaque sous-segment : dl = sqrt(dx.T @ G(milieu) @ dx).
    Calcule la longueur du chemin fourni ; ne cherche pas la géodésique.
    """
    pts = _as_finite_array(waypoints, "waypoints")
    if pts.ndim != 2:
        raise ValueError("waypoints doit être une matrice 2D.")
    if len(pts) < 2:
        return 0.0

    n_substeps = max(int(n_substeps), 1)
    ts = np.linspace(0.0, 1.0, n_substeps + 1)

    p0 = pts[:-1, None, :]
    p1 = pts[1:, None, :]
    sub = p0 * (1.0 - ts[None, :, None]) + p1 * ts[None, :, None]  # (S, T+1, d)

    left = sub[:, :-1, :].reshape(-1, pts.shape[1])
    right = sub[:, 1:, :].reshape(-1, pts.shape[1])
    mid = 0.5 * (left + right)
    delta = right - left

    G_mid = interpolate_local_metric_batch(
        mid, anchor_points, anchor_metrics, bandwidth=bandwidth
    )
    quad = np.einsum("mi,mij,mj->m", delta, G_mid, delta)
    return float(np.sum(np.sqrt(np.maximum(quad, 0.0))))


def local_volume_in_region(
    region_samples: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    region_base_volume: float,
    bandwidth: float = 1.0,
) -> Dict[str, float]:
    """
    Estime V_psi(R) = ∫_R sqrt(det G(x)) dx par Monte-Carlo (vectorisé).

    `region_samples` doit échantillonner R uniformément (ou de façon
    explicitement pondérée).
    """
    pts = _as_finite_array(region_samples, "region_samples")
    if pts.ndim != 2 or len(pts) == 0:
        raise ValueError("region_samples doit être une matrice non vide.")

    region_base_volume = float(region_base_volume)
    if region_base_volume <= 0.0:
        raise ValueError("region_base_volume doit être strictement positif.")

    G = interpolate_local_metric_batch(
        pts, anchor_points, anchor_metrics, bandwidth=bandwidth
    )
    sign_det, log_det = np.linalg.slogdet(G)
    if np.any(sign_det <= 0.0) or not np.all(np.isfinite(log_det)):
        raise RuntimeError(
            "Métrique locale non définie positive lors du calcul du volume."
        )

    densities = np.exp(0.5 * log_det)
    n = len(densities)
    mean_density = float(np.mean(densities))
    sd = float(np.std(densities, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)

    return {
        "V_psi_region": float(region_base_volume * mean_density),
        "mean_sqrt_det_G": mean_density,
        "sd_sqrt_det_G": sd,
        "mc_se_sqrt_det_G": float(se),
        "n_samples": int(n),
    }


def procrustes_align_bridges(
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    allow_reflection: bool = True,
    estimate_scale: bool = True,
    return_raw_distances: bool = False,
) -> Dict[str, object]:
    """
    Aligne coords_source sur coords_target (translation, rotation,
    réflexion éventuelle, échelle scalaire éventuelle).

    Les lignes des deux tableaux correspondent aux mêmes stimuli-ponts.
    """
    src = _as_finite_array(coords_source, "coords_source")
    tgt = _as_finite_array(coords_target, "coords_target")

    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError("Les deux arrangements doivent être des matrices 2D.")
    if src.shape[0] != tgt.shape[0]:
        raise ValueError("Les arrangements doivent contenir le même nombre de ponts.")
    if src.shape[0] < 2:
        raise ValueError("Au moins deux stimuli-ponts sont nécessaires.")

    n_points = src.shape[0]
    dimension = max(src.shape[1], tgt.shape[1])

    src_pad = np.zeros((n_points, dimension))
    tgt_pad = np.zeros((n_points, dimension))
    src_pad[:, : src.shape[1]] = src
    tgt_pad[:, : tgt.shape[1]] = tgt

    mean_src = src_pad.mean(axis=0)
    mean_tgt = tgt_pad.mean(axis=0)
    src_c = src_pad - mean_src
    tgt_c = tgt_pad - mean_tgt

    source_norm_sq = float(np.sum(src_c ** 2))
    target_norm_sq = float(np.sum(tgt_c ** 2))
    if source_norm_sq <= 1e-15:
        raise ValueError("L'arrangement source est dégénéré.")

    U, S, Vt = np.linalg.svd(src_c.T @ tgt_c, full_matrices=True)
    R = U @ Vt
    trace_term = float(np.sum(S))

    # Rotation propre imposée : on inverse l'axe associé à la plus petite
    # valeur singulière. Le test det(R) < 0 et le trace_term corrigé sont
    # évalués AVANT toute modification, de sorte que l'échelle est cohérente.
    if not allow_reflection and np.linalg.det(R) < 0.0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = U @ Vt
        trace_term = float(np.sum(S[:-1]) - S[-1])

    scale = trace_term / source_norm_sq if estimate_scale else 1.0

    aligned_centered = scale * (src_c @ R)
    aligned = aligned_centered + mean_tgt

    residual_sq = float(np.sum((aligned_centered - tgt_c) ** 2))
    disparity = math.sqrt(residual_sq / max(target_norm_sq, 1e-15))

    output: Dict[str, object] = {
        "aligned_coords": aligned[:, : tgt.shape[1]],
        "translation_source": mean_src[: src.shape[1]].tolist(),
        "translation_target": mean_tgt[: tgt.shape[1]].tolist(),
        "rotation_matrix": R.tolist(),
        "scale_factor": float(scale),
        "procrustes_disparity": float(disparity),
        "allow_reflection": bool(allow_reflection),
        "estimate_scale": bool(estimate_scale),
    }

    if return_raw_distances:
        output["raw_dist_matrix_source"] = squareform(pdist(src, metric="euclidean"))
        output["aligned_dist_matrix_source"] = squareform(
            pdist(output["aligned_coords"], metric="euclidean")
        )
        output["dist_matrix_target"] = squareform(pdist(tgt, metric="euclidean"))

    return output


def fit_linear_scale(
    d_raw: np.ndarray,
    d_ref: np.ndarray,
    fix_intercept_zero: bool = True,
) -> Dict[str, float | str]:
    """Ajuste d_ref = a + b * d_raw (a = 0 par défaut)."""
    x = _as_finite_array(d_raw, "d_raw").reshape(-1)
    y = _as_finite_array(d_ref, "d_ref").reshape(-1)

    if len(x) != len(y):
        raise ValueError("d_raw et d_ref doivent avoir la même longueur.")
    if len(x) < 2:
        raise ValueError("Au moins deux observations sont nécessaires.")
    if np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError("Les distances doivent être non négatives.")

    if fix_intercept_zero:
        denominator = float(np.dot(x, x))
        if denominator <= 1e-15:
            raise ValueError("Les distances brutes ont une variance insuffisante.")
        return {
            "type": "linear_zero_intercept",
            "a": 0.0,
            "b": float(np.dot(x, y) / denominator),
        }

    design = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return {"type": "linear_affine", "a": float(coef[0]), "b": float(coef[1])}


def apply_linear_scale(d_raw: np.ndarray, model: Dict[str, float | str]) -> np.ndarray:
    """Applique une calibration linéaire."""
    x = _as_finite_array(d_raw, "d_raw")
    return np.maximum(float(model["a"]) + float(model["b"]) * x, 0.0)


def _pool_adjacent_violators(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Projection isotone croissante (PAVA)."""
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(y) != len(w):
        raise ValueError("values et weights doivent avoir la même longueur.")

    blocks = []
    for idx in range(len(y)):
        blocks.append(
            {"start": idx, "end": idx, "weight": float(w[idx]), "mean": float(y[idx])}
        )
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            tw = left["weight"] + right["weight"]
            mean = (
                left["weight"] * left["mean"] + right["weight"] * right["mean"]
            ) / max(tw, 1e-15)
            blocks.append(
                {"start": left["start"], "end": right["end"], "weight": tw, "mean": mean}
            )

    fitted = np.empty(len(y), dtype=float)
    for b in blocks:
        fitted[b["start"] : b["end"] + 1] = b["mean"]
    return fitted


def fit_monotone_binned(
    d_raw: np.ndarray,
    d_ref: np.ndarray,
    n_bins: int = 8,
) -> Dict[str, object]:
    """
    Transformation monotone croissante par morceaux : regroupement par
    quantiles de d_raw, puis projection isotone des moyennes de d_ref.
    """
    x = _as_finite_array(d_raw, "d_raw").reshape(-1)
    y = _as_finite_array(d_ref, "d_ref").reshape(-1)

    if len(x) != len(y):
        raise ValueError("d_raw et d_ref doivent avoir la même longueur.")
    if len(x) < 4:
        raise ValueError("Au moins quatre observations sont nécessaires.")
    if np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError("Les distances doivent être non négatives.")

    n_bins = int(np.clip(n_bins, 2, len(x)))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))

    if len(edges) < 3:
        m = float(np.mean(y))
        return {
            "type": "monotone_constant",
            "edges": [float(np.min(x)), float(np.max(x))],
            "values": [m, m],
        }

    bin_idx = np.digitize(x, edges[1:-1], right=False)

    xc, ym, ws = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        xc.append(float(np.mean(x[mask])))
        ym.append(float(np.mean(y[mask])))
        ws.append(float(np.sum(mask)))

    xc, ym, ws = np.asarray(xc), np.asarray(ym), np.asarray(ws)
    order = np.argsort(xc)
    xc, ym, ws = xc[order], ym[order], ws[order]

    return {
        "type": "monotone_piecewise",
        "edges": xc.tolist(),
        "values": _pool_adjacent_violators(ym, ws).tolist(),
    }


def apply_monotone_binned(d_raw: np.ndarray, model: Dict[str, object]) -> np.ndarray:
    """Interpolation linéaire ; valeurs hors calibration tronquées aux extrémités."""
    x = _as_finite_array(d_raw, "d_raw")
    values = np.asarray(model.get("values", []), dtype=float)
    if len(values) == 0:
        raise ValueError("Modèle monotone vide.")

    if str(model.get("type", "")) == "monotone_constant" or len(values) == 1:
        return np.full_like(x, float(values[0]), dtype=float)

    edges = np.asarray(model.get("edges", []), dtype=float)
    if len(edges) != len(values):
        raise ValueError("edges et values doivent avoir la même longueur.")

    return np.interp(x, edges, values, left=float(values[0]), right=float(values[-1]))


def evaluate_calibration_split(
    d_raw_test: np.ndarray,
    d_ref_test: np.ndarray,
    pred_test: np.ndarray,
) -> Dict[str, float]:
    """Évalue une calibration sur un jeu indépendant."""
    raw = _as_finite_array(d_raw_test, "d_raw_test").reshape(-1)
    true = _as_finite_array(d_ref_test, "d_ref_test").reshape(-1)
    pred = _as_finite_array(pred_test, "pred_test").reshape(-1)

    if not (len(raw) == len(true) == len(pred)):
        raise ValueError("Les trois vecteurs doivent avoir la même longueur.")
    if len(true) < 2:
        raise ValueError("Au moins deux observations sont nécessaires.")

    error = pred - true
    var_pred = float(np.var(pred))
    if var_pred > 1e-12:
        slope = float(np.cov(pred, true, bias=True)[0, 1]) / var_pred
    else:
        slope = 0.0

    # [0] plutôt que .statistic : compatible avec toutes les versions de SciPy.
    rho = float(spearmanr(true, pred)[0])
    if not np.isfinite(rho):
        rho = 0.0

    return {
        "slope_pred_vs_ref": float(slope),
        "mean_bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error ** 2))),
        "spearman": rho,
    }


def calibrate_and_validate_bridges(
    d_raw_pairs: np.ndarray,
    d_ref_pairs: np.ndarray,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Calibration de compatibilité sur des paires déjà extraites.

    Les ensembles calibration / validation / test sont disjoints et le test
    contient toujours au moins deux paires (aucun repli sur la validation).
    Pour une analyse confirmatoire, séparer les stimuli-ponts par nœuds
    AVANT de former les paires, afin qu'un nœud n'apparaisse pas dans
    plusieurs ensembles.
    """
    raw = _as_finite_array(d_raw_pairs, "d_raw_pairs").reshape(-1)
    ref = _as_finite_array(d_ref_pairs, "d_ref_pairs").reshape(-1)

    if len(raw) != len(ref):
        raise ValueError("d_raw_pairs et d_ref_pairs doivent avoir la même longueur.")
    n = len(raw)
    if n < 8:
        raise ValueError("Au moins huit paires sont requises.")

    n_val = max(int(round(0.20 * n)), 2)
    n_cal = max(int(round(0.60 * n)), 4)
    n_test = n - n_cal - n_val
    while n_test < 2 and n_cal > 4:
        n_cal -= 1
        n_test += 1
    while n_test < 2 and n_val > 2:
        n_val -= 1
        n_test += 1
    if n_test < 2:
        raise ValueError("Découpage impossible : trop peu de paires.")

    perm = np.random.default_rng(int(seed)).permutation(n)
    idx_cal = perm[:n_cal]
    idx_val = perm[n_cal : n_cal + n_val]
    idx_test = perm[n_cal + n_val :]

    linear_model = fit_linear_scale(raw[idx_cal], ref[idx_cal], fix_intercept_zero=True)
    monotone_model = fit_monotone_binned(
        raw[idx_cal], ref[idx_cal], n_bins=min(6, max(2, len(idx_cal) // 4))
    )

    val_linear = evaluate_calibration_split(
        raw[idx_val], ref[idx_val], apply_linear_scale(raw[idx_val], linear_model)
    )
    val_monotone = evaluate_calibration_split(
        raw[idx_val], ref[idx_val], apply_monotone_binned(raw[idx_val], monotone_model)
    )

    if val_monotone["rmse"] < val_linear["rmse"]:
        selected_name, selected_model = "monotone_piecewise", monotone_model
        pred_test = apply_monotone_binned(raw[idx_test], monotone_model)
    else:
        selected_name, selected_model = "linear_zero_intercept", linear_model
        pred_test = apply_linear_scale(raw[idx_test], linear_model)

    test_metrics = evaluate_calibration_split(raw[idx_test], ref[idx_test], pred_test)

    non_equivalence = bool(
        abs(test_metrics["slope_pred_vs_ref"] - 1.0) > 0.25
        or test_metrics["spearman"] < 0.50
    )

    return {
        "selected_model": selected_name,
        "selected_parameters": selected_model,
        "val_linear_metrics": val_linear,
        "val_monotone_metrics": val_monotone,
        "test_metrics": test_metrics,
        "non_equivalence_warning": non_equivalence,
        "indices": {
            "calibration": idx_cal.tolist(),
            "validation": idx_val.tolist(),
            "test": idx_test.tolist(),
        },
    }


if __name__ == "__main__":
    rng = np.random.default_rng(2026)

    reference = np.array(
        [[-1.2, -0.4], [-0.5, 0.7], [0.1, -0.9], [0.6, 0.3], [1.1, -0.2], [1.5, 0.9]]
    )
    angle = 0.65
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    source = (
        2.4 * (reference @ rotation.T)
        + np.array([4.0, -2.5])
        + 0.05 * rng.standard_normal(reference.shape)
    )

    proc = procrustes_align_bridges(
        source, reference, allow_reflection=False, estimate_scale=True,
        return_raw_distances=True,
    )
    print("Facteur d'échelle :", round(proc["scale_factor"], 6))
    print("Disparité Procrustes :", round(proc["procrustes_disparity"], 6))

    # Test du cas réfléchi avec réflexion interdite (ancien bogue)
    mirrored = source * np.array([1.0, -1.0])
    p_nr = procrustes_align_bridges(mirrored, reference, allow_reflection=False)
    p_r = procrustes_align_bridges(mirrored, reference, allow_reflection=True)
    print("Miroir, sans réflexion : échelle", round(p_nr["scale_factor"], 4),
          "disparité", round(p_nr["procrustes_disparity"], 4))
    print("Miroir, avec réflexion : échelle", round(p_r["scale_factor"], 4),
          "disparité", round(p_r["procrustes_disparity"], 4))

    anchor_points = np.array([[-1.0, -0.5], [0.0, 0.2], [1.2, 0.8]])
    anchor_metrics = np.array(
        [[[2.0, 0.0], [0.0, 1.5]], [[1.5, 0.2], [0.2, 1.8]], [[3.0, 0.0], [0.0, 2.5]]]
    )

    dist = path_integrated_distance(anchor_points, anchor_points, anchor_metrics)
    print("Distance intégrée :", round(dist, 6))

    samples = rng.uniform(-1.5, 1.5, size=(500, 2))
    vol = local_volume_in_region(samples, anchor_points, anchor_metrics, 9.0)
    print("Volume local :", {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in vol.items()})

    raw = rng.uniform(0.2, 3.0, 20)
    ref = 1.7 * raw + 0.05 * rng.standard_normal(20)
    res = calibrate_and_validate_bridges(raw, np.abs(ref))
    print("Modèle :", res["selected_model"], "| test :",
          {k: round(v, 3) for k, v in res["test_metrics"].items()})
    sizes = {k: len(v) for k, v in res["indices"].items()}
    print("Tailles :", sizes)
