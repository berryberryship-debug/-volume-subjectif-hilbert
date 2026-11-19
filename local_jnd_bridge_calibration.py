#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration indépendante des stimuli-ponts.

Fonctions principales :

- alignement Procrustes avec translation, rotation et échelle ;
- estimation d'une métrique locale SPD G(x) avec shrinkage ;
- interpolation du champ métrique local ;
- calcul de distances intégrées ;
- calcul de volumes locaux ;
- calibration linéaire ou monotone des distances ;
- validation sur des paires de stimuli-ponts.

Les données de calibration doivent rester indépendantes des données utilisées
pour ajuster la métrique principale W_theta.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

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
    "path_integrated_distance",
    "local_volume_in_region",
    "procrustes_align_bridges",
    "calibrate_and_validate_bridges",
]


def _as_finite_array(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Convertit une entrée en tableau float et vérifie sa finitude.
    """
    arr = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contient des valeurs non finies."
        )

    return arr


def _ensure_spd(
    matrix: np.ndarray,
    eig_floor: float = 1e-10,
) -> np.ndarray:
    """
    Symétrise une matrice et lui impose une petite valeur propre minimale.
    """
    G = _as_finite_array(matrix, "matrix")

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(
            "La matrice doit être carrée."
        )

    G = 0.5 * (G + G.T)

    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(
        eigvals,
        float(eig_floor),
    )

    G_spd = (
        eigvecs
        @ np.diag(eigvals)
        @ eigvecs.T
    )

    return 0.5 * (G_spd + G_spd.T)


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

    Pour chaque direction v_i et seuil local u_i, on impose approximativement :

        u_i^2 * v_i.T @ G @ v_i = 1

    Les déplacements peuvent être fournis sous la forme :

        displacement_i = u_i * v_i

    ou sous toute autre échelle ; seule leur direction est utilisée.

    Parameters
    ----------
    displacements
        Tableau de forme (n_probes, d).

    jnd_thresholds
        Seuils JND positifs de forme (n_probes,).

    G_prior
        Métrique SPD de référence.

    shrink_lambda
        Poids du shrinkage vers G_prior, compris entre 0 et 1.

    reg
        Régularisation de la métrique locale.

    Returns
    -------
    G_final
        Matrice SPD de forme (d, d).
    """
    dx = _as_finite_array(
        displacements,
        "displacements",
    )

    u = _as_finite_array(
        jnd_thresholds,
        "jnd_thresholds",
    )

    if dx.ndim != 2:
        raise ValueError(
            "displacements doit être une matrice 2D."
        )

    if u.ndim != 1:
        raise ValueError(
            "jnd_thresholds doit être un vecteur 1D."
        )

    if len(dx) != len(u):
        raise ValueError(
            "displacements et jnd_thresholds doivent "
            "avoir le même nombre de lignes."
        )

    if len(dx) < 2:
        raise ValueError(
            "Au moins deux sondes directionnelles sont nécessaires."
        )

    if np.any(u <= 0.0):
        raise ValueError(
            "Tous les seuils JND doivent être strictement positifs."
        )

    d = dx.shape[1]

    if G_prior is None:
        G_prior_arr = np.eye(d, dtype=float)
    else:
        G_prior_arr = _ensure_spd(
            G_prior,
            eig_floor=1e-10,
        )

        if G_prior_arr.shape != (d, d):
            raise ValueError(
                "G_prior doit avoir la même dimension que les déplacements."
            )

    lam = float(np.clip(shrink_lambda, 0.0, 1.0))
    reg = max(float(reg), 0.0)

    norms = np.linalg.norm(
        dx,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 1e-12):
        raise ValueError(
            "Un déplacement nul ne définit pas une direction."
        )

    directions = dx / norms

    target_quadratic = 1.0 / np.maximum(
        u ** 2,
        1e-12,
    )

    tril_i, tril_j = np.tril_indices(d)

    def unpack(theta: np.ndarray) -> np.ndarray:
        """
        Paramétrisation SPD par facteur de Cholesky.
        """
        A = np.zeros((d, d), dtype=float)
        A[tril_i, tril_j] = theta

        G = A @ A.T
        G += 1e-10 * np.eye(d)

        return 0.5 * (G + G.T)

    def objective(theta: np.ndarray) -> float:
        G_local = unpack(theta)

        G_shrunk = (
            (1.0 - lam) * G_local
            + lam * G_prior_arr
        )

        quadratic_values = np.einsum(
            "ni,ij,nj->n",
            directions,
            G_shrunk,
            directions,
        )

        data_loss = float(
            np.mean(
                (quadratic_values - target_quadratic) ** 2
            )
        )

        prior_loss = float(
            reg * np.sum(
                (G_local - G_prior_arr) ** 2
            )
        )

        return data_loss + prior_loss

    theta0 = np.zeros(
        len(tril_i),
        dtype=float,
    )

    diagonal_mask = tril_i == tril_j
    theta0[diagonal_mask] = 1.0

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Échec de l'estimation de G locale : "
            f"{result.message}"
        )

    G_local_hat = unpack(result.x)

    G_final = (
        (1.0 - lam) * G_local_hat
        + lam * G_prior_arr
    )

    return _ensure_spd(
        G_final,
        eig_floor=1e-10,
    )


def interpolate_local_metric(
    x_query: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float = 1.0,
    eig_floor: float = 1e-10,
) -> np.ndarray:
    """
    Interpole G(x) par moyenne pondérée des métriques aux ancres.

    Les poids sont gaussiens dans l'espace des ancres :

        w_i(x) proportional to exp(-||x-a_i||^2 / (2h^2))

    La moyenne convexe de matrices SPD reste SPD ; une correction numérique
    est néanmoins appliquée à la fin.
    """
    xq = _as_finite_array(
        x_query,
        "x_query",
    )

    anchors = _as_finite_array(
        anchor_points,
        "anchor_points",
    )

    metrics = _as_finite_array(
        anchor_metrics,
        "anchor_metrics",
    )

    if xq.ndim != 1:
        raise ValueError(
            "x_query doit être un vecteur 1D."
        )

    if anchors.ndim != 2:
        raise ValueError(
            "anchor_points doit être une matrice 2D."
        )

    if metrics.ndim != 3:
        raise ValueError(
            "anchor_metrics doit être un tableau 3D."
        )

    n_anchors, dim = anchors.shape

    if len(metrics) != n_anchors:
        raise ValueError(
            "Le nombre d'ancres et de métriques doit coïncider."
        )

    if len(xq) != dim:
        raise ValueError(
            "x_query et anchor_points doivent avoir la même dimension."
        )

    if metrics.shape[1:] != (dim, dim):
        raise ValueError(
            "Chaque métrique doit être carrée et compatible avec les ancres."
        )

    bandwidth = float(bandwidth)

    if bandwidth <= 0.0:
        raise ValueError(
            "bandwidth doit être strictement positif."
        )

    distances_squared = np.sum(
        (anchors - xq[None, :]) ** 2,
        axis=1,
    )

    weights = np.exp(
        -0.5 * distances_squared / (bandwidth ** 2)
    )

    weight_sum = float(np.sum(weights))

    if weight_sum <= 1e-15:
        weights = np.full(
            n_anchors,
            1.0 / n_anchors,
            dtype=float,
        )
    else:
        weights = weights / weight_sum

    G_interp = np.einsum(
        "a,aij->ij",
        weights,
        metrics,
    )

    return _ensure_spd(
        G_interp,
        eig_floor=eig_floor,
    )


def path_integrated_distance(
    waypoints: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    n_substeps: int = 20,
    bandwidth: float = 1.0,
) -> float:
    """
    Calcule la longueur métrique d'un chemin polygonal.

    Pour chaque sous-segment :

        dl = sqrt(dx.T @ G(midpoint) @ dx)

    La fonction calcule la distance du chemin fourni.
    Elle ne recherche pas automatiquement le chemin géodésique minimal.
    """
    pts = _as_finite_array(
        waypoints,
        "waypoints",
    )

    if pts.ndim != 2:
        raise ValueError(
            "waypoints doit être une matrice 2D."
        )

    if len(pts) < 2:
        return 0.0

    n_substeps = max(int(n_substeps), 1)
    total_distance = 0.0

    for segment_idx in range(len(pts) - 1):
        p0 = pts[segment_idx]
        p1 = pts[segment_idx + 1]

        ts = np.linspace(
            0.0,
            1.0,
            n_substeps + 1,
        )

        subpoints = (
            p0[None, :] * (1.0 - ts[:, None])
            + p1[None, :] * ts[:, None]
        )

        for step_idx in range(n_substeps):
            left = subpoints[step_idx]
            right = subpoints[step_idx + 1]

            midpoint = 0.5 * (left + right)
            delta = right - left

            G_mid = interpolate_local_metric(
                midpoint,
                anchor_points,
                anchor_metrics,
                bandwidth=bandwidth,
            )

            quadratic_form = float(
                delta.T @ G_mid @ delta
            )

            total_distance += math.sqrt(
                max(quadratic_form, 0.0)
            )

    return float(total_distance)


def local_volume_in_region(
    region_samples: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    region_base_volume: float,
    bandwidth: float = 1.0,
) -> Dict[str, float]:
    """
    Estime :

        V_psi(R) = integral_R sqrt(det(G(x))) dx

    par Monte-Carlo ou quadrature empirique.

    `region_samples` doit représenter un échantillonnage uniforme ou
    explicitement pondéré de la région R.
    """
    pts = _as_finite_array(
        region_samples,
        "region_samples",
    )

    if pts.ndim != 2 or len(pts) == 0:
        raise ValueError(
            "region_samples doit être une matrice non vide."
        )

    region_base_volume = float(region_base_volume)

    if region_base_volume <= 0.0:
        raise ValueError(
            "region_base_volume doit être strictement positif."
        )

    densities = np.empty(
        len(pts),
        dtype=float,
    )

    for idx, point in enumerate(pts):
        G_x = interpolate_local_metric(
            point,
            anchor_points,
            anchor_metrics,
            bandwidth=bandwidth,
        )

        sign_det, log_det = np.linalg.slogdet(G_x)

        if sign_det <= 0.0 or not np.isfinite(log_det):
            raise RuntimeError(
                "Métrique locale non définie positive lors du calcul du volume."
            )

        densities[idx] = math.exp(
            0.5 * log_det
        )

    mean_density = float(
        np.mean(densities)
    )

    standard_deviation = float(
        np.std(
            densities,
            ddof=1,
        )
    ) if len(densities) > 1 else 0.0

    standard_error = standard_deviation / math.sqrt(
        max(len(densities), 1)
    )

    return {
        "V_psi_region": float(
            region_base_volume * mean_density
        ),
        "mean_sqrt_det_G": mean_density,
        "sd_sqrt_det_G": standard_deviation,
        "mc_se_sqrt_det_G": float(standard_error),
        "n_samples": int(len(densities)),
    }


def procrustes_align_bridges(
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    allow_reflection: bool = True,
    estimate_scale: bool = True,
    return_raw_distances: bool = False,
) -> Dict[str, object]:
    """
    Aligne les coordonnées source sur les coordonnées cibles.

    Étapes :

    1. retrait de la translation ;
    2. rotation orthogonale ;
    3. réflexion éventuelle ;
    4. estimation éventuelle d'une échelle scalaire.

    Les lignes de coords_source et coords_target doivent correspondre
    aux mêmes stimuli-ponts.
    """
    src = _as_finite_array(
        coords_source,
        "coords_source",
    )

    tgt = _as_finite_array(
        coords_target,
        "coords_target",
    )

    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError(
            "Les deux arrangements doivent être des matrices 2D."
        )

    if src.shape[0] != tgt.shape[0]:
        raise ValueError(
            "Les arrangements doivent contenir le même nombre de ponts."
        )

    if src.shape[0] < 2:
        raise ValueError(
            "Au moins deux stimuli-ponts sont nécessaires."
        )

    n_points = src.shape[0]
    dimension = max(
        src.shape[1],
        tgt.shape[1],
    )

    src_pad = np.zeros(
        (n_points, dimension),
        dtype=float,
    )

    tgt_pad = np.zeros(
        (n_points, dimension),
        dtype=float,
    )

    src_pad[:, :src.shape[1]] = src
    tgt_pad[:, :tgt.shape[1]] = tgt

    mean_src = np.mean(
        src_pad,
        axis=0,
    )

    mean_tgt = np.mean(
        tgt_pad,
        axis=0,
    )

    src_centered = src_pad - mean_src
    tgt_centered = tgt_pad - mean_tgt

    source_norm_squared = float(
        np.sum(src_centered ** 2)
    )

    target_norm_squared = float(
        np.sum(tgt_centered ** 2)
    )

    if source_norm_squared <= 1e-15:
        raise ValueError(
            "L'arrangement source est dégénéré."
        )

    cross_covariance = (
        src_centered.T @ tgt_centered
    )

    U, singular_values, Vt = np.linalg.svd(
        cross_covariance,
        full_matrices=True,
    )

    R = U @ Vt

    if (
        not allow_reflection
        and np.linalg.det(R) < 0.0
    ):
        U[:, -1] *= -1.0
        R = U @ Vt

    trace_term = float(
        np.sum(
            np.abs(singular_values)
        )
    )

    if (
        not allow_reflection
        and np.linalg.det(R) < 0.0
    ):
        trace_term = float(
            np.sum(singular_values[:-1])
            - singular_values[-1]
        )

    if estimate_scale:
        scale = trace_term / source_norm_squared
    else:
        scale = 1.0

    aligned_centered = (
        scale
        * (src_centered @ R)
    )

    aligned = aligned_centered + mean_tgt

    residual_squared = float(
        np.sum(
            (aligned_centered - tgt_centered) ** 2
        )
    )

    disparity = math.sqrt(
        residual_squared
        / max(target_norm_squared, 1e-15)
    )

    output: Dict[str, object] = {
        "aligned_coords": aligned[:, :tgt.shape[1]],
        "translation_source": mean_src[:src.shape[1]].tolist(),
        "translation_target": mean_tgt[:tgt.shape[1]].tolist(),
        "rotation_matrix": R.tolist(),
        "scale_factor": float(scale),
        "procrustes_disparity": float(disparity),
        "allow_reflection": bool(allow_reflection),
        "estimate_scale": bool(estimate_scale),
    }

    if return_raw_distances:
        output["raw_dist_matrix_source"] = squareform(
            pdist(src, metric="euclidean")
        )

        output["aligned_dist_matrix_source"] = squareform(
            pdist(
                output["aligned_coords"],
                metric="euclidean",
            )
        )

        output["dist_matrix_target"] = squareform(
            pdist(tgt, metric="euclidean")
        )

    return output


def fit_linear_scale(
    d_raw: np.ndarray,
    d_ref: np.ndarray,
    fix_intercept_zero: bool = True,
) -> Dict[str, float | str]:
    """
    Ajuste :

        d_ref = a + b * d_raw

    avec a fixé à zéro par défaut.
    """
    x = _as_finite_array(
        d_raw,
        "d_raw",
    ).reshape(-1)

    y = _as_finite_array(
        d_ref,
        "d_ref",
    ).reshape(-1)

    if len(x) != len(y):
        raise ValueError(
            "d_raw et d_ref doivent avoir la même longueur."
        )

    if len(x) < 2:
        raise ValueError(
            "Au moins deux observations sont nécessaires."
        )

    if np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError(
            "Les distances doivent être non négatives."
        )

    if fix_intercept_zero:
        denominator = float(
            np.dot(x, x)
        )

        if denominator <= 1e-15:
            raise ValueError(
                "Les distances brutes ont une variance insuffisante."
            )

        slope = float(
            np.dot(x, y) / denominator
        )

        return {
            "type": "linear_zero_intercept",
            "a": 0.0,
            "b": slope,
        }

    design = np.column_stack(
        [
            np.ones_like(x),
            x,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    return {
        "type": "linear_affine",
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
    }


def apply_linear_scale(
    d_raw: np.ndarray,
    model: Dict[str, float | str],
) -> np.ndarray:
    """
    Applique une calibration linéaire.
    """
    x = _as_finite_array(
        d_raw,
        "d_raw",
    )

    a = float(model["a"])
    b = float(model["b"])

    result = a + b * x

    return np.maximum(
        result,
        0.0,
    )


def _pool_adjacent_violators(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Projection isotone croissante par l'algorithme PAVA.
    """
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    if len(y) != len(w):
        raise ValueError(
            "values et weights doivent avoir la même longueur."
        )

    blocks = []

    for idx in range(len(y)):
        blocks.append(
            {
                "start": idx,
                "end": idx,
                "weight": float(w[idx]),
                "mean": float(y[idx]),
            }
        )

        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]

            if left["mean"] <= right["mean"]:
                break

            total_weight = (
                left["weight"]
                + right["weight"]
            )

            merged_mean = (
                left["weight"] * left["mean"]
                + right["weight"] * right["mean"]
            ) / max(total_weight, 1e-15)

            merged = {
                "start": left["start"],
                "end": right["end"],
                "weight": total_weight,
                "mean": merged_mean,
            }

            blocks = blocks[:-2]
            blocks.append(merged)

    fitted = np.empty(
        len(y),
        dtype=float,
    )

    for block in blocks:
        fitted[
            block["start"]:block["end"] + 1
        ] = block["mean"]

    return fitted


def fit_monotone_binned(
    d_raw: np.ndarray,
    d_ref: np.ndarray,
    n_bins: int = 8,
) -> Dict[str, object]:
    """
    Ajuste une transformation monotone croissante par morceaux.

    Les observations sont regroupées par quantiles de d_raw, puis les
    moyennes de d_ref sont projetées sur l'espace des fonctions croissantes.
    """
    x = _as_finite_array(
        d_raw,
        "d_raw",
    ).reshape(-1)

    y = _as_finite_array(
        d_ref,
        "d_ref",
    ).reshape(-1)

    if len(x) != len(y):
        raise ValueError(
            "d_raw et d_ref doivent avoir la même longueur."
        )

    if len(x) < 4:
        raise ValueError(
            "Au moins quatre observations sont nécessaires."
        )

    if np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError(
            "Les distances doivent être non négatives."
        )

    n_bins = int(
        np.clip(
            n_bins,
            2,
            len(x),
        )
    )

    quantile_edges = np.quantile(
        x,
        np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        ),
    )

    quantile_edges = np.unique(
        quantile_edges
    )

    if len(quantile_edges) < 3:
        return {
            "type": "monotone_constant",
            "edges": [
                float(np.min(x)),
                float(np.max(x)),
            ],
            "values": [
                float(np.mean(y)),
                float(np.mean(y)),
            ],
        }

    bin_indices = np.digitize(
        x,
        quantile_edges[1:-1],
        right=False,
    )

    x_centers = []
    y_means = []
    weights = []

    for bin_idx in range(len(quantile_edges) - 1):
        mask = bin_indices == bin_idx

        if not np.any(mask):
            continue

        x_centers.append(
            float(np.mean(x[mask]))
        )

        y_means.append(
            float(np.mean(y[mask]))
        )

        weights.append(
            float(np.sum(mask))
        )

    x_centers = np.asarray(
        x_centers,
        dtype=float,
    )

    y_means = np.asarray(
        y_means,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    order = np.argsort(x_centers)
    x_centers = x_centers[order]
    y_means = y_means[order]
    weights = weights[order]

    y_monotone = _pool_adjacent_violators(
        y_means,
        weights,
    )

    return {
        "type": "monotone_piecewise",
        "edges": x_centers.tolist(),
        "values": y_monotone.tolist(),
    }


def apply_monotone_binned(
    d_raw: np.ndarray,
    model: Dict[str, object],
) -> np.ndarray:
    """
    Applique une transformation monotone par interpolation linéaire.

    Les valeurs hors de l'intervalle de calibration sont tronquées aux
    extrémités connues.
    """
    x = _as_finite_array(
        d_raw,
        "d_raw",
    )

    model_type = str(
        model.get("type", "")
    )

    if model_type == "monotone_constant":
        values = np.asarray(
            model["values"],
            dtype=float,
        )

        if len(values) == 0:
            raise ValueError(
                "Modèle monotone constant vide."
            )

        return np.full_like(
            x,
            float(values[0]),
            dtype=float,
        )

    edges = np.asarray(
        model.get("edges", []),
        dtype=float,
    )

    values = np.asarray(
        model.get("values", []),
        dtype=float,
    )

    if len(edges) == 0 or len(values) == 0:
        raise ValueError(
            "Modèle monotone vide."
        )

    if len(edges) != len(values):
        raise ValueError(
            "edges et values doivent avoir la même longueur."
        )

    if len(edges) == 1:
        return np.full_like(
            x,
            float(values[0]),
            dtype=float,
        )

    return np.interp(
        x,
        edges,
        values,
        left=float(values[0]),
        right=float(values[-1]),
    )


def evaluate_calibration_split(
    d_raw_test: np.ndarray,
    d_ref_test: np.ndarray,
    pred_test: np.ndarray,
) -> Dict[str, float]:
    """
    Évalue une calibration sur un jeu indépendant.
    """
    raw = _as_finite_array(
        d_raw_test,
        "d_raw_test",
    ).reshape(-1)

    true = _as_finite_array(
        d_ref_test,
        "d_ref_test",
    ).reshape(-1)

    pred = _as_finite_array(
        pred_test,
        "pred_test",
    ).reshape(-1)

    if not (
        len(raw)
        == len(true)
        == len(pred)
    ):
        raise ValueError(
            "Les trois vecteurs doivent avoir la même longueur."
        )

    if len(true) < 2:
        raise ValueError(
            "Au moins deux observations sont nécessaires."
        )

    error = pred - true

    variance_pred = float(
        np.var(pred)
    )

    if variance_pred > 1e-12:
        covariance = float(
            np.cov(
                pred,
                true,
                bias=True,
            )[0, 1]
        )

        slope = covariance / variance_pred
    else:
        slope = 0.0

    bias = float(
        np.mean(error)
    )

    mae = float(
        np.mean(np.abs(error))
    )

    rmse = float(
        math.sqrt(
            np.mean(error ** 2)
        )
    )

    rho_result = spearmanr(
        true,
        pred,
    )

    rho = float(
        rho_result.statistic
    )

    if not np.isfinite(rho):
        rho = 0.0

    return {
        "slope_pred_vs_ref": float(slope),
        "mean_bias": bias,
        "mae": mae,
        "rmse": rmse,
        "spearman": rho,
    }


def calibrate_and_validate_bridges(
    d_raw_pairs: np.ndarray,
    d_ref_pairs: np.ndarray,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Calibration de compatibilité sur des paires déjà extraites.

    Pour une analyse confirmatoire, il est préférable de séparer les
    stimuli-ponts par nœuds avant de former les paires, afin d'éviter
    qu'un même nœud apparaisse dans calibration et test.
    """
    raw = _as_finite_array(
        d_raw_pairs,
        "d_raw_pairs",
    ).reshape(-1)

    ref = _as_finite_array(
        d_ref_pairs,
        "d_ref_pairs",
    ).reshape(-1)

    if len(raw) != len(ref):
        raise ValueError(
            "d_raw_pairs et d_ref_pairs doivent avoir la même longueur."
        )

    if len(raw) < 8:
        raise ValueError(
            "Au moins huit paires sont recommandées."
        )

    rng = np.random.default_rng(
        int(seed)
    )

    permutation = rng.permutation(
        len(raw)
    )

    n_cal = max(
        int(round(0.60 * len(raw))),
        3,
    )

    n_val = max(
        int(round(0.20 * len(raw))),
        2,
    )

    if n_cal + n_val >= len(raw):
        n_val = max(
            len(raw) - n_cal - 1,
            1,
        )

    idx_cal = permutation[:n_cal]
    idx_val = permutation[n_cal:n_cal + n_val]
    idx_test = permutation[n_cal + n_val:]

    if len(idx_test) == 0:
        idx_test = idx_val

    linear_model = fit_linear_scale(
        raw[idx_cal],
        ref[idx_cal],
        fix_intercept_zero=True,
    )

    monotone_model = fit_monotone_binned(
        raw[idx_cal],
        ref[idx_cal],
        n_bins=min(
            6,
            max(
                2,
                len(idx_cal) // 4,
            ),
        ),
    )

    pred_val_linear = apply_linear_scale(
        raw[idx_val],
        linear_model,
    )

    pred_val_monotone = apply_monotone_binned(
        raw[idx_val],
        monotone_model,
    )

    val_linear = evaluate_calibration_split(
        raw[idx_val],
        ref[idx_val],
        pred_val_linear,
    )

    val_monotone = evaluate_calibration_split(
        raw[idx_val],
        ref[idx_val],
        pred_val_monotone,
    )

    if val_monotone["rmse"] < val_linear["rmse"]:
        selected_name = "monotone_piecewise"
        selected_model = monotone_model
        pred_test = apply_monotone_binned(
            raw[idx_test],
            monotone_model,
        )
    else:
        selected_name = "linear_zero_intercept"
        selected_model = linear_model
        pred_test = apply_linear_scale(
            raw[idx_test],
            linear_model,
        )

    test_metrics = evaluate_calibration_split(
        raw[idx_test],
        ref[idx_test],
        pred_test,
    )

    non_equivalence = bool(
        abs(
            test_metrics["slope_pred_vs_ref"]
            - 1.0
        ) > 0.25
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
        [
            [-1.2, -0.4],
            [-0.5, 0.7],
            [0.1, -0.9],
            [0.6, 0.3],
            [1.1, -0.2],
            [1.5, 0.9],
        ],
        dtype=float,
    )

    angle = 0.65

    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=float,
    )

    source = (
        2.4 * (reference @ rotation.T)
        + np.array([4.0, -2.5])
        + 0.05 * rng.standard_normal(reference.shape)
    )

    procrustes = procrustes_align_bridges(
        coords_source=source,
        coords_target=reference,
        allow_reflection=False,
        estimate_scale=True,
        return_raw_distances=True,
    )

    print(
        "Facteur d'échelle :",
        round(
            float(procrustes["scale_factor"]),
            6,
        ),
    )

    print(
        "Disparité Procrustes :",
        round(
            float(procrustes["procrustes_disparity"]),
            6,
        ),
    )

    anchor_points = np.array(
        [
            [-1.0, -0.5],
            [0.0, 0.2],
            [1.2, 0.8],
        ],
        dtype=float,
    )

    anchor_metrics = np.array(
        [
            [[2.0, 0.0], [0.0, 1.5]],
            [[1.5, 0.2], [0.2, 1.8]],
            [[3.0, 0.0], [0.0, 2.5]],
        ],
        dtype=float,
    )

    waypoints = np.array(
        [
            [-1.0, -0.5],
            [0.0, 0.2],
            [1.2, 0.8],
        ],
        dtype=float,
    )

    integrated_distance = path_integrated_distance(
        waypoints=waypoints,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        n_substeps=20,
        bandwidth=1.0,
    )

    print(
        "Distance intégrée :",
        round(
            integrated_distance,
            6,
        ),
    )

    region_samples = rng.uniform(
        low=-1.5,
        high=1.5,
        size=(500, 2),
    )

    volume = local_volume_in_region(
        region_samples=region_samples,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        region_base_volume=9.0,
        bandwidth=1.0,
    )

    print(
        "Volume local :",
        {
            key: round(value, 6)
            if isinstance(value, float)
            else value
            for key, value in volume.items()
        },
    )
