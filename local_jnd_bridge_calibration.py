#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOCAL JND BRIDGE CALIBRATION
============================

Calibration de distances perceptives et métriques locales.

Fonctions principales :
    - alignement Procrustes ;
    - estimation de métriques SPD ;
    - interpolation de métriques locales (unitaire et par lot) ;
    - intégration de distances sur des chemins (vectorisée) ;
    - estimation de volumes locaux (vectorisée) ;
    - calibration linéaire ;
    - calibration monotone par PAVA ;
    - séparation calibration / validation / test ;
    - export JSON des résultats (`to_jsonable`).

Conventions :
    - les coordonnées sont représentées en lignes ;
    - les distances sont non négatives ;
    - les métriques locales sont symétriques définies positives ;
    - les seuils JND sont strictement positifs ;
    - les calibrations sont monotones non décroissantes.

Limites :
    - l'identifiabilité d'une métrique dépend des directions sondées ;
    - l'interpolation est une moyenne pondérée de métriques ;
    - la distance intégrée correspond au chemin fourni ;
    - le volume est une estimation par échantillonnage ;
    - le découpage de validation est effectué au niveau des paires,
      et non au niveau des nœuds partagés.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional

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
    "to_jsonable",
]


# ============================================================================
# 1. VALIDATION NUMÉRIQUE
# ============================================================================

def _as_finite_array(
    values: Any,
    name: str,
    ndim: Optional[int] = None,
) -> np.ndarray:
    """Convertit une entrée en tableau float fini."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} doit être convertible en tableau numérique."
        ) from exc

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name} doit avoir {ndim} dimensions ; "
            f"dimensions reçues : {array.ndim}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contient des valeurs NaN ou infinies.")

    return array


def _validate_nonnegative(values: Any, name: str) -> np.ndarray:
    array = _as_finite_array(values, name)
    if np.any(array < 0):
        raise ValueError(
            f"{name} doit contenir uniquement des valeurs non négatives."
        )
    return array


def _validate_points(values: Any, name: str, min_points: int = 1) -> np.ndarray:
    points = _as_finite_array(values, name, ndim=2)
    if points.shape[0] < min_points:
        raise ValueError(f"{name} doit contenir au moins {min_points} point(s).")
    if points.shape[1] < 1:
        raise ValueError(f"{name} doit contenir au moins une dimension.")
    return points


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} doit être un entier positif.")
    if value < 1:
        raise ValueError(f"{name} doit être supérieur ou égal à 1.")
    return int(value)


def _ensure_spd(matrix: np.ndarray, eig_floor: float = 1e-9) -> np.ndarray:
    """
    Symétrise et régularise une matrice (ou un lot de matrices, forme
    (..., d, d)) pour obtenir des matrices SPD.

    Les valeurs propres inférieures à eig_floor sont remplacées par eig_floor.
    Il s'agit d'une régularisation numérique : elle ne prouve pas que la
    matrice initiale était SPD.
    """
    M = _as_finite_array(matrix, "matrix")

    if M.ndim < 2 or M.shape[-1] != M.shape[-2]:
        raise ValueError("La matrice doit être carrée.")

    eig_floor = float(eig_floor)
    if not np.isfinite(eig_floor) or eig_floor <= 0:
        raise ValueError("eig_floor doit être strictement positif et fini.")

    M = 0.5 * (M + np.swapaxes(M, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, eig_floor)

    result = (eigenvectors * eigenvalues[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return 0.5 * (result + np.swapaxes(result, -1, -2))


def to_jsonable(obj: Any) -> Any:
    """Convertit récursivement tableaux et scalaires NumPy en types JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


# ============================================================================
# 2. ALIGNEMENT PROCRUSTES
# ============================================================================

def procrustes_align_bridges(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    allow_reflection: bool = True,
    return_raw_distances: bool = False,
) -> Dict[str, Any]:
    """
    Aligne une configuration source sur une configuration cible.

    Transformations : translation, rotation, réflexion optionnelle,
    changement d'échelle isotrope. Coordonnées en lignes.

        X_aligned = scale * (X_centered @ R) + target_mean

    R est calculée par SVD de H = Xc.T @ Yc = U S Vt, avec R = U D Vt.
    Aucune permutation des points n'est effectuée.
    """
    source = _validate_points(source_coords, "source_coords", min_points=2)
    target = _validate_points(target_coords, "target_coords", min_points=2)

    if source.shape[0] != target.shape[0]:
        raise ValueError(
            "Les configurations doivent avoir le même nombre de points."
        )
    if not isinstance(allow_reflection, (bool, np.bool_)):
        raise ValueError("allow_reflection doit être booléen.")

    n_points = source.shape[0]
    dimension = max(source.shape[1], target.shape[1])

    X = np.zeros((n_points, dimension))
    Y = np.zeros((n_points, dimension))
    X[:, : source.shape[1]] = source
    Y[:, : target.shape[1]] = target

    source_mean = X.mean(axis=0)
    target_mean = Y.mean(axis=0)
    Xc = X - source_mean
    Yc = Y - target_mean

    source_norm_sq = float(np.sum(Xc ** 2))
    target_norm_sq = float(np.sum(Yc ** 2))
    eps = np.finfo(float).eps

    if source_norm_sq <= eps:
        raise ValueError("La configuration source est dégénérée.")
    if target_norm_sq <= eps:
        raise ValueError("La configuration cible est dégénérée.")

    U, singular_values, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)

    correction = np.eye(dimension)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0

    R = U @ correction @ Vt
    trace_opt = float(np.sum(singular_values * np.diag(correction)))
    scale = max(0.0, trace_opt / source_norm_sq)

    aligned_full = scale * (Xc @ R) + target_mean
    aligned_coords = aligned_full[:, : target.shape[1]]

    disparity = float(np.sum((aligned_full - Y) ** 2) / target_norm_sq)

    result = {
        "aligned_coords": aligned_coords,
        "rotation": R,
        "scale": scale,
        "disparity": disparity,
        "allow_reflection": bool(allow_reflection),
        "source_mean": source_mean,
        "target_mean": target_mean,
    }

    if return_raw_distances:
        result["source_distances"] = squareform(pdist(source))
        result["target_distances"] = squareform(pdist(target))
        result["aligned_distances"] = squareform(pdist(aligned_coords))

    return result


# ============================================================================
# 3. ESTIMATION D'UNE MÉTRIQUE LOCALE SPD
# ============================================================================

def fit_local_metric_at_anchor(
    displacements: np.ndarray,
    thresholds: np.ndarray,
    G_prior: Optional[np.ndarray] = None,
    shrink_lambda: float = 0.1,
    reg: float = 1e-6,
    max_iter: int = 2000,
    eig_floor: float = 1e-9,
) -> np.ndarray:
    """
    Estime une métrique locale SPD à partir de sondes directionnelles.

    Modèle : v_i.T @ G @ v_i ≈ 1 / u_i²,  v_i = d_i / ||d_i||.

    Perte minimisée (G = A @ A.T, A triangulaire inférieure) :

        moyenne((v.T G v - 1/u²)²)
        + shrink_lambda * ||G - G_prior||_F²      (rappel vers le prior)
        + reg           * ||A - L_prior||_F²      (stabilisation du facteur)

    Les deux pénalités tirent vers le prior (et non vers zéro).
    `shrink_lambda` est un POIDS de pénalité (≥ 0), pas une combinaison
    convexe ; il n'a pas besoin d'appartenir à [0, 1].

    Identifiabilité : une matrice d'ordre d a d(d+1)/2 paramètres. Si les
    directions sondées n'identifient pas tous les paramètres, un
    avertissement est émis et la solution dépend du prior.

    Convergence : un arrêt de L-BFGS-B sans « success » émet un
    RuntimeWarning ; une erreur n'est levée que si la solution est non finie.
    """
    D = _as_finite_array(displacements, "displacements", ndim=2)
    u = _validate_nonnegative(thresholds, "thresholds").reshape(-1)

    if D.shape[0] != len(u):
        raise ValueError(
            "Le nombre de déplacements et de seuils doit correspondre."
        )
    if D.shape[0] < 1 or D.shape[1] < 1:
        raise ValueError("Les sondes ne peuvent pas être vides.")
    if np.any(u <= 0):
        raise ValueError("Tous les seuils doivent être strictement positifs.")
    if np.any(u < math.sqrt(np.finfo(float).tiny)):
        raise ValueError(
            "Un seuil est trop petit pour calculer 1/u² de façon stable."
        )

    shrink_lambda = float(shrink_lambda)
    reg = float(reg)

    if not np.isfinite(shrink_lambda) or shrink_lambda < 0:
        raise ValueError("shrink_lambda doit être fini et non négatif.")
    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg doit être fini et non négatif.")

    max_iter = _positive_integer(max_iter, "max_iter")

    norms = np.linalg.norm(D, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Un déplacement est nul ou numériquement dégénéré.")

    V = D / norms[:, None]
    d = D.shape[1]
    n_parameters = d * (d + 1) // 2

    iu, ju = np.triu_indices(d)
    design = V[:, iu] * V[:, ju] * np.where(iu == ju, 1.0, 2.0)
    design_rank = int(np.linalg.matrix_rank(design))

    if design_rank < n_parameters:
        warnings.warn(
            "Les directions ne permettent pas d'identifier seules "
            "tous les paramètres de la métrique. La solution dépend "
            "de la régularisation et du prior.",
            RuntimeWarning,
            stacklevel=2,
        )

    target = 1.0 / (u ** 2)

    if G_prior is None:
        G0 = np.eye(d)
    else:
        G0 = _as_finite_array(G_prior, "G_prior", ndim=2)
        if G0.shape != (d, d):
            raise ValueError(
                "G_prior doit avoir les mêmes dimensions que la métrique."
            )
        G0 = _ensure_spd(G0, eig_floor=eig_floor)

    L0 = np.linalg.cholesky(G0)
    tril_i, tril_j = np.tril_indices(d)
    theta0 = L0[tril_i, tril_j].copy()

    def unpack(theta: np.ndarray) -> np.ndarray:
        A = np.zeros((d, d))
        A[tril_i, tril_j] = theta
        return A

    def objective(theta: np.ndarray) -> float:
        A = unpack(theta)
        G = A @ A.T
        predicted = np.einsum("ni,ij,nj->n", V, G, V)
        data_loss = float(np.mean((predicted - target) ** 2))
        prior_loss = float(np.sum((G - G0) ** 2))
        factor_loss = float(np.sum((A - L0) ** 2))
        return data_loss + shrink_lambda * prior_loss + reg * factor_loss

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )

    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Optimisation de la métrique locale échouée : {result.message}"
        )
    if not result.success:
        warnings.warn(
            "L-BFGS-B n'a pas déclaré de convergence "
            f"({result.message}) ; la solution courante est conservée.",
            RuntimeWarning,
            stacklevel=2,
        )

    A_opt = unpack(result.x)
    return _ensure_spd(A_opt @ A_opt.T, eig_floor=eig_floor)


# ============================================================================
# 4. INTERPOLATION DES MÉTRIQUES LOCALES
# ============================================================================

def _prepare_anchor_metrics(
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    eig_floor: float,
):
    """Valide les ancres et régularise leurs métriques UNE seule fois."""
    anchors = _validate_points(anchor_points, "anchor_points")
    metrics = _as_finite_array(anchor_metrics, "anchor_metrics", ndim=3)
    n_anchors, d = anchors.shape

    if metrics.shape != (n_anchors, d, d):
        raise ValueError(
            "anchor_metrics doit avoir la forme "
            "(n_anchors, dimension, dimension)."
        )
    return anchors, _ensure_spd(metrics, eig_floor=eig_floor)


def _interpolate_prepared(
    X: np.ndarray,
    anchors: np.ndarray,
    spd_metrics: np.ndarray,
    bandwidth: float,
    eig_floor: float,
) -> np.ndarray:
    """Interpolation par lot avec ancres déjà validées et régularisées."""
    distances_sq = np.sum(
        (X[:, None, :] - anchors[None, :, :]) ** 2, axis=2
    )  # (m, n_anchors)

    log_w = -0.5 * distances_sq / (bandwidth ** 2)
    if not np.all(np.isfinite(log_w)):
        raise ValueError(
            "Les distances ou le bandwidth produisent des poids instables."
        )

    log_w -= np.max(log_w, axis=1, keepdims=True)
    weights = np.exp(log_w)
    weight_sum = np.sum(weights, axis=1, keepdims=True)

    if np.any(weight_sum <= 0) or not np.all(np.isfinite(weight_sum)):
        raise RuntimeError("Impossible de normaliser les poids d'interpolation.")

    weights /= weight_sum
    G = np.einsum("mn,nij->mij", weights, spd_metrics)
    return _ensure_spd(G, eig_floor=eig_floor)


def _check_bandwidth(bandwidth: Any) -> float:
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth doit être strictement positif et fini.")
    return bandwidth


def interpolate_local_metric_batch(
    query_points: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float,
    eig_floor: float = 1e-9,
) -> np.ndarray:
    """
    Interpole G(x) pour un lot de points : (m, d) -> (m, d, d).

        w_i ∝ exp(-||x - a_i||² / (2 h²))

    La combinaison convexe de matrices SPD reste SPD.
    """
    X = _validate_points(query_points, "query_points")
    bandwidth = _check_bandwidth(bandwidth)
    anchors, spd = _prepare_anchor_metrics(anchor_points, anchor_metrics, eig_floor)

    if X.shape[1] != anchors.shape[1]:
        raise ValueError(
            "Dimensions incompatibles entre query_points et anchor_points."
        )
    return _interpolate_prepared(X, anchors, spd, bandwidth, eig_floor)


def interpolate_local_metric(
    query_point: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float,
    eig_floor: float = 1e-9,
) -> np.ndarray:
    """Interpole la métrique locale en un seul point (vecteur 1D)."""
    x = _as_finite_array(query_point, "query_point", ndim=1)
    return interpolate_local_metric_batch(
        x[None, :], anchor_points, anchor_metrics, bandwidth, eig_floor
    )[0]


# ============================================================================
# 5. INTÉGRATION DE DISTANCE SUR UN CHEMIN
# ============================================================================

def path_integrated_distance(
    waypoints: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float,
    n_substeps: int = 20,
    eig_floor: float = 1e-9,
) -> float:
    """
    Intègre une longueur métrique sur un chemin polygonal (vectorisé).

        ds = sqrt(dx.T @ G(x) @ dx)

    G est évaluée au milieu de chaque sous-segment. Quadrature sur le
    chemin fourni, pas de calcul de géodésique minimale.
    """
    path = _validate_points(waypoints, "waypoints", min_points=1)
    n_substeps = _positive_integer(n_substeps, "n_substeps")
    bandwidth = _check_bandwidth(bandwidth)
    anchors, spd = _prepare_anchor_metrics(anchor_points, anchor_metrics, eig_floor)

    if path.shape[1] != anchors.shape[1]:
        raise ValueError(
            "waypoints et anchor_points doivent avoir la même dimension."
        )

    if path.shape[0] == 1:
        _interpolate_prepared(path, anchors, spd, bandwidth, eig_floor)
        return 0.0

    starts = path[:-1]
    deltas = path[1:] - starts  # (S, d)

    t_mid = (np.arange(n_substeps) + 0.5) / n_substeps
    mids = (
        starts[:, None, :] + t_mid[None, :, None] * deltas[:, None, :]
    ).reshape(-1, path.shape[1])

    G = _interpolate_prepared(mids, anchors, spd, bandwidth, eig_floor)
    deltas_rep = np.repeat(deltas, n_substeps, axis=0)

    local_sq = np.einsum("mi,mij,mj->m", deltas_rep, G, deltas_rep)

    if np.any(local_sq < -1e-10):
        raise RuntimeError(
            "La métrique a produit une longueur quadratique négative."
        )

    return float(np.sum(np.sqrt(np.maximum(local_sq, 0.0))) / n_substeps)


# ============================================================================
# 6. VOLUME LOCAL
# ============================================================================

def local_volume_in_region(
    region_samples: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float,
    region_base_volume: float,
    eig_floor: float = 1e-9,
    chunk_size: int = 4096,
) -> Dict[str, float]:
    """
    Estime un volume transformé par une métrique locale (vectorisé, par blocs).

        densité(x) = sqrt(det G(x)),   V ≈ V_base * mean(densité(x_i))

    L'interprétation Monte-Carlo suppose des échantillons indépendants et
    uniformes dans la région. L'erreur standard empirique n'est pas une
    garantie d'incertitude pour des échantillons déterministes ou corrélés.
    """
    samples = _validate_points(region_samples, "region_samples", min_points=1)
    bandwidth = _check_bandwidth(bandwidth)
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    anchors, spd = _prepare_anchor_metrics(anchor_points, anchor_metrics, eig_floor)

    base_volume = float(region_base_volume)
    if not np.isfinite(base_volume) or base_volume <= 0:
        raise ValueError(
            "region_base_volume doit être strictement positif et fini."
        )
    if samples.shape[1] != anchors.shape[1]:
        raise ValueError(
            "region_samples et anchor_points doivent avoir la même dimension."
        )

    log_max = math.log(np.finfo(float).max)
    parts = []

    for start in range(0, len(samples), chunk_size):
        block = samples[start : start + chunk_size]
        G = _interpolate_prepared(block, anchors, spd, bandwidth, eig_floor)
        sign, logdet = np.linalg.slogdet(G)

        if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
            raise RuntimeError(
                "La métrique ne possède pas un déterminant positif fini."
            )

        log_density = 0.5 * logdet
        if np.any(log_density > log_max):
            raise OverflowError(
                "La densité volumique dépasse la capacité numérique."
            )
        parts.append(np.exp(log_density))

    densities = np.concatenate(parts)
    n = len(densities)
    mean_density = float(np.mean(densities))

    if n > 1:
        se_density = float(np.std(densities, ddof=1) / math.sqrt(n))
    else:
        se_density = 0.0

    return {
        "estimated_volume": float(base_volume * mean_density),
        "mean_density": mean_density,
        "standard_error": float(base_volume * se_density),
        "n_samples": int(n),
    }


# ============================================================================
# 7. CALIBRATION LINÉAIRE
# ============================================================================

def fit_linear_scale(
    raw_distances: np.ndarray,
    true_distances: np.ndarray,
    fit_intercept: bool = True,
) -> Dict[str, float]:
    """
    Ajuste true ≈ a + b * raw. La pente doit être non négative afin de
    préserver l'ordre des distances.
    """
    x = _validate_nonnegative(raw_distances, "raw_distances").reshape(-1)
    y = _validate_nonnegative(true_distances, "true_distances").reshape(-1)

    if x.shape != y.shape:
        raise ValueError("Les deux vecteurs doivent avoir la même longueur.")
    if len(x) < 2:
        raise ValueError("Au moins deux observations sont nécessaires.")

    eps = np.finfo(float).eps

    if fit_intercept:
        if np.ptp(x) <= eps:
            raise ValueError(
                "Les distances brutes sont constantes : "
                "pente et interception non identifiables."
            )
        design = np.column_stack([np.ones_like(x), x])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            raise ValueError("Le problème de régression est de rang insuffisant.")
        a, b = map(float, coefficients)
    else:
        denominator = float(np.dot(x, x))
        if denominator <= eps:
            raise ValueError("Les distances brutes sont nulles.")
        a = 0.0
        b = float(np.dot(x, y) / denominator)

    if not np.isfinite(a) or not np.isfinite(b):
        raise RuntimeError("La régression a produit des coefficients non finis.")
    if b < -1e-12:
        raise ValueError(
            "La pente estimée est négative : "
            "calibration linéaire monotone impossible."
        )

    return {"a": a, "b": max(0.0, b)}


def apply_linear_scale(
    raw_distances: np.ndarray,
    model: Dict[str, float],
) -> np.ndarray:
    """Applique une calibration linéaire non décroissante."""
    x = _as_finite_array(raw_distances, "raw_distances")

    if not isinstance(model, dict):
        raise ValueError("model doit être un dictionnaire.")

    try:
        a = float(model["a"])
        b = float(model["b"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "model doit contenir les coefficients numériques a et b."
        ) from exc

    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Les coefficients doivent être finis.")
    if b < 0:
        raise ValueError("La pente doit être non négative.")

    return np.maximum(a + b * x, 0.0)


# ============================================================================
# 8. CALIBRATION MONOTONE : PAVA
# ============================================================================

def _pool_adjacent_violators(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Régression isotone non décroissante par PAVA."""
    y = _as_finite_array(values, "values", ndim=1)
    w = _as_finite_array(weights, "weights", ndim=1)

    if len(y) != len(w):
        raise ValueError("values et weights doivent avoir la même longueur.")
    if len(y) == 0:
        raise ValueError("PAVA ne peut pas traiter un vecteur vide.")
    if np.any(w <= 0):
        raise ValueError("Tous les poids doivent être strictement positifs.")

    vals, wts, starts, ends = [], [], [], []

    for i, (value, weight) in enumerate(zip(y, w)):
        vals.append(float(value))
        wts.append(float(weight))
        starts.append(i)
        ends.append(i + 1)

        while len(vals) >= 2 and vals[-2] > vals[-1]:
            w1, w2 = wts[-2], wts[-1]
            merged_w = w1 + w2
            merged_v = (w1 * vals[-2] + w2 * vals[-1]) / merged_w
            start, end = starts[-2], ends[-1]

            vals[-2:] = [merged_v]
            wts[-2:] = [merged_w]
            starts[-2:] = [start]
            ends[-2:] = [end]

    result = np.empty(len(y))
    for value, start, end in zip(vals, starts, ends):
        result[start:end] = value
    return result


def fit_monotone_binned(
    raw_distances: np.ndarray,
    true_distances: np.ndarray,
    n_bins: int = 6,
) -> Dict[str, Any]:
    """
    Calibration monotone par classes de quantiles : les moyennes de chaque
    classe sont projetées sur une séquence non décroissante par PAVA.
    """
    x = _validate_nonnegative(raw_distances, "raw_distances").reshape(-1)
    y = _validate_nonnegative(true_distances, "true_distances").reshape(-1)

    if x.shape != y.shape:
        raise ValueError("Les vecteurs doivent avoir la même longueur.")
    if len(x) < 2:
        raise ValueError("Au moins deux observations sont nécessaires.")

    n_bins = _positive_integer(n_bins, "n_bins")
    if n_bins < 2:
        raise ValueError("n_bins doit être supérieur ou égal à 2.")

    if np.ptp(x) <= np.finfo(float).eps:
        return {
            "type": "monotone_constant",
            "edges": np.array([float(np.min(x)), float(np.max(x))]),
            "centers": np.array([float(np.mean(x))]),
            "values": np.array([float(np.mean(y))]),
        }

    n_bins = min(n_bins, len(x))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))

    if len(edges) < 2:
        return {
            "type": "monotone_constant",
            "edges": edges,
            "centers": np.array([float(np.mean(x))]),
            "values": np.array([float(np.mean(y))]),
        }

    bin_ids = np.digitize(x, edges[1:-1], right=False)

    centers, means, weights = [], [], []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        centers.append(float(np.mean(x[mask])))
        means.append(float(np.mean(y[mask])))
        weights.append(float(np.sum(mask)))

    centers = np.asarray(centers)
    means = np.asarray(means)
    weights = np.asarray(weights)

    order = np.argsort(centers)
    centers, means, weights = centers[order], means[order], weights[order]

    return {
        "type": "monotone_piecewise",
        "edges": edges,
        "centers": centers,
        "values": _pool_adjacent_violators(means, weights),
        "weights": weights,
    }


def apply_monotone_binned(
    raw_distances: np.ndarray,
    model: Dict[str, Any],
) -> np.ndarray:
    """Applique une calibration monotone par interpolation linéaire."""
    x = _as_finite_array(raw_distances, "raw_distances")

    if not isinstance(model, dict):
        raise ValueError("model doit être un dictionnaire.")

    model_type = model.get("type")

    try:
        centers = _as_finite_array(model["centers"], "model.centers", ndim=1)
        values = _validate_nonnegative(model["values"], "model.values").reshape(-1)
    except KeyError as exc:
        raise ValueError("Le modèle monotone est incomplet.") from exc

    if len(centers) == 0 or len(centers) != len(values):
        raise ValueError(
            "centers et values doivent avoir la même longueur non nulle."
        )
    if np.any(np.diff(centers) <= 0):
        raise ValueError("Les centres doivent être strictement croissants.")
    if np.any(np.diff(values) < -1e-12):
        raise ValueError("Les valeurs du modèle doivent être non décroissantes.")

    if model_type == "monotone_constant":
        return np.full_like(x, float(values[0]), dtype=float)

    if model_type != "monotone_piecewise":
        raise ValueError(f"Type de modèle monotone inconnu : {model_type}")

    return np.interp(x, centers, values, left=values[0], right=values[-1])


# ============================================================================
# 9. ÉVALUATION D'UNE CALIBRATION
# ============================================================================

def evaluate_calibration_split(
    raw_distances: np.ndarray,
    true_distances: np.ndarray,
    predicted_distances: np.ndarray,
) -> Dict[str, Any]:
    """
    Évalue une calibration sur un ensemble indépendant de l'ajustement.

    Pente : true ≈ intercept + slope * predicted.
    Spearman et pente valent None lorsqu'ils ne sont pas définis.
    """
    raw = _validate_nonnegative(raw_distances, "raw_distances").reshape(-1)
    true = _validate_nonnegative(true_distances, "true_distances").reshape(-1)
    pred = _validate_nonnegative(predicted_distances, "predicted_distances").reshape(-1)

    if not (len(raw) == len(true) == len(pred)):
        raise ValueError("Les trois vecteurs doivent avoir la même longueur.")
    if len(raw) < 2:
        raise ValueError("Au moins deux observations sont nécessaires.")

    residual = pred - true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    if np.ptp(pred) > np.finfo(float).eps:
        slope = float(np.cov(pred, true, ddof=0)[0, 1] / np.var(pred))
    else:
        slope = None

    # Accès par indice : compatible avec toutes les versions de SciPy.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation = spearmanr(pred, true)[0]

    spearman = float(correlation) if np.isfinite(correlation) else None

    return {
        "n": int(len(raw)),
        "mae": mae,
        "rmse": rmse,
        "slope_pred_vs_ref": slope,
        "spearman": spearman,
    }


# ============================================================================
# 10. CALIBRATION ET VALIDATION DES PONTS
# ============================================================================

def calibrate_and_validate_bridges(
    raw_distances: np.ndarray,
    true_distances: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare une calibration linéaire et une calibration monotone.

    Séparation :
        - calibration : ajustement des modèles ;
        - validation : sélection du modèle ;
        - test : évaluation finale.

    Les ensembles sont disjoints au niveau des paires. Si plusieurs paires
    partagent des nœuds, une dépendance peut subsister : cette fonction ne
    réalise pas de séparation par groupes de nœuds.

    Si la régression linéaire est impossible (pente négative, distances
    brutes constantes…), le modèle linéaire est écarté et le modèle monotone
    est retenu ; la raison figure dans `diagnostics["linear_unavailable"]`.

    Minimum : 8 paires.
    """
    raw = _validate_nonnegative(raw_distances, "raw_distances").reshape(-1)
    true = _validate_nonnegative(true_distances, "true_distances").reshape(-1)

    if len(raw) != len(true):
        raise ValueError(
            "raw_distances et true_distances doivent avoir la même longueur."
        )

    n = len(raw)
    if n < 8:
        raise ValueError(
            "Au moins 8 paires sont nécessaires pour constituer "
            "les trois ensembles sans réutiliser les observations."
        )

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed doit être un entier.")

    indices = np.random.default_rng(int(seed)).permutation(n)

    n_test = max(2, int(round(0.20 * n)))
    n_validation = max(2, int(round(0.20 * n)))
    n_calibration = n - n_test - n_validation

    if n_calibration < 4:
        raise ValueError("Nombre insuffisant de paires pour la calibration.")

    idx_cal = indices[:n_calibration]
    idx_val = indices[n_calibration : n_calibration + n_validation]
    idx_test = indices[n_calibration + n_validation :]

    if (
        set(idx_cal) & set(idx_val)
        or set(idx_cal) & set(idx_test)
        or set(idx_val) & set(idx_test)
    ):
        raise RuntimeError("Les ensembles ne sont pas disjoints.")

    # --- Ajustement exclusivement sur la calibration ---------------------
    linear_model = None
    linear_unavailable = None
    try:
        linear_model = fit_linear_scale(raw[idx_cal], true[idx_cal], fit_intercept=True)
    except ValueError as exc:
        linear_unavailable = str(exc)

    monotone_model = fit_monotone_binned(
        raw[idx_cal],
        true[idx_cal],
        n_bins=min(6, max(2, len(idx_cal) // 4)),
    )

    # --- Validation ------------------------------------------------------
    monotone_val_metrics = evaluate_calibration_split(
        raw[idx_val],
        true[idx_val],
        apply_monotone_binned(raw[idx_val], monotone_model),
    )

    if linear_model is not None:
        linear_val_metrics = evaluate_calibration_split(
            raw[idx_val],
            true[idx_val],
            apply_linear_scale(raw[idx_val], linear_model),
        )
    else:
        linear_val_metrics = None

    # --- Sélection (sans réajustement) -----------------------------------
    if linear_val_metrics is not None and (
        linear_val_metrics["rmse"] <= monotone_val_metrics["rmse"]
    ):
        selected_name = "linear"
        selected_model = linear_model
        selected_validation = linear_val_metrics
    else:
        selected_name = monotone_model["type"]
        selected_model = monotone_model
        selected_validation = monotone_val_metrics

    # --- Test final -------------------------------------------------------
    if selected_name == "linear":
        test_predictions = apply_linear_scale(raw[idx_test], selected_model)
    else:
        test_predictions = apply_monotone_binned(raw[idx_test], selected_model)

    test_metrics = evaluate_calibration_split(
        raw[idx_test], true[idx_test], test_predictions
    )

    slope = test_metrics["slope_pred_vs_ref"]
    rho = test_metrics["spearman"]

    slope_warning = slope is None or abs(slope - 1.0) > 0.25
    correlation_warning = rho is None or rho < 0.50

    return {
        "selected_model": selected_name,
        "model": selected_model,
        "validation": selected_validation,
        "test": test_metrics,
        "split": {
            "method": "random_pair_split",
            "split_unit": "pair",
            "n_total": int(n),
            "n_calibration": int(len(idx_cal)),
            "n_validation": int(len(idx_val)),
            "n_test": int(len(idx_test)),
            "calibration_indices": idx_cal.copy(),
            "validation_indices": idx_val.copy(),
            "test_indices": idx_test.copy(),
            "seed": int(seed),
        },
        "candidate_validation": {
            "linear": linear_val_metrics,
            "monotone": monotone_val_metrics,
        },
        "diagnostics": {
            "non_equivalence_warning": bool(slope_warning or correlation_warning),
            "slope_warning": bool(slope_warning),
            "correlation_warning": bool(correlation_warning),
            "linear_unavailable": linear_unavailable,
            "warning_rule": (
                "Alerte opérationnelle si la pente test est absente "
                "ou s'écarte de 1 de plus de 0.25, ou si Spearman "
                "est absent ou inférieur à 0.50. Ces seuils ne "
                "constituent pas un test statistique formel "
                "d'équivalence."
            ),
            "independence_warning": (
                "La séparation est réalisée au niveau des paires. "
                "Si plusieurs paires partagent des nœuds, une "
                "dépendance entre calibration, validation et test "
                "peut subsister. Une séparation par groupes de "
                "nœuds nécessite leurs identifiants."
            ),
        },
    }


# ============================================================================
# 11. DÉMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("LOCAL JND BRIDGE CALIBRATION")
    print("=" * 72)

    rng = np.random.default_rng(42)

    # 11.1 Procrustes
    reference = rng.normal(size=(30, 3))
    angle = 0.65
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    source = 1.8 * reference @ rotation.T + np.array([2.0, -1.0, 0.5])

    procrustes = procrustes_align_bridges(
        source, reference, allow_reflection=True, return_raw_distances=True
    )
    print("\n[1] PROCRUSTES")
    print("Scale :", procrustes["scale"])
    print("Disparity :", procrustes["disparity"])

    # 11.2 Métrique locale
    displacements = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [1.0, -1.0, 0.5], [-0.5, 1.0, 1.0],
    ])
    true_metric = np.array([
        [2.0, 0.2, 0.1],
        [0.2, 1.5, 0.15],
        [0.1, 0.15, 1.2],
    ])
    directions = displacements / np.linalg.norm(displacements, axis=1, keepdims=True)
    thresholds = 1.0 / np.sqrt(
        np.einsum("ni,ij,nj->n", directions, true_metric, directions)
    )

    estimated_metric = fit_local_metric_at_anchor(
        displacements, thresholds, G_prior=np.eye(3),
        shrink_lambda=0.01, reg=1e-8,
    )
    print("\n[2] MÉTRIQUE LOCALE")
    print("Métrique de référence :")
    print(true_metric)
    print("Métrique estimée :")
    print(estimated_metric)
    print("Valeurs propres estimées :", np.linalg.eigvalsh(estimated_metric))

    # 11.3 Interpolation
    anchor_points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    anchor_metrics = np.stack([np.eye(3), true_metric, 1.5 * np.eye(3)])
    query = np.array([0.1, 0.2, 0.0])

    print("\n[3] INTERPOLATION")
    print(interpolate_local_metric(query, anchor_points, anchor_metrics, bandwidth=1.0))

    # 11.4 Distance intégrée
    waypoints = np.array([[-0.8, 0.0, 0.0], [0.0, 0.5, 0.0], [0.8, 0.0, 0.0]])
    print("\n[4] DISTANCE INTÉGRÉE")
    print(path_integrated_distance(
        waypoints, anchor_points, anchor_metrics, bandwidth=1.0, n_substeps=50
    ))

    # 11.5 Volume local
    region_samples = rng.uniform(-1.0, 1.0, size=(200, 3))
    print("\n[5] VOLUME LOCAL")
    print(local_volume_in_region(
        region_samples, anchor_points, anchor_metrics,
        bandwidth=1.0, region_base_volume=8.0,
    ))

    # 11.6 Calibration
    n_pairs = 200
    raw_distances = rng.uniform(0.1, 10.0, size=n_pairs)
    noise = rng.normal(0.0, 0.15, size=n_pairs)
    true_distances = np.maximum(
        1.2 + 1.7 * raw_distances + 0.08 * raw_distances ** 2 + noise, 0.0
    )

    calibration_result = calibrate_and_validate_bridges(
        raw_distances, true_distances, seed=42
    )
    print("\n[6] CALIBRATION")
    print("Modèle sélectionné :", calibration_result["selected_model"])
    print("Validation :", calibration_result["validation"])
    print("Test indépendant au niveau des paires :", calibration_result["test"])
    print("Diagnostics :", calibration_result["diagnostics"])

    print("\n" + "=" * 72)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 72)
