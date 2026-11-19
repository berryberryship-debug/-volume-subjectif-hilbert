#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOCAL JND BRIDGE CALIBRATION
============================

Calibration de distances perceptives et estimation de métriques locales SPD.

Fonctions principales
---------------------
- Alignement Procrustes : estimation et application figée.
- Centrage commun ou par condition.
- Estimation de métriques locales SPD.
- Interpolation locale linéaire ou log-euclidienne.
- Intégration de distances sur des chemins polygonaux.
- Estimation de volumes locaux par échantillonnage.
- Calibration linéaire et monotone par PAVA.
- Séparation calibration / validation / test.
- Export des résultats vers des structures compatibles JSON.

API historique
--------------
Les fonctions historiques du module 1 sont conservées :
- spd_logm
- spd_expm
- fit_frozen_procrustes
- transform_frozen_procrustes
- estimate_local_spd_metric
- interpolate_metric_field_log_euclidean
- compute_integrated_path_distance
- compute_local_volume_element
- pava_isotonic_regression
- calibrate_scale_pava

Principes numériques
--------------------
- Les coordonnées sont représentées en lignes.
- Les entrées numériques doivent être finies.
- Les distances sont non négatives.
- Les seuils JND sont strictement positifs.
- Les métriques locales sont symétriques définies positives.
- Les calibrations monotones sont non décroissantes.
- Les transformations Procrustes sont estimées une fois puis figées.
- Les interpolations utilisent des poids gaussiens stabilisés.
- L'intégration calcule la longueur du chemin fourni, et non une
  géodésique minimale.

Limites méthodologiques
----------------------
- L'identifiabilité d'une métrique dépend des directions sondées.
- Une régularisation ne remplace pas l'information directionnelle absente.
- L'interpolation produit une moyenne pondérée des métriques.
- Le volume est estimé par échantillonnage de la région.
- L'incertitude Monte-Carlo suppose des échantillons indépendants et
  représentatifs.
- Le découpage de validation est effectué au niveau des paires.
  Des nœuds partagés peuvent donc maintenir une dépendance entre ensembles.
- La validation au niveau des paires ne constitue pas une validation
  indépendante au niveau des unités expérimentales.

Version : 1.2.1
"""

from __future__ import annotations

import math
import warnings

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


__version__ = "1.2.1"


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
    "apply_procrustes_transform",
    "calibrate_and_validate_bridges",
    "to_jsonable",
    "spd_logm",
    "spd_expm",
    "fit_frozen_procrustes",
    "transform_frozen_procrustes",
    "estimate_local_spd_metric",
    "interpolate_metric_field_log_euclidean",
    "compute_integrated_path_distance",
    "compute_local_volume_element",
    "pava_isotonic_regression",
    "calibrate_scale_pava",
    "__version__",
]


# ============================================================================
# 0. CONSTANTES ET TYPES
# ============================================================================

_CENTERING_MODES = ("common", "per_condition")
_INTERPOLATIONS = ("linear", "log_euclidean")

_SLOPE_TOLERANCE = 0.25
_MIN_SPEARMAN = 0.50

_DEFAULT_EIG_FLOOR = 1e-9
_DEFAULT_SHRINK_LAMBDA = 0.1
_DEFAULT_REG = 1e-6


CenteringMode = Literal["common", "per_condition"]
InterpolationMode = Literal["linear", "log_euclidean"]


class LinearScaleModel(TypedDict):
    a: float
    b: float


class MonotoneModel(TypedDict, total=False):
    type: Literal["monotone_constant", "monotone_piecewise"]
    edges: NDArray[np.floating]
    centers: NDArray[np.floating]
    values: NDArray[np.floating]
    weights: NDArray[np.floating]


class CalibrationMetrics(TypedDict):
    n: int
    mae: float
    rmse: float
    slope_pred_vs_ref: Optional[float]
    spearman: Optional[float]


# ============================================================================
# 1. VALIDATION NUMÉRIQUE
# ============================================================================

def _as_finite_array(
    values: ArrayLike,
    name: str,
    ndim: Optional[int] = None,
) -> NDArray[np.floating]:
    """Convertit une entrée en tableau float et vérifie sa finitude."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} doit être convertible en tableau numérique."
        ) from exc

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name} doit avoir {ndim} dimension(s) ; "
            f"dimensions reçues : {array.ndim}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contient des valeurs NaN ou infinies.")

    return array


def _validate_nonnegative(
    values: ArrayLike,
    name: str,
) -> NDArray[np.floating]:
    array = _as_finite_array(values, name)
    if np.any(array < 0):
        raise ValueError(
            f"{name} doit contenir uniquement des valeurs non négatives."
        )
    return array


def _validate_points(
    values: ArrayLike,
    name: str,
    min_points: int = 1,
) -> NDArray[np.floating]:
    points = _as_finite_array(values, name, ndim=2)

    if points.shape[0] < min_points:
        raise ValueError(
            f"{name} doit contenir au moins {min_points} point(s)."
        )

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


def _ensure_spd(
    matrix: ArrayLike,
    eig_floor: float = _DEFAULT_EIG_FLOOR,
) -> NDArray[np.floating]:
    """
    Symétrise et régularise une matrice ou un lot de matrices (..., d, d).

    Les valeurs propres inférieures à eig_floor sont remplacées par
    eig_floor. Cette opération assure une sortie SPD, sans démontrer que
    la matrice initiale était SPD.
    """
    M = _as_finite_array(matrix, "matrix")

    if M.ndim < 2 or M.shape[-1] != M.shape[-2]:
        raise ValueError("La matrice doit être carrée.")

    eig_floor = float(eig_floor)

    if not np.isfinite(eig_floor) or eig_floor <= 0:
        raise ValueError(
            "eig_floor doit être strictement positif et fini."
        )

    M = 0.5 * (M + np.swapaxes(M, -1, -2))

    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, eig_floor)

    result = (
        eigenvectors * eigenvalues[..., None, :]
    ) @ np.swapaxes(eigenvectors, -1, -2)

    return 0.5 * (result + np.swapaxes(result, -1, -2))


def to_jsonable(obj: Any) -> Any:
    """Convertit récursivement les objets NumPy en types compatibles JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Mapping):
        return {
            str(key): to_jsonable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]

    if isinstance(obj, set):
        return [to_jsonable(value) for value in obj]

    return obj


# ============================================================================
# 2. ALIGNEMENT PROCRUSTES : ESTIMATION ET APPLICATION FIGÉE
# ============================================================================

def _check_centering_mode(value: Any) -> CenteringMode:
    if value not in _CENTERING_MODES:
        raise ValueError(
            f"centering_mode doit appartenir à {_CENTERING_MODES} ; "
            f"valeur reçue : {value!r}."
        )

    return str(value)  # type: ignore[return-value]


def _prepare_condition_labels(
    labels: ArrayLike,
    n_points: int,
    name: str,
) -> NDArray[Any]:
    """Valide un vecteur d'étiquettes, une par ligne."""
    if labels is None:
        raise ValueError(
            f"{name} est obligatoire lorsque "
            "centering_mode='per_condition'."
        )

    arr = np.asarray(labels)

    if arr.ndim != 1 or len(arr) != n_points:
        raise ValueError(
            f"{name} doit être un vecteur 1D de longueur {n_points}."
        )

    return arr


def procrustes_align_bridges(
    source_coords: ArrayLike,
    target_coords: ArrayLike,
    allow_reflection: bool = True,
    return_raw_distances: bool = False,
    centering_mode: CenteringMode = "common",
    condition_labels: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """
    Estime une similitude de Procrustes de la source vers la cible.

    Transformation :
        X_aligned = scale * ((X - source_mean) @ R) + target_mean

    En mode common :
        un centre source et un centre cible sont estimés globalement.

    En mode per_condition :
        les centres sont estimés séparément pour chaque condition ;
        la rotation et l'échelle restent communes.

    La transformation enregistrée peut être appliquée à de nouvelles
    coordonnées sans réestimation avec apply_procrustes_transform().
    """
    source = _validate_points(
        source_coords, "source_coords", min_points=2
    )
    target = _validate_points(
        target_coords, "target_coords", min_points=2
    )

    mode = _check_centering_mode(centering_mode)

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

    X[:, :source.shape[1]] = source
    Y[:, :target.shape[1]] = target

    source_mean = X.mean(axis=0)
    target_mean = Y.mean(axis=0)

    source_means_by_condition = None
    target_means_by_condition = None

    if mode == "common":
        Xc = X - source_mean
        Yc = Y - target_mean

    else:
        labels = _prepare_condition_labels(
            condition_labels,
            n_points,
            "condition_labels",
        )

        Xc = np.empty_like(X)
        Yc = np.empty_like(Y)

        source_means_by_condition = {}
        target_means_by_condition = {}

        for label in np.unique(labels):
            mask = labels == label

            if int(np.sum(mask)) < 2:
                raise ValueError(
                    f"La condition {label!r} contient moins de "
                    "2 ponts : centrage par condition impossible."
                )

            key = label.item() if hasattr(label, "item") else label

            sm = X[mask].mean(axis=0)
            tm = Y[mask].mean(axis=0)

            source_means_by_condition[key] = sm
            target_means_by_condition[key] = tm

            Xc[mask] = X[mask] - sm
            Yc[mask] = Y[mask] - tm

    source_norm_sq = float(np.sum(Xc ** 2))
    target_norm_sq = float(np.sum(Yc ** 2))

    eps = np.finfo(float).eps

    if source_norm_sq <= eps:
        raise ValueError("La configuration source est dégénérée.")

    if target_norm_sq <= eps:
        raise ValueError("La configuration cible est dégénérée.")

    U, singular_values, Vt = np.linalg.svd(
        Xc.T @ Yc,
        full_matrices=False,
    )

    correction = np.eye(dimension)

    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0

    R = U @ correction @ Vt

    trace_opt = float(
        np.sum(singular_values * np.diag(correction))
    )

    scale = max(0.0, trace_opt / source_norm_sq)

    transform = {
        "centering_mode": mode,
        "rotation": R,
        "scale": scale,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "source_means_by_condition": source_means_by_condition,
        "target_means_by_condition": target_means_by_condition,
        "source_dim": int(source.shape[1]),
        "target_dim": int(target.shape[1]),
        "dimension": int(dimension),
        "allow_reflection": bool(allow_reflection),
    }

    aligned_coords = apply_procrustes_transform(
        source,
        transform,
        condition_labels=(
            condition_labels if mode == "per_condition" else None
        ),
    )

    residual = aligned_coords - target
    disparity = float(
        np.sum(residual ** 2) / target_norm_sq
    )

    result = {
        "aligned_coords": aligned_coords,
        "rotation": R,
        "scale": scale,
        "disparity": disparity,
        "allow_reflection": bool(allow_reflection),
        "centering_mode": mode,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "transform": transform,
    }

    if return_raw_distances:
        result["source_distances"] = squareform(pdist(source))
        result["target_distances"] = squareform(pdist(target))
        result["aligned_distances"] = squareform(
            pdist(aligned_coords)
        )

    return result


def apply_procrustes_transform(
    coords: ArrayLike,
    transform: Mapping[str, Any],
    condition_labels: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """Applique une transformation Procrustes déjà estimée."""
    if not isinstance(transform, Mapping):
        raise ValueError("transform doit être un dictionnaire.")

    if (
        "transform" in transform
        and isinstance(transform["transform"], Mapping)
    ):
        transform = transform["transform"]

    required = (
        "centering_mode",
        "rotation",
        "scale",
        "source_mean",
        "target_mean",
        "source_dim",
        "target_dim",
        "dimension",
    )

    missing = [
        key for key in required
        if key not in transform
    ]

    if missing:
        raise ValueError(
            f"transform incomplet ; clés manquantes : {missing}."
        )

    mode = _check_centering_mode(transform["centering_mode"])
    X_in = _validate_points(coords, "coords")

    dimension = int(transform["dimension"])
    source_dim = int(transform["source_dim"])
    target_dim = int(transform["target_dim"])

    if X_in.shape[1] != source_dim:
        raise ValueError(
            f"coords doit avoir {source_dim} colonne(s) ; "
            f"reçu {X_in.shape[1]}."
        )

    if dimension < 1 or source_dim < 1 or target_dim < 1:
        raise ValueError("Dimensions de transformation invalides.")

    R = _as_finite_array(
        transform["rotation"],
        "transform.rotation",
        ndim=2,
    )

    if R.shape != (dimension, dimension):
        raise ValueError(
            "transform.rotation a une forme incompatible."
        )

    scale = float(transform["scale"])

    if not np.isfinite(scale) or scale < 0:
        raise ValueError(
            "transform.scale doit être fini et non négatif."
        )

    X = np.zeros((X_in.shape[0], dimension))
    X[:, :source_dim] = X_in

    if mode == "common":
        sm = _as_finite_array(
            transform["source_mean"],
            "transform.source_mean",
            ndim=1,
        )
        tm = _as_finite_array(
            transform["target_mean"],
            "transform.target_mean",
            ndim=1,
        )

        if sm.shape != (dimension,) or tm.shape != (dimension,):
            raise ValueError(
                "Les centres Procrustes ont des dimensions incompatibles."
            )

        aligned = scale * ((X - sm) @ R) + tm
        return aligned[:, :target_dim]

    labels = _prepare_condition_labels(
        condition_labels,
        X.shape[0],
        "condition_labels",
    )

    sm_by = transform.get("source_means_by_condition")
    tm_by = transform.get("target_means_by_condition")

    if not sm_by or not tm_by:
        raise ValueError(
            "transform ne contient pas les moyennes par condition."
        )

    aligned = np.empty_like(X)

    for label in np.unique(labels):
        key = label.item() if hasattr(label, "item") else label

        if key not in sm_by or key not in tm_by:
            raise ValueError(
                f"Condition {key!r} absente de l'entraînement : "
                "aucun centre enregistré. Aucun repli vers le centre "
                "commun n'est appliqué."
            )

        mask = labels == label

        sm = _as_finite_array(
            sm_by[key],
            "source_means_by_condition",
            ndim=1,
        )
        tm = _as_finite_array(
            tm_by[key],
            "target_means_by_condition",
            ndim=1,
        )

        if sm.shape != (dimension,) or tm.shape != (dimension,):
            raise ValueError(
                "Un centre par condition possède une dimension invalide."
            )

        aligned[mask] = scale * ((X[mask] - sm) @ R) + tm

    return aligned[:, :target_dim]


# ============================================================================
# 3. ESTIMATION D'UNE MÉTRIQUE LOCALE SPD
# ============================================================================

def fit_local_metric_at_anchor(
    displacements: ArrayLike,
    thresholds: ArrayLike,
    G_prior: Optional[ArrayLike] = None,
    shrink_lambda: float = _DEFAULT_SHRINK_LAMBDA,
    reg: float = _DEFAULT_REG,
    max_iter: int = 2000,
    eig_floor: float = _DEFAULT_EIG_FLOOR,
) -> NDArray[np.floating]:
    """
    Estime une métrique locale SPD à partir de sondes directionnelles.

    Modèle :
        v_i.T @ G @ v_i ≈ 1 / u_i²
        v_i = d_i / ||d_i||

    Paramétrisation :
        G = A @ A.T
        A triangulaire inférieure.

    Fonction objectif :
        moyenne((v.T G v - 1/u²)²)
        + shrink_lambda * ||G - G_prior||_F²
        + reg * ||A - L_prior||_F²

    shrink_lambda et reg sont des poids de pénalité non négatifs.
    Ils ne sont pas des coefficients de combinaison convexe.

    Identifiabilité :
        une métrique symétrique d'ordre d comporte d(d+1)/2 paramètres.
        Si les directions sondées ne permettent pas de les identifier,
        la solution dépend du prior et de la régularisation.

    L'optimisation est réalisée dans le facteur A. La fonction objectif
    exprimée en G comporte une structure quadratique, mais la
    paramétrisation G = A A.T introduit une optimisation non convexe
    et redondante dans l'espace des facteurs.
    """
    D = _as_finite_array(
        displacements,
        "displacements",
        ndim=2,
    )

    u = _validate_nonnegative(
        thresholds,
        "thresholds",
    ).reshape(-1)

    if D.shape[0] != len(u):
        raise ValueError(
            "Le nombre de déplacements et de seuils doit correspondre."
        )

    if D.shape[0] < 1 or D.shape[1] < 1:
        raise ValueError("Les sondes ne peuvent pas être vides.")

    if np.any(u <= 0):
        raise ValueError(
            "Tous les seuils doivent être strictement positifs."
        )

    if np.any(u < math.sqrt(np.finfo(float).tiny)):
        raise ValueError(
            "Un seuil est trop petit pour calculer 1/u² de façon stable."
        )

    shrink_lambda = float(shrink_lambda)
    reg = float(reg)

    if not np.isfinite(shrink_lambda) or shrink_lambda < 0:
        raise ValueError(
            "shrink_lambda doit être fini et non négatif."
        )

    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg doit être fini et non négatif.")

    max_iter = _positive_integer(max_iter, "max_iter")

    norms = np.linalg.norm(D, axis=1)

    if np.any(norms <= np.finfo(float).eps):
        raise ValueError(
            "Un déplacement est nul ou numériquement dégénéré."
        )

    V = D / norms[:, None]

    d = D.shape[1]
    n_parameters = d * (d + 1) // 2

    iu, ju = np.triu_indices(d)

    design = (
        V[:, iu]
        * V[:, ju]
        * np.where(iu == ju, 1.0, 2.0)
    )

    design_rank = int(np.linalg.matrix_rank(design))

    if design_rank < n_parameters:
        warnings.warn(
            "Les directions ne permettent pas d'identifier seules "
            "tous les paramètres de la métrique. La solution dépend "
            "du prior et de la régularisation.",
            RuntimeWarning,
            stacklevel=2,
        )

    target = 1.0 / (u ** 2)

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "Les seuils produisent des cibles numériques non finies."
        )

    if G_prior is None:
        G0 = np.eye(d)
    else:
        G0 = _as_finite_array(
            G_prior,
            "G_prior",
            ndim=2,
        )

        if G0.shape != (d, d):
            raise ValueError(
                "G_prior doit avoir les mêmes dimensions que la métrique."
            )

        G0 = _ensure_spd(
            G0,
            eig_floor=eig_floor,
        )

    L0 = np.linalg.cholesky(G0)

    tril_i, tril_j = np.tril_indices(d)
    theta0 = L0[tril_i, tril_j].copy()

    def unpack(theta: NDArray[np.floating]) -> NDArray[np.floating]:
        A = np.zeros((d, d))
        A[tril_i, tril_j] = theta
        return A

    def objective(theta: NDArray[np.floating]) -> float:
        A = unpack(theta)
        G = A @ A.T

        predicted = np.einsum(
            "ni,ij,nj->n",
            V,
            G,
            V,
        )

        data_loss = float(
            np.mean((predicted - target) ** 2)
        )

        prior_loss = float(
            np.sum((G - G0) ** 2)
        )

        factor_loss = float(
            np.sum((A - L0) ** 2)
        )

        return (
            data_loss
            + shrink_lambda * prior_loss
            + reg * factor_loss
        )

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    if (
        not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
    ):
        raise RuntimeError(
            "Optimisation de la métrique locale échouée : "
            f"{result.message}"
        )

    if not result.success:
        warnings.warn(
            "L-BFGS-B n'a pas déclaré de convergence "
            f"({result.message}) ; la solution courante est conservée.",
            RuntimeWarning,
            stacklevel=2,
        )

    A_opt = unpack(result.x)

    return _ensure_spd(
        A_opt @ A_opt.T,
        eig_floor=eig_floor,
    )


# ============================================================================
# 4. INTERPOLATION DES MÉTRIQUES LOCALES
# ============================================================================

def _check_interpolation(value: Any) -> InterpolationMode:
    if value not in _INTERPOLATIONS:
        raise ValueError(
            f"interpolation doit appartenir à {_INTERPOLATIONS} ; "
            f"valeur reçue : {value!r}."
        )

    return str(value)  # type: ignore[return-value]


def _check_bandwidth(bandwidth: Any) -> float:
    bandwidth = float(bandwidth)

    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(
            "bandwidth doit être strictement positif et fini."
        )

    return bandwidth


def _prepare_anchor_metrics(
    anchor_points: ArrayLike,
    anchor_metrics: ArrayLike,
    eig_floor: float,
    interpolation: InterpolationMode = "linear",
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Valide les ancres et prépare les métriques.

    En interpolation linéaire, retourne les métriques SPD.

    En interpolation log-euclidienne, retourne leurs logarithmes
    matriciels.
    """
    interpolation = _check_interpolation(interpolation)

    anchors = _validate_points(
        anchor_points,
        "anchor_points",
    )

    metrics = _as_finite_array(
        anchor_metrics,
        "anchor_metrics",
        ndim=3,
    )

    n_anchors, d = anchors.shape

    if metrics.shape != (n_anchors, d, d):
        raise ValueError(
            "anchor_metrics doit avoir la forme "
            "(n_anchors, dimension, dimension)."
        )

    spd = _ensure_spd(
        metrics,
        eig_floor=eig_floor,
    )

    if interpolation == "log_euclidean":
        eigvals, eigvecs = np.linalg.eigh(spd)

        logs = (
            eigvecs * np.log(eigvals)[..., None, :]
        ) @ np.swapaxes(eigvecs, -1, -2)

        return (
            anchors,
            0.5 * (logs + np.swapaxes(logs, -1, -2)),
        )

    return anchors, spd


def _interpolate_prepared(
    X: NDArray[np.floating],
    anchors: NDArray[np.floating],
    spd_metrics: NDArray[np.floating],
    bandwidth: float,
    eig_floor: float,
    interpolation: InterpolationMode = "linear",
) -> NDArray[np.floating]:
    """Interpolation par lot à partir d'ancres déjà préparées."""
    distances_sq = np.sum(
        (X[:, None, :] - anchors[None, :, :]) ** 2,
        axis=2,
    )

    log_w = -0.5 * distances_sq / (bandwidth ** 2)

    if not np.all(np.isfinite(log_w)):
        raise ValueError(
            "Les distances ou le bandwidth produisent des poids instables."
        )

    # Stabilisation log-sum-exp.
    # Lorsque le noyau est très local, l'ancre la plus proche domine.
    # Aucun repli discontinu vers une moyenne globale n'est effectué.
    log_w -= np.max(log_w, axis=1, keepdims=True)

    weights = np.exp(log_w)
    weight_sum = np.sum(weights, axis=1, keepdims=True)

    if (
        np.any(weight_sum <= 0)
        or not np.all(np.isfinite(weight_sum))
    ):
        raise RuntimeError(
            "Impossible de normaliser les poids d'interpolation."
        )

    weights /= weight_sum

    G = np.einsum(
        "mn,nij->mij",
        weights,
        spd_metrics,
    )

    if interpolation == "log_euclidean":
        G = 0.5 * (
            G + np.swapaxes(G, -1, -2)
        )

        eigvals, eigvecs = np.linalg.eigh(G)

        G = (
            eigvecs * np.exp(eigvals)[..., None, :]
        ) @ np.swapaxes(eigvecs, -1, -2)

        return 0.5 * (
            G + np.swapaxes(G, -1, -2)
        )

    return _ensure_spd(
        G,
        eig_floor=eig_floor,
    )


def interpolate_local_metric_batch(
    query_points: ArrayLike,
    anchor_points: ArrayLike,
    anchor_metrics: ArrayLike,
    bandwidth: float,
    eig_floor: float = _DEFAULT_EIG_FLOOR,
    interpolation: InterpolationMode = "linear",
) -> NDArray[np.floating]:
    """
    Interpole G(x) pour un lot de points.

    Entrée :
        query_points : (m, d)

    Sortie :
        métriques interpolées : (m, d, d)

    Modes :
        linear :
            G(x) = somme_i w_i G_i

        log_euclidean :
            G(x) = exp(somme_i w_i log(G_i))
    """
    interpolation = _check_interpolation(interpolation)

    X = _validate_points(
        query_points,
        "query_points",
    )

    bandwidth = _check_bandwidth(bandwidth)

    anchors, prepared = _prepare_anchor_metrics(
        anchor_points,
        anchor_metrics,
        eig_floor,
        interpolation,
    )

    if X.shape[1] != anchors.shape[1]:
        raise ValueError(
            "Dimensions incompatibles entre query_points et anchor_points."
        )

    return _interpolate_prepared(
        X,
        anchors,
        prepared,
        bandwidth,
        eig_floor,
        interpolation,
    )


def interpolate_local_metric(
    query_point: ArrayLike,
    anchor_points: ArrayLike,
    anchor_metrics: ArrayLike,
    bandwidth: float,
    eig_floor: float = _DEFAULT_EIG_FLOOR,
    interpolation: InterpolationMode = "linear",
) -> NDArray[np.floating]:
    """Interpole la métrique locale en un seul point."""
    x = _as_finite_array(
        query_point,
        "query_point",
        ndim=1,
    )

    return interpolate_local_metric_batch(
        x[None, :],
        anchor_points,
        anchor_metrics,
        bandwidth,
        eig_floor,
        interpolation,
    )[0]


# ============================================================================
# 5. INTÉGRATION DE DISTANCE SUR UN CHEMIN
# ============================================================================

def path_integrated_distance(
    waypoints: ArrayLike,
    anchor_points: ArrayLike,
    anchor_metrics: ArrayLike,
    bandwidth: float,
    n_substeps: int = 20,
    eig_floor: float = _DEFAULT_EIG_FLOOR,
    interpolation: InterpolationMode = "linear",
) -> float:
    """
    Intègre une longueur métrique sur un chemin polygonal.

    Élément de longueur :
        ds = sqrt(dx.T @ G(x) @ dx)

    La métrique est évaluée au milieu de chaque sous-segment.

    Cette fonction calcule la longueur du chemin fourni.
    Elle ne recherche pas une géodésique minimale.
    """
    path = _validate_points(
        waypoints,
        "waypoints",
        min_points=1,
    )

    n_substeps = _positive_integer(
        n_substeps,
        "n_substeps",
    )

    bandwidth = _check_bandwidth(bandwidth)
    interpolation = _check_interpolation(interpolation)

    anchors, prepared = _prepare_anchor_metrics(
        anchor_points,
        anchor_metrics,
        eig_floor,
        interpolation,
    )

    if path.shape[1] != anchors.shape[1]:
        raise ValueError(
            "waypoints et anchor_points doivent avoir la même dimension."
        )

    if path.shape[0] == 1:
        _interpolate_prepared(
            path,
            anchors,
            prepared,
            bandwidth,
            eig_floor,
            interpolation,
        )
        return 0.0

    starts = path[:-1]
    deltas = path[1:] - starts

    t_mid = (
        np.arange(n_substeps) + 0.5
    ) / n_substeps

    mids = (
        starts[:, None, :]
        + t_mid[None, :, None] * deltas[:, None, :]
    ).reshape(-1, path.shape[1])

    G = _interpolate_prepared(
        mids,
        anchors,
        prepared,
        bandwidth,
        eig_floor,
        interpolation,
    )

    deltas_rep = np.repeat(
        deltas,
        n_substeps,
        axis=0,
    )

    local_sq = np.einsum(
        "mi,mij,mj->m",
        deltas_rep,
        G,
        deltas_rep,
    )

    # Correction : NaN ou inf ne doivent jamais être propagés
    # silencieusement jusqu'au résultat final.
    if not np.all(np.isfinite(local_sq)):
        raise RuntimeError(
            "La métrique a produit une longueur quadratique non finie."
        )

    if np.any(local_sq < -1e-10):
        raise RuntimeError(
            "La métrique a produit une longueur quadratique négative."
        )

    return float(
        np.sum(np.sqrt(np.maximum(local_sq, 0.0)))
        / n_substeps
    )


# ============================================================================
# 6. VOLUME LOCAL
# ============================================================================

def local_volume_in_region(
    region_samples: ArrayLike,
    anchor_points: ArrayLike,
    anchor_metrics: ArrayLike,
    bandwidth: float,
    region_base_volume: float,
    eig_floor: float = _DEFAULT_EIG_FLOOR,
    chunk_size: int = 4096,
    interpolation: InterpolationMode = "linear",
) -> Dict[str, float]:
    """
    Estime un volume transformé par une métrique locale.

    Densité volumique :
        density(x) = sqrt(det(G(x)))

    Estimation :
        V ≈ V_base * moyenne(density(x_i))

    L'erreur standard empirique suppose des échantillons indépendants
    et représentatifs de la région. Elle ne constitue pas une garantie
    d'incertitude pour des échantillons déterministes ou corrélés.
    """
    samples = _validate_points(
        region_samples,
        "region_samples",
        min_points=1,
    )

    bandwidth = _check_bandwidth(bandwidth)
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    interpolation = _check_interpolation(interpolation)

    anchors, prepared = _prepare_anchor_metrics(
        anchor_points,
        anchor_metrics,
        eig_floor,
        interpolation,
    )

    base_volume = float(region_base_volume)

    if not np.isfinite(base_volume) or base_volume <= 0:
        raise ValueError(
            "region_base_volume doit être strictement positif et fini."
        )

    if samples.shape[1] != anchors.shape[1]:
        raise ValueError(
            "region_samples et anchor_points doivent avoir "
            "la même dimension."
        )

    log_max = math.log(np.finfo(float).max)
    parts: List[NDArray[np.floating]] = []

    for start in range(0, len(samples), chunk_size):
        block = samples[start:start + chunk_size]

        G = _interpolate_prepared(
            block,
            anchors,
            prepared,
            bandwidth,
            eig_floor,
            interpolation,
        )

        sign, logdet = np.linalg.slogdet(G)

        if (
            np.any(sign <= 0)
            or not np.all(np.isfinite(logdet))
        ):
            raise RuntimeError(
                "La métrique ne possède pas un déterminant "
                "positif fini."
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

    if not np.isfinite(mean_density):
        raise RuntimeError(
            "La densité moyenne produite est non finie."
        )

    if n > 1:
        se_density = float(
            np.std(densities, ddof=1) / math.sqrt(n)
        )
    else:
        se_density = 0.0

    estimated_volume = base_volume * mean_density
    standard_error = base_volume * se_density

    if (
        not np.isfinite(estimated_volume)
        or not np.isfinite(standard_error)
    ):
        raise OverflowError(
            "Le volume estimé dépasse la capacité numérique."
        )

    return {
        "estimated_volume": float(estimated_volume),
        "mean_density": mean_density,
        "standard_error": float(standard_error),
        "n_samples": int(n),
    }


# ============================================================================
# 7. CALIBRATION LINÉAIRE
# ============================================================================

def fit_linear_scale(
    raw_distances: ArrayLike,
    true_distances: ArrayLike,
    fit_intercept: bool = True,
) -> LinearScaleModel:
    """
    Ajuste :
        true ≈ a + b * raw

    La pente doit être non négative pour préserver l'ordre des distances.
    """
    x = _validate_nonnegative(
        raw_distances,
        "raw_distances",
    ).reshape(-1)

    y = _validate_nonnegative(
        true_distances,
        "true_distances",
    ).reshape(-1)

    if x.shape != y.shape:
        raise ValueError(
            "Les deux vecteurs doivent avoir la même longueur."
        )

    if len(x) < 2:
        raise ValueError(
            "Au moins deux observations sont nécessaires."
        )

    eps = np.finfo(float).eps

    if fit_intercept:
        if np.ptp(x) <= eps:
            raise ValueError(
                "Les distances brutes sont constantes : "
                "pente et interception non identifiables."
            )

        design = np.column_stack([
            np.ones_like(x),
            x,
        ])

        coefficients, _, rank, _ = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )

        if rank < 2:
            raise ValueError(
                "Le problème de régression est de rang insuffisant."
            )

        a, b = map(float, coefficients)

    else:
        denominator = float(np.dot(x, x))

        if denominator <= eps:
            raise ValueError(
                "Les distances brutes sont nulles."
            )

        a = 0.0
        b = float(np.dot(x, y) / denominator)

    if not np.isfinite(a) or not np.isfinite(b):
        raise RuntimeError(
            "La régression a produit des coefficients non finis."
        )

    if b < -1e-12:
        raise ValueError(
            "La pente estimée est négative : "
            "calibration linéaire monotone impossible."
        )

    return {
        "a": a,
        "b": max(0.0, b),
    }


def apply_linear_scale(
    raw_distances: ArrayLike,
    model: Mapping[str, float],
) -> NDArray[np.floating]:
    """Applique une calibration linéaire non décroissante."""
    x = _as_finite_array(
        raw_distances,
        "raw_distances",
    )

    if not isinstance(model, Mapping):
        raise ValueError(
            "model doit être un dictionnaire."
        )

    try:
        a = float(model["a"])
        b = float(model["b"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "model doit contenir les coefficients numériques a et b."
        ) from exc

    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError(
            "Les coefficients doivent être finis."
        )

    if b < 0:
        raise ValueError(
            "La pente doit être non négative."
        )

    return np.maximum(a + b * x, 0.0)


# ============================================================================
# 8. CALIBRATION MONOTONE : PAVA
# ============================================================================

def _pool_adjacent_violators(
    values: ArrayLike,
    weights: ArrayLike,
) -> NDArray[np.floating]:
    """Régression isotone non décroissante par PAVA."""
    y = _as_finite_array(
        values,
        "values",
        ndim=1,
    )

    w = _as_finite_array(
        weights,
        "weights",
        ndim=1,
    )

    if len(y) != len(w):
        raise ValueError(
            "values et weights doivent avoir la même longueur."
        )

    if len(y) == 0:
        raise ValueError(
            "PAVA ne peut pas traiter un vecteur vide."
        )

    if np.any(w <= 0):
        raise ValueError(
            "Tous les poids doivent être strictement positifs."
        )

    vals: List[float] = []
    wts: List[float] = []
    starts: List[int] = []
    ends: List[int] = []

    for i, (value, weight) in enumerate(zip(y, w)):
        vals.append(float(value))
        wts.append(float(weight))
        starts.append(i)
        ends.append(i + 1)

        while len(vals) >= 2 and vals[-2] > vals[-1]:
            w1, w2 = wts[-2], wts[-1]
            merged_w = w1 + w2

            merged_v = (
                w1 * vals[-2] + w2 * vals[-1]
            ) / merged_w

            start = starts[-2]
            end = ends[-1]

            vals[-2:] = [merged_v]
            wts[-2:] = [merged_w]
            starts[-2:] = [start]
            ends[-2:] = [end]

    result = np.empty(len(y))

    for value, start, end in zip(vals, starts, ends):
        result[start:end] = value

    return result


def fit_monotone_binned(
    raw_distances: ArrayLike,
    true_distances: ArrayLike,
    n_bins: int = 6,
) -> MonotoneModel:
    """
    Calibre les distances par classes de quantiles.

    Les moyennes de chaque classe sont projetées sur une séquence
    non décroissante par PAVA.
    """
    x = _validate_nonnegative(
        raw_distances,
        "raw_distances",
    ).reshape(-1)

    y = _validate_nonnegative(
        true_distances,
        "true_distances",
    ).reshape(-1)

    if x.shape != y.shape:
        raise ValueError(
            "Les vecteurs doivent avoir la même longueur."
        )

    if len(x) < 2:
        raise ValueError(
            "Au moins deux observations sont nécessaires."
        )

    n_bins = _positive_integer(
        n_bins,
        "n_bins",
    )

    if n_bins < 2:
        raise ValueError(
            "n_bins doit être supérieur ou égal à 2."
        )

    if np.ptp(x) <= np.finfo(float).eps:
        return {
            "type": "monotone_constant",
            "edges": np.array([
                float(np.min(x)),
                float(np.max(x)),
            ]),
            "centers": np.array([
                float(np.mean(x)),
            ]),
            "values": np.array([
                float(np.mean(y)),
            ]),
        }

    n_bins = min(n_bins, len(x))

    edges = np.unique(
        np.quantile(
            x,
            np.linspace(0.0, 1.0, n_bins + 1),
        )
    )

    if len(edges) < 2:
        return {
            "type": "monotone_constant",
            "edges": edges,
            "centers": np.array([float(np.mean(x))]),
            "values": np.array([float(np.mean(y))]),
        }

    bin_ids = np.digitize(
        x,
        edges[1:-1],
        right=False,
    )

    centers_list = []
    means_list = []
    weights_list = []

    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id

        if not np.any(mask):
            continue

        centers_list.append(
            float(np.mean(x[mask]))
        )

        means_list.append(
            float(np.mean(y[mask]))
        )

        weights_list.append(
            float(np.sum(mask))
        )

    centers = np.asarray(centers_list)
    means = np.asarray(means_list)
    weights = np.asarray(weights_list)

    order = np.argsort(centers)

    centers = centers[order]
    means = means[order]
    weights = weights[order]

    return {
        "type": "monotone_piecewise",
        "edges": edges,
        "centers": centers,
        "values": _pool_adjacent_violators(
            means,
            weights,
        ),
        "weights": weights,
    }


def apply_monotone_binned(
    raw_distances: ArrayLike,
    model: Mapping[str, Any],
) -> NDArray[np.floating]:
    """Applique une calibration monotone par interpolation linéaire."""
    x = _as_finite_array(
        raw_distances,
        "raw_distances",
    )

    if not isinstance(model, Mapping):
        raise ValueError(
            "model doit être un dictionnaire."
        )

    model_type = model.get("type")

    try:
        centers = _as_finite_array(
            model["centers"],
            "model.centers",
            ndim=1,
        )

        values = _validate_nonnegative(
            model["values"],
            "model.values",
        ).reshape(-1)

    except KeyError as exc:
        raise ValueError(
            "Le modèle monotone est incomplet."
        ) from exc

    if (
        len(centers) == 0
        or len(centers) != len(values)
    ):
        raise ValueError(
            "centers et values doivent avoir la même longueur non nulle."
        )

    if np.any(np.diff(centers) <= 0):
        raise ValueError(
            "Les centres doivent être strictement croissants."
        )

    if np.any(np.diff(values) < -1e-12):
        raise ValueError(
            "Les valeurs du modèle doivent être non décroissantes."
        )

    if model_type == "monotone_constant":
        return np.full_like(
            x,
            float(values[0]),
            dtype=float,
        )

    if model_type != "monotone_piecewise":
        raise ValueError(
            f"Type de modèle monotone inconnu : {model_type}"
        )

    return np.interp(
        x,
        centers,
        values,
        left=values[0],
        right=values[-1],
    )


# ============================================================================
# 9. ÉVALUATION D'UNE CALIBRATION
# ============================================================================

def evaluate_calibration_split(
    raw_distances: ArrayLike,
    true_distances: ArrayLike,
    predicted_distances: ArrayLike,
) -> CalibrationMetrics:
    """
    Évalue une calibration sur un ensemble distinct de l'ajustement.

    Pente :
        true ≈ intercept + slope * predicted

    Spearman et pente valent None lorsqu'ils ne sont pas définis.
    """
    raw = _validate_nonnegative(
        raw_distances,
        "raw_distances",
    ).reshape(-1)

    true = _validate_nonnegative(
        true_distances,
        "true_distances",
    ).reshape(-1)

    pred = _validate_nonnegative(
        predicted_distances,
        "predicted_distances",
    ).reshape(-1)

    if not (
        len(raw) == len(true) == len(pred)
    ):
        raise ValueError(
            "Les trois vecteurs doivent avoir la même longueur."
        )

    if len(raw) < 2:
        raise ValueError(
            "Au moins deux observations sont nécessaires."
        )

    residual = pred - true

    mae = float(
        np.mean(np.abs(residual))
    )

    rmse = float(
        np.sqrt(np.mean(residual ** 2))
    )

    if np.ptp(pred) > np.finfo(float).eps:
        slope = float(
            np.cov(pred, true, ddof=0)[0, 1]
            / np.var(pred)
        )
    else:
        slope = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation = spearmanr(pred, true)[0]

    spearman = (
        float(correlation)
        if np.isfinite(correlation)
        else None
    )

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
    raw_distances: ArrayLike,
    true_distances: ArrayLike,
    seed: int = 42,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
    slope_tolerance: float = _SLOPE_TOLERANCE,
    min_spearman: float = _MIN_SPEARMAN,
) -> Dict[str, Any]:
    """
    Compare une calibration linéaire et une calibration monotone.

    Séparation :
        calibration : ajustement des modèles ;
        validation  : sélection du modèle ;
        test        : évaluation finale.

    Les ensembles sont disjoints au niveau des paires.
    Des paires partageant des nœuds peuvent toutefois conserver
    une dépendance entre les ensembles.

    Minimum :
        8 paires.

    Les seuils slope_tolerance et min_spearman sont des seuils
    d'alerte opérationnelle. Ils ne constituent pas un test
    statistique formel d'équivalence.
    """
    raw = _validate_nonnegative(
        raw_distances,
        "raw_distances",
    ).reshape(-1)

    true = _validate_nonnegative(
        true_distances,
        "true_distances",
    ).reshape(-1)

    if len(raw) != len(true):
        raise ValueError(
            "raw_distances et true_distances doivent avoir "
            "la même longueur."
        )

    n = len(raw)

    if n < 8:
        raise ValueError(
            "Au moins 8 paires sont nécessaires pour constituer "
            "les trois ensembles sans réutiliser les observations."
        )

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ValueError(
            "seed doit être un entier."
        )

    test_fraction = float(test_fraction)
    validation_fraction = float(validation_fraction)

    if (
        not np.isfinite(test_fraction)
        or not np.isfinite(validation_fraction)
        or not (0.0 < test_fraction < 1.0)
        or not (0.0 < validation_fraction < 1.0)
    ):
        raise ValueError(
            "test_fraction et validation_fraction doivent être "
            "finies et dans (0, 1)."
        )

    if test_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "La somme test_fraction + validation_fraction doit être < 1."
        )

    slope_tolerance = float(slope_tolerance)
    min_spearman = float(min_spearman)

    # Correction : les comparaisons ordinaires avec NaN renvoient False.
    # Les seuils doivent donc être explicitement finis.
    if (
        not np.isfinite(slope_tolerance)
        or not np.isfinite(min_spearman)
        or slope_tolerance < 0
        or min_spearman < -1
        or min_spearman > 1
    ):
        raise ValueError(
            "Seuils d'alerte invalides."
        )

    indices = np.random.default_rng(
        int(seed)
    ).permutation(n)

    n_test = max(
        2,
        int(round(test_fraction * n)),
    )

    n_validation = max(
        2,
        int(round(validation_fraction * n)),
    )

    n_calibration = n - n_test - n_validation

    if n_calibration < 4:
        raise ValueError(
            "Nombre insuffisant de paires pour la calibration."
        )

    idx_cal = indices[:n_calibration]

    idx_val = indices[
        n_calibration:n_calibration + n_validation
    ]

    idx_test = indices[
        n_calibration + n_validation:
    ]

    if (
        set(idx_cal) & set(idx_val)
        or set(idx_cal) & set(idx_test)
        or set(idx_val) & set(idx_test)
    ):
        raise RuntimeError(
            "Les ensembles ne sont pas disjoints."
        )

    # ------------------------------------------------------------------
    # Ajustement sur calibration uniquement
    # ------------------------------------------------------------------

    linear_model = None
    linear_unavailable = None

    try:
        linear_model = fit_linear_scale(
            raw[idx_cal],
            true[idx_cal],
            fit_intercept=True,
        )
    except ValueError as exc:
        linear_unavailable = str(exc)

    monotone_model = fit_monotone_binned(
        raw[idx_cal],
        true[idx_cal],
        n_bins=min(
            6,
            max(2, len(idx_cal) // 4),
        ),
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    monotone_val_metrics = evaluate_calibration_split(
        raw[idx_val],
        true[idx_val],
        apply_monotone_binned(
            raw[idx_val],
            monotone_model,
        ),
    )

    if linear_model is not None:
        linear_val_metrics = evaluate_calibration_split(
            raw[idx_val],
            true[idx_val],
            apply_linear_scale(
                raw[idx_val],
                linear_model,
            ),
        )
    else:
        linear_val_metrics = None

    # ------------------------------------------------------------------
    # Sélection du modèle sans réajustement
    # ------------------------------------------------------------------

    if (
        linear_val_metrics is not None
        and linear_val_metrics["rmse"]
        <= monotone_val_metrics["rmse"]
    ):
        selected_name = "linear"
        selected_model = linear_model
        selected_validation = linear_val_metrics

    else:
        selected_name = monotone_model["type"]
        selected_model = monotone_model
        selected_validation = monotone_val_metrics

    # ------------------------------------------------------------------
    # Test final
    # ------------------------------------------------------------------

    if selected_name == "linear":
        test_predictions = apply_linear_scale(
            raw[idx_test],
            selected_model,
        )
    else:
        test_predictions = apply_monotone_binned(
            raw[idx_test],
            selected_model,
        )

    test_metrics = evaluate_calibration_split(
        raw[idx_test],
        true[idx_test],
        test_predictions,
    )

    slope = test_metrics["slope_pred_vs_ref"]
    rho = test_metrics["spearman"]

    slope_warning = (
        slope is None
        or abs(slope - 1.0) > slope_tolerance
    )

    correlation_warning = (
        rho is None
        or rho < min_spearman
    )

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
            "test_fraction": test_fraction,
            "validation_fraction": validation_fraction,
        },

        "candidate_validation": {
            "linear": linear_val_metrics,
            "monotone": monotone_val_metrics,
        },

        "diagnostics": {
            "non_equivalence_warning": bool(
                slope_warning or correlation_warning
            ),
            "slope_warning": bool(slope_warning),
            "correlation_warning": bool(correlation_warning),
            "linear_unavailable": linear_unavailable,
            "slope_tolerance": slope_tolerance,
            "min_spearman": min_spearman,

            "warning_rule": (
                "Alerte opérationnelle si la pente test est absente "
                "ou s'écarte de 1 de plus que slope_tolerance, ou si "
                "Spearman est absent ou inférieur à min_spearman. "
                "Ces seuils ne constituent pas un test statistique "
                "formel d'équivalence."
            ),

            "independence_warning": (
                "La séparation est réalisée au niveau des paires. "
                "Si plusieurs paires partagent des nœuds, une "
                "dépendance entre calibration, validation et test "
                "peut subsister. Une séparation par groupes de nœuds "
                "nécessite les identifiants des nœuds."
            ),
        },
    }


# ============================================================================
# 11. API HISTORIQUE — MODULE 1 D'ORIGINE
# ============================================================================
#
# C1 : logarithme et exponentielle matriciels compatibles avec les lots.
#      Validation de floor dans spd_logm.
#
# C2 : Procrustes figé délégué au noyau principal.
#
# C3 : directions normalisées dans estimate_local_spd_metric.
#
# C4 : interpolation log-euclidienne stabilisée par log-sum-exp.
#      Aucun repli discontinu vers une moyenne globale.
#
# C5 : intégration au point milieu, sans biais n/(n-1).
#
# C6 : déterminant calculé par slogdet ; regroupement des valeurs d_geo
#      identiques avant PAVA.


def spd_logm(
    G: ArrayLike,
    floor: float = 1e-12,
) -> NDArray[np.floating]:
    """
    Logarithme matriciel d'une matrice SPD ou d'un lot (..., d, d).

    Les valeurs propres sont planchées à floor pour stabiliser
    le logarithme. Le plancher doit être strictement positif et fini.
    """
    A = _as_finite_array(G, "G")

    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("G doit être carrée.")

    floor = float(floor)

    # Correction C1 : même règle de validation que _ensure_spd.
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError(
            "floor doit être strictement positif et fini."
        )

    S = 0.5 * (
        A + np.swapaxes(A, -1, -2)
    )

    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, floor)

    return (
        eigvecs * np.log(eigvals)[..., None, :]
    ) @ np.swapaxes(eigvecs, -1, -2)


def spd_expm(
    L: ArrayLike,
) -> NDArray[np.floating]:
    """
    Exponentielle matricielle d'une matrice symétrique ou d'un lot.

    La fonction symétrise l'entrée avant diagonalisation.
    Elle ne vérifie pas que l'entrée provient effectivement du
    logarithme d'une matrice SPD.
    """
    A = _as_finite_array(L, "L")

    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("L doit être carrée.")

    S = 0.5 * (
        A + np.swapaxes(A, -1, -2)
    )

    eigvals, eigvecs = np.linalg.eigh(S)

    result = (
        eigvecs * np.exp(eigvals)[..., None, :]
    ) @ np.swapaxes(eigvecs, -1, -2)

    return 0.5 * (
        result + np.swapaxes(result, -1, -2)
    )


def fit_frozen_procrustes(
    X_source: ArrayLike,
    X_target: ArrayLike,
) -> Dict[str, Any]:
    """
    Ajuste une similitude Procrustes à rotation propre (det = +1).

    Retourne :
        R
        s
        mu_source
        mu_target
        transform
    """
    result = procrustes_align_bridges(
        X_source,
        X_target,
        allow_reflection=False,
        centering_mode="common",
    )

    transform = result["transform"]

    return {
        "R": transform["rotation"],
        "s": float(transform["scale"]),
        "mu_source": transform["source_mean"],
        "mu_target": transform["target_mean"],
        "transform": transform,
    }


def transform_frozen_procrustes(
    X_new: ArrayLike,
    procrustes_params: Mapping[str, Any],
) -> NDArray[np.floating]:
    """Applique une transformation Procrustes figée."""
    if not isinstance(procrustes_params, Mapping):
        raise ValueError(
            "procrustes_params doit être un dictionnaire."
        )

    if isinstance(
        procrustes_params.get("transform"),
        Mapping,
    ):
        return apply_procrustes_transform(
            X_new,
            procrustes_params["transform"],
        )

    try:
        R = _as_finite_array(
            procrustes_params["R"],
            "R",
            ndim=2,
        )

        s = float(procrustes_params["s"])

        mu_src = _as_finite_array(
            procrustes_params["mu_source"],
            "mu_source",
            ndim=1,
        )

        mu_tgt = _as_finite_array(
            procrustes_params["mu_target"],
            "mu_target",
            ndim=1,
        )

    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "procrustes_params doit contenir R, s, "
            "mu_source et mu_target."
        ) from exc

    if not np.isfinite(s) or s < 0:
        raise ValueError(
            "Le facteur d'échelle doit être fini et non négatif."
        )

    if R.shape[0] != R.shape[1]:
        raise ValueError("R doit être carrée.")

    if len(mu_src) != R.shape[0] or len(mu_tgt) != R.shape[0]:
        raise ValueError(
            "Les centres et la rotation ont des dimensions incompatibles."
        )

    transform = {
        "centering_mode": "common",
        "rotation": R,
        "scale": s,
        "source_mean": mu_src,
        "target_mean": mu_tgt,
        "source_dim": int(len(mu_src)),
        "target_dim": int(len(mu_tgt)),
        "dimension": int(R.shape[0]),
    }

    return apply_procrustes_transform(
        X_new,
        transform,
    )


def estimate_local_spd_metric(
    directions: ArrayLike,
    jnds: ArrayLike,
    dim: int = 2,
    reg_diag: float = 1e-4,
) -> NDArray[np.floating]:
    """
    Estime une métrique SPD à partir de directions et de seuils JND.

    Modèle :
        u.T @ G @ u ≈ 1 / JND_u²

    Les directions sont normalisées avant estimation.

    Paramétrisation :
        G = L @ L.T + reg_diag * I

    La diagonale de L est paramétrée par exp(.) pour préserver
    la positivité du facteur triangulaire.
    """
    U = _as_finite_array(
        directions,
        "directions",
        ndim=2,
    )

    dim = _positive_integer(
        dim,
        "dim",
    )

    j = _validate_nonnegative(
        jnds,
        "jnds",
    ).reshape(-1)

    reg_diag = float(reg_diag)

    if U.shape[1] != dim:
        raise ValueError(
            "directions doit avoir dim colonnes."
        )

    if U.shape[0] != len(j):
        raise ValueError(
            "Le nombre de directions et de JND doit correspondre."
        )

    if np.any(j <= 0):
        raise ValueError(
            "Tous les JND doivent être strictement positifs."
        )

    if not np.isfinite(reg_diag) or reg_diag < 0:
        raise ValueError(
            "reg_diag doit être fini et non négatif."
        )

    norms = np.linalg.norm(U, axis=1)

    if np.any(norms <= np.finfo(float).eps):
        raise ValueError(
            "Une direction est nulle."
        )

    # C3 : normalisation explicite des directions.
    U = U / norms[:, None]

    y = 1.0 / (j ** 2)

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "Les JND produisent des cibles numériques non finies."
        )

    n_params = dim * (dim + 1) // 2

    iu, ju = np.triu_indices(dim)

    design = (
        U[:, iu]
        * U[:, ju]
        * np.where(iu == ju, 1.0, 2.0)
    )

    if int(np.linalg.matrix_rank(design)) < n_params:
        warnings.warn(
            "Les directions ne permettent pas d'identifier "
            "tous les paramètres de la métrique.",
            RuntimeWarning,
            stacklevel=2,
        )

    def unpack_L(
        params: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        L = np.zeros((dim, dim))
        idx = 0

        for i in range(dim):
            for k in range(i + 1):
                if i == k:
                    L[i, k] = math.exp(params[idx])
                else:
                    L[i, k] = params[idx]

                idx += 1

        return L

    def loss(
        params: NDArray[np.floating],
    ) -> float:
        L = unpack_L(params)
        G = L @ L.T + reg_diag * np.eye(dim)

        preds = np.sum(
            (U @ G) * U,
            axis=1,
        )

        return float(
            np.mean((preds - y) ** 2)
        )

    result = minimize(
        loss,
        x0=np.zeros(n_params),
        method="BFGS",
    )

    if (
        not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
    ):
        raise RuntimeError(
            "Estimation de la métrique échouée : "
            f"{result.message}"
        )

    if not result.success:
        warnings.warn(
            "BFGS n'a pas déclaré de convergence "
            f"({result.message}).",
            RuntimeWarning,
            stacklevel=2,
        )

    L_opt = unpack_L(result.x)

    return _ensure_spd(
        L_opt @ L_opt.T + reg_diag * np.eye(dim),
        eig_floor=_DEFAULT_EIG_FLOOR,
    )


def interpolate_metric_field_log_euclidean(
    x_query: ArrayLike,
    grid_coords: ArrayLike,
    log_G_grid: ArrayLike,
    bandwidth: float = 1.0,
) -> NDArray[np.floating]:
    """
    Interpole un champ tensoriel défini par ses logarithmes matriciels.

    Les poids sont gaussiens et normalisés de manière stable.
    L'interpolation est effectuée dans l'espace des logarithmes,
    puis ramenée dans l'espace SPD par exponentielle matricielle.
    """
    x = _as_finite_array(
        x_query,
        "x_query",
        ndim=1,
    )

    grid = _validate_points(
        grid_coords,
        "grid_coords",
    )

    logs = _as_finite_array(
        log_G_grid,
        "log_G_grid",
        ndim=3,
    )

    bandwidth = _check_bandwidth(bandwidth)

    if x.shape[0] != grid.shape[1]:
        raise ValueError(
            "x_query et grid_coords doivent avoir "
            "la même dimension."
        )

    if logs.shape != (
        grid.shape[0],
        grid.shape[1],
        grid.shape[1],
    ):
        raise ValueError(
            "log_G_grid doit avoir la forme (G, d, d)."
        )

    logs = 0.5 * (
        logs + np.swapaxes(logs, -1, -2)
    )

    return _interpolate_prepared(
        x[None, :],
        grid,
        logs,
        bandwidth,
        _DEFAULT_EIG_FLOOR,
        "log_euclidean",
    )[0]


def compute_integrated_path_distance(
    xA: ArrayLike,
    xB: ArrayLike,
    grid_coords: ArrayLike,
    log_G_grid: ArrayLike,
    bandwidth: float = 1.0,
    n_steps: int = 20,
) -> float:
    """
    Longueur métrique le long du segment xA -> xB.

    L'intégration utilise n_steps sous-segments et une quadrature
    au point milieu.
    """
    a = _as_finite_array(
        xA,
        "xA",
        ndim=1,
    )

    b = _as_finite_array(
        xB,
        "xB",
        ndim=1,
    )

    n_steps = _positive_integer(
        n_steps,
        "n_steps",
    )

    logs = _as_finite_array(
        log_G_grid,
        "log_G_grid",
        ndim=3,
    )

    if a.shape != b.shape:
        raise ValueError(
            "xA et xB doivent avoir la même dimension."
        )

    if logs.shape[1:] != (
        a.shape[0],
        a.shape[0],
    ):
        raise ValueError(
            "log_G_grid a des dimensions incompatibles avec xA et xB."
        )

    G_grid = spd_expm(logs)

    return path_integrated_distance(
        np.vstack([a, b]),
        grid_coords,
        G_grid,
        bandwidth,
        n_substeps=n_steps,
        interpolation="log_euclidean",
    )


def compute_local_volume_element(
    x: ArrayLike,
    grid_coords: ArrayLike,
    log_G_grid: ArrayLike,
    bandwidth: float = 1.0,
) -> float:
    """Calcule l'élément de volume local dV = sqrt(det(G(x)))."""
    G_x = interpolate_metric_field_log_euclidean(
        x,
        grid_coords,
        log_G_grid,
        bandwidth,
    )

    sign, logdet = np.linalg.slogdet(G_x)

    if sign <= 0 or not np.isfinite(logdet):
        raise RuntimeError(
            "Déterminant non positif ou non fini."
        )

    log_volume = 0.5 * logdet

    if log_volume > math.log(np.finfo(float).max):
        raise OverflowError(
            "L'élément de volume dépasse la capacité numérique."
        )

    return float(math.exp(log_volume))


def pava_isotonic_regression(
    y_obs: ArrayLike,
    weights: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """Régression isotone croissante par PAVA, avec poids optionnels."""
    y = _as_finite_array(
        y_obs,
        "y_obs",
        ndim=1,
    )

    if weights is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = _as_finite_array(
            weights,
            "weights",
            ndim=1,
        )

    return _pool_adjacent_violators(
        y,
        w,
    )


def calibrate_scale_pava(
    d_geo: ArrayLike,
    d_subj: ArrayLike,
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Calibre les distances géodésiques sur les distances subjectives.

    Les valeurs d_geo identiques sont regroupées avant PAVA.
    Leur valeur subjective moyenne est calculée, avec un poids égal
    au nombre d'observations regroupées.

    Le résultat est renvoyé dans l'ordre trié des observations originales.
    """
    x = _validate_nonnegative(
        d_geo,
        "d_geo",
    ).reshape(-1)

    y = _validate_nonnegative(
        d_subj,
        "d_subj",
    ).reshape(-1)

    if x.shape != y.shape:
        raise ValueError(
            "d_geo et d_subj doivent avoir la même longueur."
        )

    if len(x) == 0:
        raise ValueError(
            "Vecteurs vides."
        )

    order = np.argsort(
        x,
        kind="stable",
    )

    x_sorted = x[order]
    y_sorted = y[order]

    # C6 : regroupement des ex æquo avant PAVA.
    unique_x, inverse, counts = np.unique(
        x_sorted,
        return_inverse=True,
        return_counts=True,
    )

    sums = np.bincount(
        inverse,
        weights=y_sorted,
    )

    means = sums / counts

    iso = _pool_adjacent_violators(
        means,
        counts.astype(float),
    )

    return (
        x_sorted,
        iso[inverse],
    )


# ============================================================================
# 12. DÉMONSTRATION
# ============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print(
        f"LOCAL JND BRIDGE CALIBRATION  v{__version__}"
    )
    print("=" * 72)

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # 12.1 PROCRUSTES
    # ------------------------------------------------------------------

    reference = rng.normal(
        size=(30, 3)
    )

    angle = 0.65

    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])

    source = (
        1.8 * reference @ rotation.T
        + np.array([2.0, -1.0, 0.5])
    )

    procrustes = procrustes_align_bridges(
        source,
        reference,
        allow_reflection=True,
        return_raw_distances=True,
    )

    print("\n[1] PROCRUSTES")
    print("Scale :", procrustes["scale"])
    print("Disparity :", procrustes["disparity"])

    # ------------------------------------------------------------------
    # 12.2 MÉTRIQUE LOCALE
    # ------------------------------------------------------------------

    displacements = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 0.5],
        [-0.5, 1.0, 1.0],
    ])

    true_metric = np.array([
        [2.0, 0.2, 0.1],
        [0.2, 1.5, 0.15],
        [0.1, 0.15, 1.2],
    ])

    directions = displacements / np.linalg.norm(
        displacements,
        axis=1,
        keepdims=True,
    )

    thresholds = 1.0 / np.sqrt(
        np.einsum(
            "ni,ij,nj->n",
            directions,
            true_metric,
            directions,
        )
    )

    estimated_metric = fit_local_metric_at_anchor(
        displacements,
        thresholds,
        G_prior=np.eye(3),
        shrink_lambda=0.01,
        reg=1e-8,
    )

    print("\n[2] MÉTRIQUE LOCALE")
    print("Métrique de référence :")
    print(true_metric)
    print("Métrique estimée :")
    print(estimated_metric)
    print(
        "Valeurs propres estimées :",
        np.linalg.eigvalsh(estimated_metric),
    )

    # ------------------------------------------------------------------
    # 12.3 INTERPOLATION
    # ------------------------------------------------------------------

    anchor_points = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    anchor_metrics = np.stack([
        np.eye(3),
        true_metric,
        1.5 * np.eye(3),
    ])

    query = np.array([
        0.1,
        0.2,
        0.0,
    ])

    print("\n[3] INTERPOLATION")

    print(
        interpolate_local_metric(
            query,
            anchor_points,
            anchor_metrics,
            bandwidth=1.0,
        )
    )

    # ------------------------------------------------------------------
    # 12.4 DISTANCE INTÉGRÉE
    # ------------------------------------------------------------------

    waypoints = np.array([
        [-0.8, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.8, 0.0, 0.0],
    ])

    print("\n[4] DISTANCE INTÉGRÉE")

    print(
        path_integrated_distance(
            waypoints,
            anchor_points,
            anchor_metrics,
            bandwidth=1.0,
            n_substeps=50,
        )
    )

    # ------------------------------------------------------------------
    # 12.5 VOLUME LOCAL
    # ------------------------------------------------------------------

    region_samples = rng.uniform(
        -1.0,
        1.0,
        size=(200, 3),
    )

    print("\n[5] VOLUME LOCAL")

    print(
        local_volume_in_region(
            region_samples,
            anchor_points,
            anchor_metrics,
            bandwidth=1.0,
            region_base_volume=8.0,
        )
    )

    # ------------------------------------------------------------------
    # 12.6 CALIBRATION
    # ------------------------------------------------------------------

    n_pairs = 200

    raw_distances = rng.uniform(
        0.1,
        10.0,
        size=n_pairs,
    )

    noise = rng.normal(
        0.0,
        0.15,
        size=n_pairs,
    )

    true_distances = np.maximum(
        1.2
        + 1.7 * raw_distances
        + 0.08 * raw_distances ** 2
        + noise,
        0.0,
    )

    calibration_result = calibrate_and_validate_bridges(
        raw_distances,
        true_distances,
        seed=42,
    )

    print("\n[6] CALIBRATION")
    print(
        "Modèle sélectionné :",
        calibration_result["selected_model"],
    )

    print(
        "Validation :",
        calibration_result["validation"],
    )

    print(
        "Test indépendant au niveau des paires :",
        calibration_result["test"],
    )

    print(
        "Diagnostics :",
        calibration_result["diagnostics"],
    )

    # ------------------------------------------------------------------
    # 12.7 API HISTORIQUE
    # ------------------------------------------------------------------

    print("\n[7] API HISTORIQUE")

    fp = fit_frozen_procrustes(
        reference,
        source,
    )

    back = transform_frozen_procrustes(
        reference,
        fp,
    )

    print(
        "Procrustes figé, erreur max :",
        float(np.max(np.abs(back - source))),
    )

    g_est = estimate_local_spd_metric(
        directions,
        thresholds,
        dim=3,
        reg_diag=0.0,
    )

    print(
        "Métrique historique, erreur max :",
        float(np.max(np.abs(g_est - true_metric))),
    )

    logs = np.stack([
        spd_logm(g)
        for g in anchor_metrics
    ])

    print(
        "Distance historique :",
        compute_integrated_path_distance(
            waypoints[0],
            waypoints[-1],
            anchor_points,
            logs,
            1.0,
            50,
        ),
    )

    xs, ys = calibrate_scale_pava(
        raw_distances,
        true_distances,
    )

    print(
        "PAVA monotone :",
        bool(np.all(np.diff(ys) >= -1e-12)),
    )

    # ------------------------------------------------------------------
    # 12.8 TESTS DE VALIDATION DES CORRECTIONS
    # ------------------------------------------------------------------

    print("\n[8] TESTS DES CORRECTIONS NUMÉRIQUES")

    # C1 : floor invalide.
    try:
        spd_logm(np.eye(2), floor=np.nan)
    except ValueError:
        print("C1 : floor non fini rejeté.")

    # C1 : floor nul.
    try:
        spd_logm(np.eye(2), floor=0.0)
    except ValueError:
        print("C1 : floor nul rejeté.")

    # C4 : interpolation log-euclidienne.
    G_log = interpolate_local_metric(
        query,
        anchor_points,
        anchor_metrics,
        bandwidth=1.0,
        interpolation="log_euclidean",
    )

    print(
        "C4 : valeurs propres log-euclidiennes :",
        np.linalg.eigvalsh(G_log),
    )

    # C6 : égalité des valeurs calibrées pour les mêmes distances.
    x_test = np.array([
        1.0, 2.0, 1.0, 3.0, 2.0,
    ])

    y_test = np.array([
        1.0, 4.0, 3.0, 5.0, 2.0,
    ])

    x_cal, y_cal = calibrate_scale_pava(
        x_test,
        y_test,
    )

    for value in np.unique(x_cal):
        associated = y_cal[x_cal == value]

        assert np.allclose(
            associated,
            associated[0],
        )

    print(
        "C6 : les ex æquo de x reçoivent une valeur calibrée identique."
    )

    print("\n" + "=" * 72)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 72)
