#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline complet de calibration et d'estimation des volumes subjectifs.

Étapes :

1. séparation des stimuli-ponts par nœuds ;
2. alignement Procrustes sur les nœuds de calibration ;
3. calibration linéaire ou monotone des distances ;
4. estimation d'un champ métrique local G(x) ;
5. apprentissage de W_theta sur des données distinctes ;
6. estimation volumique par Monte-Carlo uniforme dans un ellipsoïde ;
7. comparaison avec une densité globale équivalente ;
8. test de permutation apparié ou stratifié ;
9. export atomique JSON et CSV.

Les données-ponts ne sont jamais utilisées pour ajuster W_theta.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform

from local_jnd_bridge_calibration import (
    apply_linear_scale,
    apply_monotone_binned,
    evaluate_calibration_split,
    fit_linear_scale,
    fit_local_metric_at_anchor,
    fit_monotone_binned,
    interpolate_local_metric,
    path_integrated_distance,
    procrustes_align_bridges,
)

from volume_subjectif_formats import (
    ball_volume_factor,
    fit_from_pairwise_dissimilarity,
    similarity_to_dissimilarity,
    upper_tri_pairs,
)


def compute_unified_scaler(
    X_train: np.ndarray,
    ddof: int = 1,
):
    """
    Estime un scaler unique sur les seules données d'entraînement.
    """
    X = np.asarray(X_train, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X_train doit être une matrice 2D."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X_train contient des valeurs non finies."
        )

    mu = np.mean(
        X,
        axis=0,
    )

    sd = np.maximum(
        np.std(
            X,
            axis=0,
            ddof=int(ddof),
        ),
        1e-8,
    )

    return mu, sd


def apply_unified_scaler(
    X: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """
    Applique un scaler appris antérieurement.
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X doit être une matrice 2D."
        )

    if mu.ndim != 1 or sd.ndim != 1:
        raise ValueError(
            "mu et sd doivent être des vecteurs 1D."
        )

    if X.shape[1] != len(mu) or len(mu) != len(sd):
        raise ValueError(
            "Dimensions incompatibles entre X, mu et sd."
        )

    if np.any(sd <= 0.0):
        raise ValueError(
            "Les écarts-types doivent être strictement positifs."
        )

    return (X - mu[None, :]) / sd[None, :]


def project_states(
    X: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    L_hat: np.ndarray,
    center_states: bool = True,
) -> np.ndarray:
    """
    Projette les états dans l'espace latent.

    Si center_states=True, le centre propre au jeu d'états fourni est retiré.
    Le volume obtenu mesure alors principalement la dispersion interne.
    """
    X_std = apply_unified_scaler(
        X,
        mu,
        sd,
    )

    if center_states:
        X_std = (
            X_std
            - np.mean(
                X_std,
                axis=0,
                keepdims=True,
            )
        )

    L_hat = np.asarray(
        L_hat,
        dtype=float,
    )

    if L_hat.ndim != 2:
        raise ValueError(
            "L_hat doit être une matrice 2D."
        )

    if X_std.shape[1] != L_hat.shape[0]:
        raise ValueError(
            "Dimensions incompatibles entre X et L_hat."
        )

    return X_std @ L_hat


def apply_procrustes_transform(
    new_coords: np.ndarray,
    procrustes_fit: Dict[str, object],
) -> np.ndarray:
    """
    Applique une transformation Procrustes déjà estimée.
    """
    pts = np.asarray(
        new_coords,
        dtype=float,
    )

    if pts.ndim != 2:
        raise ValueError(
            "new_coords doit être une matrice 2D."
        )

    mu_src = np.asarray(
        procrustes_fit["translation_source"],
        dtype=float,
    )

    mu_tgt = np.asarray(
        procrustes_fit["translation_target"],
        dtype=float,
    )

    rotation = np.asarray(
        procrustes_fit["rotation_matrix"],
        dtype=float,
    )

    scale = float(
        procrustes_fit["scale_factor"]
    )

    if pts.shape[1] != len(mu_src):
        raise ValueError(
            "La dimension de new_coords ne correspond pas "
            "à la transformation Procrustes."
        )

    d_rotation = rotation.shape[0]

    pts_pad = np.zeros(
        (len(pts), d_rotation),
        dtype=float,
    )

    pts_pad[:, :pts.shape[1]] = (
        pts - mu_src[None, :]
    )

    aligned = scale * (
        pts_pad @ rotation
    )

    aligned[:, :len(mu_tgt)] += mu_tgt[None, :]

    return aligned[:, :len(mu_tgt)]


def _apply_scalar_scale_about_center(
    coords: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Applique une échelle scalaire autour d'un centre.
    """
    coords = np.asarray(
        coords,
        dtype=float,
    )

    center = np.asarray(
        center,
        dtype=float,
    )

    if scale <= 0.0:
        raise ValueError(
            "Le facteur d'échelle doit être positif."
        )

    return center[None, :] + scale * (
        coords - center[None, :]
    )


def make_uniform_unit_ball_points(
    n_samples: int,
    dimension: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Génère des points uniformes dans la boule unité de dimension k.

    Les points peuvent être réutilisés dans toutes les permutations afin
    d'éliminer le bruit Monte-Carlo différentiel.
    """
    n_samples = int(n_samples)
    dimension = int(dimension)

    if n_samples < 1:
        raise ValueError(
            "n_samples doit être positif."
        )

    if dimension < 1:
        raise ValueError(
            "dimension doit être positive."
        )

    rng = np.random.default_rng(
        int(seed)
    )

    directions = rng.standard_normal(
        (n_samples, dimension)
    )

    norms = np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    directions = directions / np.maximum(
        norms,
        1e-12,
    )

    radii = rng.uniform(
        0.0,
        1.0,
        size=(n_samples, 1),
    ) ** (1.0 / dimension)

    return directions * radii


def sample_uniform_in_ellipsoid(
    center: np.ndarray,
    cov_matrix: np.ndarray,
    unit_ball_points: np.ndarray,
    eig_floor: float = 1e-10,
):
    """
    Transforme des points uniformes de la boule unité en points uniformes
    dans l'ellipsoïde défini par cov_matrix.

    Retourne :
        samples
        volume euclidien de l'ellipsoïde
        valeurs propres
        rang numérique
    """
    center = np.asarray(
        center,
        dtype=float,
    )

    cov = np.asarray(
        cov_matrix,
        dtype=float,
    )

    unit_points = np.asarray(
        unit_ball_points,
        dtype=float,
    )

    if center.ndim != 1:
        raise ValueError(
            "center doit être un vecteur 1D."
        )

    dimension = len(center)

    if cov.shape != (dimension, dimension):
        raise ValueError(
            "cov_matrix a une dimension incompatible."
        )

    if unit_points.ndim != 2:
        raise ValueError(
            "unit_ball_points doit être une matrice 2D."
        )

    if unit_points.shape[1] != dimension:
        raise ValueError(
            "unit_ball_points a une dimension incompatible."
        )

    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(
        cov
    )

    eigvals = np.maximum(
        eigvals,
        float(eig_floor),
    )

    radii = np.sqrt(
        eigvals
    )

    samples = center[None, :] + (
        unit_points * radii[None, :]
    ) @ eigvecs.T

    volume = (
        ball_volume_factor(dimension)
        * float(np.prod(radii))
    )

    rank_threshold = max(
        float(np.max(eigvals)) * 1e-8,
        float(eig_floor),
    )

    effective_rank = int(
        np.sum(eigvals > rank_threshold)
    )

    return (
        samples,
        float(volume),
        eigvals,
        effective_rank,
    )


def compute_calibrated_subjective_volume(
    X_condition: np.ndarray,
    L_hat: np.ndarray,
    mu_train: np.ndarray,
    sd_train: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float = 1.5,
    n_mc_samples: int = 400,
    eig_floor: float = 1e-10,
    seed: int = 42,
    unit_ball_points: Optional[np.ndarray] = None,
):
    """
    Estime :

        V_psi(R)
        =
        V_euclidien(R)
        *
        moyenne_R[sqrt(det(G(z)))]

    L'échantillonnage est uniforme dans l'ellipsoïde estimé par la
    covariance des états projetés.
    """
    Z = project_states(
        X_condition,
        mu_train,
        sd_train,
        L_hat,
        center_states=True,
    )

    n_states, dimension = Z.shape

    if n_states < 2:
        raise ValueError(
            "Au moins deux états sont nécessaires."
        )

    center = np.mean(
        Z,
        axis=0,
    )

    centered = Z - center[None, :]

    covariance = (
        centered.T @ centered
    ) / max(
        n_states - 1,
        1,
    )

    covariance = 0.5 * (
        covariance + covariance.T
    )

    if unit_ball_points is None:
        unit_ball_points = make_uniform_unit_ball_points(
            n_samples=n_mc_samples,
            dimension=dimension,
            seed=seed,
        )
    else:
        unit_ball_points = np.asarray(
            unit_ball_points,
            dtype=float,
        )

        if unit_ball_points.shape[1] != dimension:
            raise ValueError(
                "unit_ball_points a une dimension incompatible."
            )

        n_mc_samples = len(unit_ball_points)

    samples, raw_volume, eigvals, effective_rank = (
        sample_uniform_in_ellipsoid(
            center=center,
            cov_matrix=covariance,
            unit_ball_points=unit_ball_points,
            eig_floor=eig_floor,
        )
    )

    densities = np.empty(
        len(samples),
        dtype=float,
    )

    for idx, point in enumerate(samples):
        G_point = interpolate_local_metric(
            point,
            anchor_points,
            anchor_metrics,
            bandwidth=bandwidth,
        )

        sign_det, log_det = np.linalg.slogdet(
            G_point
        )

        if sign_det <= 0.0 or not np.isfinite(log_det):
            raise RuntimeError(
                "La métrique locale n'est pas SPD."
            )

        densities[idx] = math.exp(
            0.5 * log_det
        )

    mean_density = float(
        np.mean(densities)
    )

    mc_se_density = float(
        np.std(
            densities,
            ddof=1,
        ) / math.sqrt(
            max(len(densities), 1)
        )
    ) if len(densities) > 1 else 0.0

    calibrated_volume = (
        raw_volume
        * mean_density
    )

    eigenvalues_positive = np.maximum(
        eigvals,
        0.0,
    )

    denominator = float(
        np.sum(eigenvalues_positive ** 2)
    )

    if denominator > 1e-15:
        effective_dimension = float(
            np.sum(eigenvalues_positive) ** 2
            / denominator
        )
    else:
        effective_dimension = 0.0

    return {
        "V_k_raw_space": float(raw_volume),
        "mean_uniform_density_sqrt_det_G": mean_density,
        "mc_se_density": mc_se_density,
        "V_k_calibrated_U_psi_k": float(
            calibrated_volume
        ),
        "effective_dim_ouverture": effective_dimension,
        "effective_rank": int(effective_rank),
        "eigenvalues_raw": [
            float(value)
            for value in eigenvalues_positive
        ],
        "n_mc_samples": int(n_mc_samples),
    }


def compute_reference_global_density(
    reference_points: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float = 1.5,
    k_rank: int = 2,
):
    """
    Estime la densité globale de référence :

        g_bar = moyenne[sqrt(det(G(x)))]

    et l'unité isotrope équivalente :

        u_vol = g_bar^(-1/k)
    """
    points = np.asarray(
        reference_points,
        dtype=float,
    )

    if points.ndim != 2 or len(points) == 0:
        raise ValueError(
            "reference_points doit être une matrice non vide."
        )

    densities = []

    for point in points:
        G_point = interpolate_local_metric(
            point,
            anchor_points,
            anchor_metrics,
            bandwidth=bandwidth,
        )

        sign_det, log_det = np.linalg.slogdet(
            G_point
        )

        if sign_det <= 0.0 or not np.isfinite(log_det):
            raise RuntimeError(
                "Métrique locale non SPD dans la référence globale."
            )

        densities.append(
            math.exp(0.5 * log_det)
        )

    g_bar = max(
        float(np.mean(densities)),
        1e-12,
    )

    k_rank = max(
        int(k_rank),
        1,
    )

    u_vol = g_bar ** (
        -1.0 / k_rank
    )

    return {
        "global_density_g_bar": g_bar,
        "equivalent_isotropic_u_vol": float(
            u_vol
        ),
        "n_reference_points": int(
            len(points)
        ),
    }


def compare_global_vs_local_volume(
    X_condition: np.ndarray,
    L_hat: np.ndarray,
    mu_train: np.ndarray,
    sd_train: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    reference_density_info: Dict[str, object],
    bandwidth: float = 1.5,
    n_mc_samples: int = 400,
    eig_floor: float = 1e-10,
    seed: int = 42,
    unit_ball_points: Optional[np.ndarray] = None,
):
    """
    Compare :

    - l'intégrale avec G(x) locale ;
    - le volume euclidien multiplié par une densité globale équivalente.
    """
    local_result = compute_calibrated_subjective_volume(
        X_condition=X_condition,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        n_mc_samples=n_mc_samples,
        eig_floor=eig_floor,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    global_density = float(
        reference_density_info[
            "global_density_g_bar"
        ]
    )

    local_volume = float(
        local_result[
            "V_k_calibrated_U_psi_k"
        ]
    )

    global_volume = (
        float(local_result["V_k_raw_space"])
        * global_density
    )

    ratio = local_volume / max(
        global_volume,
        1e-12,
    )

    return {
        "volume_local_G_x": local_volume,
        "volume_global_equiv_density": float(
            global_volume
        ),
        "global_density_g_bar": global_density,
        "equivalent_isotropic_u_vol": float(
            reference_density_info[
                "equivalent_isotropic_u_vol"
            ]
        ),
        "diff_local_minus_global": float(
            local_volume - global_volume
        ),
        "ratio_local_over_global": float(
            ratio
        ),
        "mean_uniform_density_sqrt_det_G": float(
            local_result[
                "mean_uniform_density_sqrt_det_G"
            ]
        ),
        "effective_dim_ouverture": float(
            local_result[
                "effective_dim_ouverture"
            ]
        ),
    }


def _pairs_strictly_within_nodes(
    node_indices: Sequence[int],
):
    """
    Retourne toutes les paires internes à un ensemble de nœuds.
    """
    indices = sorted(
        set(
            int(index)
            for index in node_indices
        )
    )

    pairs = []

    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            pairs.append(
                (
                    indices[left],
                    indices[right],
                )
            )

    return pairs


def calibrate_bridges_by_node_split(
    spatial_coords_bridges: np.ndarray,
    ref_coords_bridges: np.ndarray,
    cal_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    allow_reflection: bool = False,
):
    """
    Calibration par séparation des nœuds-ponts.

    L'alignement Procrustes est ajusté uniquement sur les nœuds de calibration.
    L'échelle des coordonnées utilisées ensuite est estimée par le modèle linéaire
    sur les paires de calibration.

    Une transformation monotone est évaluée comme modèle de distance, mais elle
    ne peut pas être appliquée directement à des coordonnées euclidiennes sans
    reconstruction géométrique supplémentaire.
    """
    source = np.asarray(
        spatial_coords_bridges,
        dtype=float,
    )

    reference = np.asarray(
        ref_coords_bridges,
        dtype=float,
    )

    if source.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            "Les coordonnées doivent être des matrices 2D."
        )

    if source.shape[0] != reference.shape[0]:
        raise ValueError(
            "Les deux ensembles doivent contenir le même nombre de ponts."
        )

    cal_ids = sorted(
        set(int(index) for index in cal_ids)
    )

    val_ids = sorted(
        set(int(index) for index in val_ids)
    )

    test_ids = sorted(
        set(int(index) for index in test_ids)
    )

    all_ids = set(
        range(source.shape[0])
    )

    if (
        set(cal_ids)
        & set(val_ids)
        or set(cal_ids)
        & set(test_ids)
        or set(val_ids)
        & set(test_ids)
    ):
        raise ValueError(
            "Les ensembles calibration, validation et test doivent être disjoints."
        )

    if set(cal_ids) | set(val_ids) | set(test_ids) != all_ids:
        raise ValueError(
            "Tous les nœuds-ponts doivent appartenir à un seul ensemble."
        )

    if len(cal_ids) < 3:
        raise ValueError(
            "Au moins trois nœuds sont nécessaires pour l'alignement."
        )

    procrustes_fit = procrustes_align_bridges(
        coords_source=source[cal_ids],
        coords_target=reference[cal_ids],
        allow_reflection=allow_reflection,
        estimate_scale=False,
        return_raw_distances=False,
    )

    aligned_without_scale = apply_procrustes_transform(
        source,
        procrustes_fit,
    )

    target_center = np.mean(
        reference[cal_ids],
        axis=0,
    )

    def extract_distances(
        node_ids: Sequence[int],
    ):
        pairs = _pairs_strictly_within_nodes(
            node_ids
        )

        if len(pairs) < 2:
            raise ValueError(
                "Pas assez de paires internes dans un sous-ensemble."
            )

        raw_values = []
        ref_values = []

        for left, right in pairs:
            raw_values.append(
                np.linalg.norm(
                    aligned_without_scale[left]
                    - aligned_without_scale[right]
                )
            )

            ref_values.append(
                np.linalg.norm(
                    reference[left]
                    - reference[right]
                )
            )

        return (
            np.asarray(raw_values, dtype=float),
            np.asarray(ref_values, dtype=float),
        )

    d_cal_raw, d_cal_ref = extract_distances(
        cal_ids
    )

    d_val_raw, d_val_ref = extract_distances(
        val_ids
    )

    d_test_raw, d_test_ref = extract_distances(
        test_ids
    )

    linear_model = fit_linear_scale(
        d_cal_raw,
        d_cal_ref,
        fix_intercept_zero=True,
    )

    monotone_model = fit_monotone_binned(
        d_cal_raw,
        d_cal_ref,
        n_bins=min(
            6,
            max(
                2,
                len(d_cal_raw) // 4,
            ),
        ),
    )

    pred_val_linear = apply_linear_scale(
        d_val_raw,
        linear_model,
    )

    pred_val_monotone = apply_monotone_binned(
        d_val_raw,
        monotone_model,
    )

    val_linear_metrics = evaluate_calibration_split(
        d_val_raw,
        d_val_ref,
        pred_val_linear,
    )

    val_monotone_metrics = evaluate_calibration_split(
        d_val_raw,
        d_val_ref,
        pred_val_monotone,
    )

    if (
        val_monotone_metrics["rmse"]
        < val_linear_metrics["rmse"]
    ):
        selected_model = "monotone_piecewise"
        selected_distance_model = monotone_model
        selected_val_metrics = val_monotone_metrics
    else:
        selected_model = "linear_zero_intercept"
        selected_distance_model = linear_model
        selected_val_metrics = val_linear_metrics

    pred_test_linear = apply_linear_scale(
        d_test_raw,
        linear_model,
    )

    pred_test_monotone = apply_monotone_binned(
        d_test_raw,
        monotone_model,
    )

    test_linear_metrics = evaluate_calibration_split(
        d_test_raw,
        d_test_ref,
        pred_test_linear,
    )

    test_monotone_metrics = evaluate_calibration_split(
        d_test_raw,
        d_test_ref,
        pred_test_monotone,
    )

    if (
        selected_model
        == "monotone_piecewise"
    ):
        selected_test_metrics = test_monotone_metrics
    else:
        selected_test_metrics = test_linear_metrics

    coordinate_scale = float(
        linear_model["b"]
    )

    aligned_all = _apply_scalar_scale_about_center(
        aligned_without_scale,
        target_center,
        coordinate_scale,
    )

    non_equivalence_warning = bool(
        abs(
            selected_test_metrics[
                "slope_pred_vs_ref"
            ] - 1.0
        ) > 0.25
        or selected_test_metrics[
            "spearman"
        ] < 0.50
    )

    return {
        "procrustes_fit": procrustes_fit,
        "aligned_without_scale": aligned_without_scale,
        "aligned_all_bridges": aligned_all,
        "coordinate_scale_model": "linear_zero_intercept",
        "coordinate_scale": coordinate_scale,
        "selected_model": selected_model,
        "selected_distance_model": selected_distance_model,
        "selected_validation_metrics": selected_val_metrics,
        "val_linear_metrics": val_linear_metrics,
        "val_monotone_metrics": val_monotone_metrics,
        "test_linear_metrics": test_linear_metrics,
        "test_monotone_metrics": test_monotone_metrics,
        "test_metrics": selected_test_metrics,
        "monotone_coordinate_transform_applied": False,
        "non_equivalence_warning": non_equivalence_warning,
        "indices": {
            "calibration": cal_ids,
            "validation": val_ids,
            "test": test_ids,
        },
    }


def _validate_pair_ids(
    pair_ids: Sequence,
    n_total: int,
    base_labels: np.ndarray,
):
    """
    Vérifie qu'une paire contient exactement une observation A et une B.
    """
    pair_array = np.asarray(
        pair_ids
    )

    if len(pair_array) != n_total:
        raise ValueError(
            "pair_ids doit avoir n_A + n_B éléments."
        )

    groups = []

    for pair_value in np.unique(pair_array):
        indices = np.where(
            pair_array == pair_value
        )[0]

        labels = base_labels[indices]

        if (
            len(indices) != 2
            or np.sum(labels == 0) != 1
            or np.sum(labels == 1) != 1
        ):
            raise ValueError(
                "Chaque paire doit contenir exactement "
                "une observation A et une observation B."
            )

        groups.append(indices)

    return groups


def _compute_condition_volume_from_labels(
    pooled: np.ndarray,
    labels: np.ndarray,
    L_hat: np.ndarray,
    mu_train: np.ndarray,
    sd_train: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float,
    unit_ball_points: np.ndarray,
    eig_floor: float,
    seed: int,
):
    """
    Calcule les volumes A et B à partir d'étiquettes permutées.
    """
    group_a = pooled[labels == 0]
    group_b = pooled[labels == 1]

    if len(group_a) < 2 or len(group_b) < 2:
        return None

    result_a = compute_calibrated_subjective_volume(
        X_condition=group_a,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        eig_floor=eig_floor,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    result_b = compute_calibrated_subjective_volume(
        X_condition=group_b,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        eig_floor=eig_floor,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    delta_volume = (
        result_b["V_k_calibrated_U_psi_k"]
        - result_a["V_k_calibrated_U_psi_k"]
    )

    delta_deff = (
        result_b["effective_dim_ouverture"]
        - result_a["effective_dim_ouverture"]
    )

    return {
        "delta_volume": float(delta_volume),
        "delta_deff": float(delta_deff),
    }


def permutation_test_conditions(
    X_cond_A: np.ndarray,
    X_cond_B: np.ndarray,
    L_hat: np.ndarray,
    mu_train: np.ndarray,
    sd_train: np.ndarray,
    anchor_points: np.ndarray,
    anchor_metrics: np.ndarray,
    bandwidth: float = 1.5,
    n_mc_samples: int = 400,
    eig_floor: float = 1e-10,
    n_perm: int = 499,
    pair_ids: Optional[Sequence] = None,
    strata: Optional[Sequence] = None,
    seed: int = 42,
):
    """
    Test de permutation bilatéral sur :

    - delta_volume = V_B - V_A ;
    - delta_deff = d_eff_B - d_eff_A ;
    - statistique conjointe T_max.

    Si pair_ids est fourni, les étiquettes sont échangées à l'intérieur
    de chaque paire.

    Sinon, si strata est fourni, les étiquettes sont permutées à l'intérieur
    de chaque strate.

    Les mêmes points Monte-Carlo de la boule unité sont réutilisés pour
    toutes les permutations.
    """
    A = np.asarray(
        X_cond_A,
        dtype=float,
    )

    B = np.asarray(
        X_cond_B,
        dtype=float,
    )

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            "Les deux conditions doivent être des matrices 2D."
        )

    if A.shape[1] != B.shape[1]:
        raise ValueError(
            "Les deux conditions doivent avoir le même nombre de variables."
        )

    n_a = len(A)
    n_b = len(B)

    if n_a < 2 or n_b < 2:
        raise ValueError(
            "Chaque condition doit contenir au moins deux états."
        )

    if n_perm < 1:
        raise ValueError(
            "n_perm doit être positif."
        )

    dimension_latente = np.asarray(
        L_hat
    ).shape[1]

    unit_ball_points = make_uniform_unit_ball_points(
        n_samples=n_mc_samples,
        dimension=dimension_latente,
        seed=seed,
    )

    observed_a = compute_calibrated_subjective_volume(
        X_condition=A,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        eig_floor=eig_floor,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    observed_b = compute_calibrated_subjective_volume(
        X_condition=B,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        eig_floor=eig_floor,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    observed_delta_volume = (
        observed_b["V_k_calibrated_U_psi_k"]
        - observed_a["V_k_calibrated_U_psi_k"]
    )

    observed_delta_deff = (
        observed_b["effective_dim_ouverture"]
        - observed_a["effective_dim_ouverture"]
    )

    pooled = np.vstack(
        [A, B]
    )

    n_total = n_a + n_b

    base_labels = np.concatenate(
        [
            np.zeros(n_a, dtype=int),
            np.ones(n_b, dtype=int),
        ]
    )

    pair_groups = None

    if pair_ids is not None:
        pair_groups = _validate_pair_ids(
            pair_ids,
            n_total,
            base_labels,
        )

    strata_array = None

    if strata is not None:
        strata_array = np.asarray(
            strata
        )

        if len(strata_array) != n_total:
            raise ValueError(
                "strata doit avoir n_A + n_B éléments."
            )

    rng = np.random.default_rng(
        seed
    )

    null_delta_volume = []
    null_delta_deff = []

    for permutation_index in range(
        int(n_perm)
    ):
        if pair_groups is not None:
            permuted_labels = base_labels.copy()

            for indices in pair_groups:
                if rng.integers(0, 2) == 1:
                    first, second = indices

                    (
                        permuted_labels[first],
                        permuted_labels[second],
                    ) = (
                        permuted_labels[second],
                        permuted_labels[first],
                    )

            design = "paired_within_pair_swap"

        elif strata_array is not None:
            permuted_labels = base_labels.copy()

            for stratum in np.unique(
                strata_array
            ):
                indices = np.where(
                    strata_array == stratum
                )[0]

                permuted_labels[indices] = rng.permutation(
                    base_labels[indices]
                )

            design = "stratified"

        else:
            permuted_labels = rng.permutation(
                base_labels
            )

            design = "unpaired"

        null_result = _compute_condition_volume_from_labels(
            pooled=pooled,
            labels=permuted_labels,
            L_hat=L_hat,
            mu_train=mu_train,
            sd_train=sd_train,
            anchor_points=anchor_points,
            anchor_metrics=anchor_metrics,
            bandwidth=bandwidth,
            unit_ball_points=unit_ball_points,
            eig_floor=eig_floor,
            seed=seed,
        )

        if null_result is None:
            continue

        null_delta_volume.append(
            null_result["delta_volume"]
        )

        null_delta_deff.append(
            null_result["delta_deff"]
        )

    null_delta_volume = np.asarray(
        null_delta_volume,
        dtype=float,
    )

    null_delta_deff = np.asarray(
        null_delta_deff,
        dtype=float,
    )

    n_valid = len(
        null_delta_volume
    )

    if n_valid == 0:
        raise RuntimeError(
            "Aucune permutation valide n'a été produite."
        )

    sd_volume = float(
        np.std(
            null_delta_volume,
            ddof=1,
        )
    ) if n_valid > 1 else 1.0

    sd_deff = float(
        np.std(
            null_delta_deff,
            ddof=1,
        )
    ) if n_valid > 1 else 1.0

    sd_volume = max(
        sd_volume,
        1e-12,
    )

    sd_deff = max(
        sd_deff,
        1e-12,
    )

    observed_t_volume = abs(
        observed_delta_volume
    ) / sd_volume

    observed_t_deff = abs(
        observed_delta_deff
    ) / sd_deff

    observed_t_max = max(
        observed_t_volume,
        observed_t_deff,
    )

    null_t_max = np.maximum(
        np.abs(null_delta_volume) / sd_volume,
        np.abs(null_delta_deff) / sd_deff,
    )

    p_volume = (
        1.0
        + np.sum(
            np.abs(null_delta_volume)
            >= abs(observed_delta_volume)
        )
    ) / (
        n_valid + 1.0
    )

    p_deff = (
        1.0
        + np.sum(
            np.abs(null_delta_deff)
            >= abs(observed_delta_deff)
        )
    ) / (
        n_valid + 1.0
    )

    p_tmax = (
        1.0
        + np.sum(
            null_t_max >= observed_t_max
        )
    ) / (
        n_valid + 1.0
    )

    return {
        "design": design,
        "n_permutations_requested": int(n_perm),
        "n_permutations_valid": int(n_valid),
        "delta_volume_B_minus_A": float(
            observed_delta_volume
        ),
        "p_value_delta_volume_two_sided": float(
            p_volume
        ),
        "null_volume_q05_q95": [
            float(
                np.quantile(
                    null_delta_volume,
                    0.05,
                )
            ),
            float(
                np.quantile(
                    null_delta_volume,
                    0.95,
                )
            ),
        ],
        "delta_ouverture_deff_B_minus_A": float(
            observed_delta_deff
        ),
        "p_value_delta_ouverture_two_sided": float(
            p_deff
        ),
        "null_ouverture_q05_q95": [
            float(
                np.quantile(
                    null_delta_deff,
                    0.05,
                )
            ),
            float(
                np.quantile(
                    null_delta_deff,
                    0.95,
                )
            ),
        ],
        "joint_t_max_stat": float(
            observed_t_max
        ),
        "p_value_joint_t_max": float(
            p_tmax
        ),
        "sd_null_volume": float(
            sd_volume
        ),
        "sd_null_deff": float(
            sd_deff
        ),
    }


def _compute_file_sha256(
    filepath: str,
) -> str:
    """
    Calcule le hash SHA-256 d'un fichier.
    """
    digest = hashlib.sha256()

    with open(
        filepath,
        "rb",
    ) as file_handle:
        for chunk in iter(
            lambda: file_handle.read(65536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def export_pipeline_results(
    summary: Dict[str, object],
    filepath: str,
    fmt: Optional[str] = None,
) -> str:
    """
    Exporte le résultat en JSON ou CSV par écriture atomique.

    L'opération utilise :

    1. fichier temporaire dans le même répertoire ;
    2. flush et fsync du fichier ;
    3. os.replace ;
    4. fsync du répertoire parent.
    """
    target_path = os.path.abspath(
        filepath
    )

    parent_dir = os.path.dirname(
        target_path
    ) or "."

    os.makedirs(
        parent_dir,
        exist_ok=True,
    )

    resolved_format = (
        fmt
        or os.path.splitext(
            target_path
        )[1].lstrip(".")
    ).lower()

    if resolved_format not in {
        "json",
        "csv",
    }:
        raise ValueError(
            "Le format doit être 'json' ou 'csv'."
        )

    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_volume_subjectif_",
        suffix=f".{resolved_format}",
        dir=parent_dir,
    )

    try:
        if resolved_format == "json":
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
            ) as file_handle:
                json.dump(
                    summary,
                    file_handle,
                    indent=2,
                    ensure_ascii=False,
                )

                file_handle.flush()
                os.fsync(
                    file_handle.fileno()
                )

        else:
            metadata = summary.get(
                "metadata",
                {},
            )

            bridge = summary.get(
                "bridge_calibration",
                {},
            )

            bridge_test = bridge.get(
                "test_metrics",
                {},
            )

            permutation = summary.get(
                "permutation_test_B_vs_A",
                {},
            )

            comparison = summary.get(
                "global_vs_local_comparison",
                {},
            )

            rows = []

            for condition_key, condition_label in [
                ("condition_A", "A"),
                ("condition_B", "B"),
            ]:
                condition_data = summary.get(
                    condition_key,
                    {},
                )

                comparison_data = comparison.get(
                    condition_key,
                    {},
                )

                rows.append(
                    {
                        "condition": condition_label,
                        "seed": metadata.get("seed"),
                        "n_perm": metadata.get("n_perm"),
                        "k_rank": metadata.get("k_rank"),
                        "bandwidth": metadata.get("bandwidth"),
                        "V_k_raw_space": condition_data.get(
                            "V_k_raw_space"
                        ),
                        "mean_uniform_density_sqrt_det_G": condition_data.get(
                            "mean_uniform_density_sqrt_det_G"
                        ),
                        "V_k_calibrated_U_psi_k": condition_data.get(
                            "V_k_calibrated_U_psi_k"
                        ),
                        "effective_rank": condition_data.get(
                            "effective_rank"
                        ),
                        "effective_dim_ouverture": condition_data.get(
                            "effective_dim_ouverture"
                        ),
                        "volume_global_equiv_density": comparison_data.get(
                            "volume_global_equiv_density"
                        ),
                        "ratio_local_over_global": comparison_data.get(
                            "ratio_local_over_global"
                        ),
                        "straight_path_distance_A_to_B": summary.get(
                            "straight_path_distance_A_to_B"
                        ),
                        "procrustes_disparity": bridge.get(
                            "procrustes_disparity"
                        ),
                        "coordinate_scale": bridge.get(
                            "coordinate_scale"
                        ),
                        "bridge_test_spearman": bridge_test.get(
                            "spearman"
                        ),
                        "bridge_non_equivalence_warning": bridge.get(
                            "non_equivalence_warning"
                        ),
                        "delta_volume_B_minus_A": permutation.get(
                            "delta_volume_B_minus_A"
                        ),
                        "p_value_delta_volume": permutation.get(
                            "p_value_delta_volume_two_sided"
                        ),
                        "delta_ouverture_B_minus_A": permutation.get(
                            "delta_ouverture_deff_B_minus_A"
                        ),
                        "p_value_delta_ouverture": permutation.get(
                            "p_value_delta_ouverture_two_sided"
                        ),
                        "p_value_joint_t_max": permutation.get(
                            "p_value_joint_t_max"
                        ),
                    }
                )

            fieldnames = list(
                rows[0].keys()
            )

            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as file_handle:
                writer = csv.DictWriter(
                    file_handle,
                    fieldnames=fieldnames,
                )

                writer.writeheader()
                writer.writerows(rows)

                file_handle.flush()
                os.fsync(
                    file_handle.fileno()
                )

        os.replace(
            temporary_path,
            target_path,
        )

        if hasattr(
            os,
            "O_DIRECTORY",
        ):
            directory_descriptor = os.open(
                parent_dir,
                os.O_RDONLY
                | os.O_DIRECTORY,
            )

            try:
                os.fsync(
                    directory_descriptor
                )
            finally:
                os.close(
                    directory_descriptor
                )

    except Exception:
        if os.path.exists(
            temporary_path
        ):
            os.remove(
                temporary_path
            )

        raise

    return target_path


def _format_volume_result(
    result: Dict[str, object],
) -> Dict[str, object]:
    """
    Formate légèrement un résultat pour l'impression.
    """
    formatted = {}

    for key, value in result.items():
        if isinstance(value, float):
            formatted[key] = round(
                value,
                6,
            )
        elif isinstance(value, list):
            formatted[key] = [
                round(item, 6)
                if isinstance(item, float)
                else item
                for item in value
            ]
        else:
            formatted[key] = value

    return formatted


def run_full_demonstration(
    seed: int = 2026,
    n_perm: int = 499,
    k_rank: int = 2,
    bandwidth: float = 1.5,
    scaler_ddof: int = 1,
    n_mc_samples: int = 400,
    export_json_path: Optional[str] = None,
    export_csv_path: Optional[str] = None,
):
    """
    Exécute une démonstration complète sur données simulées.
    """
    rng = np.random.default_rng(
        int(seed)
    )

    p_dimensions = 7

    L_true = np.zeros(
        (p_dimensions, k_rank),
        dtype=float,
    )

    if k_rank >= 2:
        L_true[0, 0] = 1.4
        L_true[2, 0] = -1.0
        L_true[4, 1] = 1.3
        L_true[6, 1] = 0.9
    else:
        L_true[0, 0] = 1.4
        L_true[2, 0] = -1.0

    # ------------------------------------------------------------------
    # 1. Stimuli-ponts indépendants
    # ------------------------------------------------------------------
    n_bridges = 15

    X_bridges = rng.standard_normal(
        (n_bridges, p_dimensions)
    )

    Z_bridges_reference = (
        X_bridges @ L_true
    )

    bridge_permutation = rng.permutation(
        n_bridges
    )

    cal_ids = sorted(
        bridge_permutation[:9].tolist()
    )

    val_ids = sorted(
        bridge_permutation[9:12].tolist()
    )

    test_ids = sorted(
        bridge_permutation[12:15].tolist()
    )

    angle = 0.72

    rotation_canvas = np.array(
        [
            [
                math.cos(angle),
                -math.sin(angle),
            ],
            [
                math.sin(angle),
                math.cos(angle),
            ],
        ],
        dtype=float,
    )

    spatial_canvas_bridges = (
        2.1
        * (Z_bridges_reference @ rotation_canvas.T)
        + np.array([3.5, -1.8])
        + 0.08
        * rng.standard_normal(
            Z_bridges_reference.shape
        )
    )

    bridge_calibration = calibrate_bridges_by_node_split(
        spatial_coords_bridges=spatial_canvas_bridges,
        ref_coords_bridges=Z_bridges_reference,
        cal_ids=cal_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        allow_reflection=False,
    )

    aligned_bridges = bridge_calibration[
        "aligned_all_bridges"
    ]

    # ------------------------------------------------------------------
    # 2. Champ métrique local G(x)
    # ------------------------------------------------------------------
    anchor_indices = cal_ids[:min(4, len(cal_ids))]

    anchor_points = aligned_bridges[
        anchor_indices
    ]

    G_prior = np.eye(
        k_rank,
        dtype=float,
    )

    anchor_metrics = []

    probe_directions = np.eye(
        k_rank,
        dtype=float,
    )

    if k_rank >= 2:
        diagonal_direction = np.ones(
            k_rank,
            dtype=float,
        )

        diagonal_direction /= np.linalg.norm(
            diagonal_direction
        )

        probe_directions = np.vstack(
            [
                probe_directions,
                diagonal_direction,
            ]
        )

    for anchor in anchor_points:
        if anchor[0] > 0.0:
            base_jnd = 0.30
        else:
            base_jnd = 0.65

        jnds = np.maximum(
            base_jnd
            + 0.04
            * rng.standard_normal(
                len(probe_directions)
            ),
            0.08,
        )

        displacements = (
            probe_directions
            * jnds[:, None]
        )

        G_local = fit_local_metric_at_anchor(
            displacements=displacements,
            jnd_thresholds=jnds,
            G_prior=G_prior,
            shrink_lambda=0.25,
            reg=1e-3,
        )

        anchor_metrics.append(
            G_local
        )

    anchor_metrics = np.asarray(
        anchor_metrics,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 3. Données d'entraînement de W_theta
    # ------------------------------------------------------------------
    n_train = 24

    X_train = rng.standard_normal(
        (n_train, p_dimensions)
    )

    i_train, j_train = upper_tri_pairs(
        n_train
    )

    true_d_train = np.linalg.norm(
        (
            X_train[i_train]
            - X_train[j_train]
        ) @ L_true,
        axis=1,
    )

    similarity_train = np.zeros(
        (n_train, n_train),
        dtype=float,
    )

    similarity_values = np.clip(
        7.0
        - 1.5 * true_d_train
        + 0.25
        * rng.standard_normal(
            len(true_d_train)
        ),
        1.0,
        7.0,
    )

    similarity_train[
        i_train,
        j_train
    ] = similarity_values

    similarity_train[
        j_train,
        i_train
    ] = similarity_values

    np.fill_diagonal(
        similarity_train,
        7.0,
    )

    D_subjective_train = similarity_to_dissimilarity(
        similarity_train,
        mode="linear_max",
        min_similarity=1.0,
        max_similarity=7.0,
    )

    mu_train, sd_train = compute_unified_scaler(
        X_train,
        ddof=scaler_ddof,
    )

    X_train_scaled = apply_unified_scaler(
        X_train,
        mu_train,
        sd_train,
    )

    L_hat, W_hat = fit_from_pairwise_dissimilarity(
        X_train_scaled,
        D_subjective_train,
        k=k_rank,
        reg=1e-2,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 4. Référence de densité globale sur les ponts de validation
    # ------------------------------------------------------------------
    reference_density = compute_reference_global_density(
        reference_points=aligned_bridges[val_ids],
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        k_rank=k_rank,
    )

    # Points Monte-Carlo communs aux analyses observées et permutées.
    unit_ball_points = make_uniform_unit_ball_points(
        n_samples=n_mc_samples,
        dimension=k_rank,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 5. Conditions expérimentales appariées
    # ------------------------------------------------------------------
    n_pairs = 16

    base_states = rng.standard_normal(
        (n_pairs, p_dimensions)
    )

    X_condition_a = (
        0.65 * base_states
        + 0.10
        * rng.standard_normal(
            (n_pairs, p_dimensions)
        )
    )

    X_condition_b = (
        1.35 * base_states
        + 0.60
        + 0.10
        * rng.standard_normal(
            (n_pairs, p_dimensions)
        )
    )

    pair_ids = np.concatenate(
        [
            np.arange(n_pairs),
            np.arange(n_pairs),
        ]
    )

    volume_a = compute_calibrated_subjective_volume(
        X_condition=X_condition_a,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        n_mc_samples=n_mc_samples,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    volume_b = compute_calibrated_subjective_volume(
        X_condition=X_condition_b,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        n_mc_samples=n_mc_samples,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    comparison_a = compare_global_vs_local_volume(
        X_condition=X_condition_a,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        reference_density_info=reference_density,
        bandwidth=bandwidth,
        n_mc_samples=n_mc_samples,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    comparison_b = compare_global_vs_local_volume(
        X_condition=X_condition_b,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        reference_density_info=reference_density,
        bandwidth=bandwidth,
        n_mc_samples=n_mc_samples,
        seed=seed,
        unit_ball_points=unit_ball_points,
    )

    # ------------------------------------------------------------------
    # 6. Distance du chemin rectiligne entre les barycentres
    # ------------------------------------------------------------------
    Z_a = project_states(
        X_condition_a,
        mu_train,
        sd_train,
        L_hat,
        center_states=False,
    )

    Z_b = project_states(
        X_condition_b,
        mu_train,
        sd_train,
        L_hat,
        center_states=False,
    )

    mean_a = np.mean(
        Z_a,
        axis=0,
    )

    mean_b = np.mean(
        Z_b,
        axis=0,
    )

    straight_path_distance = path_integrated_distance(
        waypoints=np.vstack(
            [
                mean_a,
                mean_b,
            ]
        ),
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        n_substeps=20,
        bandwidth=bandwidth,
    )

    # ------------------------------------------------------------------
    # 7. Test de permutation
    # ------------------------------------------------------------------
    permutation_result = permutation_test_conditions(
        X_cond_A=X_condition_a,
        X_cond_B=X_condition_b,
        L_hat=L_hat,
        mu_train=mu_train,
        sd_train=sd_train,
        anchor_points=anchor_points,
        anchor_metrics=anchor_metrics,
        bandwidth=bandwidth,
        n_mc_samples=n_mc_samples,
        n_perm=n_perm,
        pair_ids=pair_ids,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------
    if (
        "__file__" in globals()
        and os.path.exists(__file__)
    ):
        pipeline_hash = _compute_file_sha256(
            __file__
        )
    else:
        pipeline_hash = "inline"

    summary = {
        "metadata": {
            "schema_version": 1,
            "seed": int(seed),
            "n_perm": int(n_perm),
            "n_mc_samples": int(n_mc_samples),
            "k_rank": int(k_rank),
            "bandwidth": float(bandwidth),
            "scaler_ddof": int(scaler_ddof),
            "bridge_calibration_ids": cal_ids,
            "bridge_validation_ids": val_ids,
            "bridge_test_ids": test_ids,
            "code_hash": pipeline_hash,
        },
        "bridge_calibration": {
            "selected_model": bridge_calibration[
                "selected_model"
            ],
            "coordinate_scale_model": bridge_calibration[
                "coordinate_scale_model"
            ],
            "coordinate_scale": bridge_calibration[
                "coordinate_scale"
            ],
            "procrustes_disparity": bridge_calibration[
                "procrustes_fit"
            ]["procrustes_disparity"],
            "test_metrics": bridge_calibration[
                "test_metrics"
            ],
            "non_equivalence_warning": bridge_calibration[
                "non_equivalence_warning"
            ],
            "reference_density": reference_density,
        },
        "condition_A": _format_volume_result(
            volume_a
        ),
        "condition_B": _format_volume_result(
            volume_b
        ),
        "global_vs_local_comparison": {
            "condition_A": comparison_a,
            "condition_B": comparison_b,
        },
        "straight_path_distance_A_to_B": float(
            straight_path_distance
        ),
        "permutation_test_B_vs_A": permutation_result,
    }

    if export_json_path is not None:
        export_pipeline_results(
            summary,
            export_json_path,
            fmt="json",
        )

    if export_csv_path is not None:
        export_pipeline_results(
            summary,
            export_csv_path,
            fmt="csv",
        )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    return summary


if __name__ == "__main__":
    run_full_demonstration(
        seed=2026,
        n_perm=499,
        k_rank=2,
        bandwidth=1.5,
        scaler_ddof=1,
        n_mc_samples=400,
        export_json_path="resultats_volume_subjectif.json",
        export_csv_path="resultats_volume_subjectif.csv",
  )
