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
    D_ref = np.linalg.norm(X_ref[:, None,
