#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Méthodes de compression de graphes pour le modèle exploratoire.

Ces fonctions ne représentent pas une procédure empirique validée.
La compression pondérée est une heuristique dépendant des champs F et I.
"""

from __future__ import annotations

import numpy as np


def _validate_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Vérifie et normalise une matrice d'adjacence non orientée.
    """
    matrix = np.asarray(
        A,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "A doit être une matrice 2D."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "A doit être carrée."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "A contient des valeurs non finies."
        )

    if np.any(matrix < 0.0):
        raise ValueError(
            "A ne peut pas contenir de valeurs négatives."
        )

    matrix = 0.5 * (
        matrix
        + matrix.T
    )

    np.fill_diagonal(
        matrix,
        0.0,
    )

    return matrix


def extraire_aretes(
    A: np.ndarray,
) -> np.ndarray:
    """
    Retourne les arêtes non orientées sous la forme (i, j).
    """
    matrix = _validate_adjacency(
        A
    )

    return np.argwhere(
        np.triu(
            matrix,
            1,
        )
        > 0.0
    )


def compresser_aleatoire(
    A: np.ndarray,
    n_aretes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Conserve aléatoirement n_aretes arêtes non orientées.
    """
    matrix = _validate_adjacency(
        A
    )

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng doit être un numpy.random.Generator."
        )

    edges = extraire_aretes(
        matrix
    )

    n_aretes = int(
        n_aretes
    )

    if not (
        0
        <= n_aretes
        <= len(edges)
    ):
        raise ValueError(
            "Nombre d'arêtes invalide."
        )

    compressed = np.zeros_like(
        matrix
    )

    if n_aretes == 0:
        return compressed

    selected = edges[
        rng.choice(
            len(edges),
            size=n_aretes,
            replace=False,
        )
    ]

    compressed[
        selected[:, 0],
        selected[:, 1],
    ] = 1.0

    compressed[
        selected[:, 1],
        selected[:, 0],
    ] = 1.0

    return compressed


def compresser_ponderee(
    A: np.ndarray,
    F: np.ndarray,
    I: np.ndarray,
    n_aretes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compression heuristique pondérée par la similarité locale des champs.

    Les arêtes reliant des nœuds ayant des valeurs proches de F et I
    reçoivent une probabilité plus élevée. Cette règle est un choix
    algorithmique exploratoire et non une loi physique établie.
    """
    matrix = _validate_adjacency(
        A
    )

    field_f = np.asarray(
        F,
        dtype=float,
    ).reshape(-1)

    field_i = np.asarray(
        I,
        dtype=float,
    ).reshape(-1)

    n_nodes = matrix.shape[0]

    if len(field_f) != n_nodes:
        raise ValueError(
            "F doit avoir autant d'éléments que A contient de nœuds."
        )

    if len(field_i) != n_nodes:
        raise ValueError(
            "I doit avoir autant d'éléments que A contient de nœuds."
        )

    if not np.all(
        np.isfinite(field_f)
    ) or not np.all(
        np.isfinite(field_i)
    ):
        raise ValueError(
            "F et I contiennent des valeurs non finies."
        )

    edges = extraire_aretes(
        matrix
    )

    n_aretes = int(
        n_aretes
    )

    if not (
        0
        <= n_aretes
        <= len(edges)
    ):
        raise ValueError(
            "Nombre d'arêtes invalide."
        )

    compressed = np.zeros_like(
        matrix
    )

    if n_aretes == 0:
        return compressed

    delta_f = (
        field_f[edges[:, 0]]
        - field_f[edges[:, 1]]
    )

    delta_i = (
        field_i[edges[:, 0]]
        - field_i[edges[:, 1]]
    )

    scale_f = max(
        float(np.std(field_f)),
        1e-12,
    )

    scale_i = max(
        float(np.std(field_i)),
        1e-12,
    )

    scores = np.exp(
        -np.abs(delta_f) / scale_f
        -0.5 * np.abs(delta_i) / scale_i
    )

    probabilities = (
        scores
        / max(
            float(np.sum(scores)),
            1e-15,
        )
    )

    selected_indices = rng.choice(
        len(edges),
        size=n_aretes,
        replace=False,
        p=probabilities,
    )

    selected = edges[
        selected_indices
    ]

    compressed[
        selected[:, 0],
        selected[:, 1],
    ] = 1.0

    compressed[
        selected[:, 1],
        selected[:, 0],
    ] = 1.0

    return compressed
