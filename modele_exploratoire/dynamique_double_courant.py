#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamique exploratoire double courant sur un graphe.

Les champs sont :

- F : champ principal ;
- I : champ secondaire ;
- B : variable binaire dérivée ;
- Q = 2B - 1 : courant discret.

La dynamique est définie sur le Laplacien de graphe :

    L = A - D

Ce module ne constitue pas une mesure biologique ou phénoménologique.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


ALPHA = 0.25
BETA = 0.05
G = 0.6
D_F = 0.1
D_I = 8.0
GAMMA = 1.0
SEUIL_B = 0.0


def valider_adjacence(
    A: np.ndarray,
) -> np.ndarray:
    """
    Vérifie une matrice d'adjacence symétrique.
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


def laplacien_graphe(
    A: np.ndarray,
) -> np.ndarray:
    """
    Construit le Laplacien diffusif :

        L = A - D

    où D est la matrice diagonale des degrés.
    """
    matrix = valider_adjacence(
        A
    )

    degrees = np.sum(
        matrix,
        axis=1,
    )

    return (
        matrix
        - np.diag(degrees)
    )


def _valider_champs(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vérifie les champs nodaux.
    """
    field_f = np.asarray(
        F,
        dtype=float,
    ).reshape(-1)

    field_i = np.asarray(
        I,
        dtype=float,
    ).reshape(-1)

    field_b = np.asarray(
        B,
        dtype=float,
    ).reshape(-1)

    if not (
        len(field_f)
        == len(field_i)
        == len(field_b)
        == n_nodes
    ):
        raise ValueError(
            "F, I et B doivent avoir la taille du graphe."
        )

    if not np.all(
        np.isfinite(field_f)
    ) or not np.all(
        np.isfinite(field_i)
    ) or not np.all(
        np.isfinite(field_b)
    ):
        raise ValueError(
            "F, I ou B contient des valeurs non finies."
        )

    return field_f, field_i, field_b


def step(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    L: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    g: float = G,
    d_f: float = D_F,
    d_i: float = D_I,
    gamma: float = GAMMA,
    seuil: float = SEUIL_B,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Effectue un pas de dynamique double courant sur un graphe.
    """
    laplacian = np.asarray(
        L,
        dtype=float,
    )

    if laplacian.ndim != 2:
        raise ValueError(
            "L doit être une matrice 2D."
        )

    if laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError(
            "L doit être carrée."
        )

    n_nodes = laplacian.shape[0]

    field_f, field_i, field_b = _valider_champs(
        F,
        I,
        B,
        n_nodes,
    )

    current = (
        2.0
        * field_b
        - 1.0
    )

    f_next = (
        (1.0 - alpha)
        * field_f
        + alpha
        * np.tanh(
            field_i
            + g * current
            + d_f * (laplacian @ field_f)
        )
    )

    i_next = (
        (1.0 - beta)
        * field_i
        + beta
        * np.tanh(
            f_next
            + d_i * (laplacian @ field_i)
            - gamma * field_i ** 3
        )
    )

    b_next = (
        f_next
        + 0.5 * i_next
        > seuil
    ).astype(float)

    return f_next, i_next, b_next


def spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Corrélation de Spearman sans dépendance supplémentaire.
    """
    x_arr = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    y_arr = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    if len(x_arr) != len(y_arr):
        raise ValueError(
            "x et y doivent avoir la même longueur."
        )

    if len(x_arr) < 2:
        return 0.0

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(
            values,
            kind="mergesort",
        )

        ranks = np.empty(
            len(values),
            dtype=float,
        )

        start = 0

        while start < len(values):
            end = start + 1

            while (
                end < len(values)
                and values[
                    order[end]
                ]
                == values[
                    order[start]
                ]
            ):
                end += 1

            mean_rank = 0.5 * (
                start
                + end
                - 1
            )

            ranks[
                order[start:end]
            ] = mean_rank

            start = end

        return ranks

    rank_x = rank(
        x_arr
    )

    rank_y = rank(
        y_arr
    )

    std_x = float(
        np.std(rank_x)
    )

    std_y = float(
        np.std(rank_y)
    )

    if std_x <= 1e-15 or std_y <= 1e-15:
        return 0.0

    return float(
        np.corrcoef(
            rank_x,
            rank_y,
        )[0, 1]
    )


def diagnostique_dynamo(
    A: np.ndarray,
    F_history: np.ndarray,
    I_history: np.ndarray,
) -> float:
    """
    Mesure un couplage moyen des gradients sur les arêtes du graphe.

    La mesure utilisée est :

        mean((F_i-F_j)(I_i-I_j))
        /
        sqrt(mean((F_i-F_j)^2) mean((I_i-I_j)^2))
    """
    adjacency = valider_adjacence(
        A
    )

    f_hist = np.asarray(
        F_history,
        dtype=float,
    )

    i_hist = np.asarray(
        I_history,
        dtype=float,
    )

    if f_hist.ndim != 2 or i_hist.ndim != 2:
        raise ValueError(
            "Les historiques doivent être des matrices 2D."
        )

    if f_hist.shape != i_hist.shape:
        raise ValueError(
            "F_history et I_history doivent avoir la même forme."
        )

    if f_hist.shape[1] != adjacency.shape[0]:
        raise ValueError(
            "Les historiques doivent correspondre au graphe."
        )

    edges = np.argwhere(
        np.triu(
            adjacency,
            1,
        )
        > 0.0
    )

    if len(edges) == 0:
        return 0.0

    delta_f = (
        f_hist[:, edges[:, 0]]
        - f_hist[:, edges[:, 1]]
    )

    delta_i = (
        i_hist[:, edges[:, 0]]
        - i_hist[:, edges[:, 1]]
    )

    numerator = float(
        np.mean(
            delta_f
            * delta_i
        )
    )

    denominator = math_sqrt_safe(
        float(
            np.mean(
                delta_f ** 2
            )
        )
        * float(
            np.mean(
                delta_i ** 2
            )
        )
    )

    return float(
        numerator
        / max(
            denominator,
            1e-12,
        )
    )


def math_sqrt_safe(
    value: float,
) -> float:
    """
    Racine carrée protégée contre les petites valeurs négatives numériques.
    """
    return float(
        np.sqrt(
            max(
                value,
                0.0,
            )
        )
    )


def simuler_graphe(
    A: np.ndarray,
    seed: int = 42,
    t_total: int = 500,
    alpha: float = ALPHA,
    beta: float = BETA,
    g: float = G,
    d_f: float = D_F,
    d_i: float = D_I,
    gamma: float = GAMMA,
    seuil: float = SEUIL_B,
) -> Dict[str, np.ndarray]:
    """
    Simule la dynamique double courant sur un graphe.
    """
    adjacency = valider_adjacence(
        A
    )

    t_total = int(
        t_total
    )

    if t_total < 1:
        raise ValueError(
            "t_total doit être au moins égal à 1."
        )

    n_nodes = adjacency.shape[0]
    laplacian = laplacien_graphe(
        adjacency
    )

    rng = np.random.default_rng(
        int(seed)
    )

    F = rng.uniform(
        -0.5,
        0.5,
        size=n_nodes,
    )

    I = rng.uniform(
        -0.5,
        0.5,
        size=n_nodes,
    )

    B = (
        F
        + 0.5 * I
        > seuil
    ).astype(float)

    Q_history = []
    F_history = []
    I_history = []

    for _ in range(
        t_total
    ):
        Q = (
            2.0
            * B
            - 1.0
        )

        F, I, B = step(
            F,
            I,
            B,
            laplacian,
            alpha=alpha,
            beta=beta,
            g=g,
            d_f=d_f,
            d_i=d_i,
            gamma=gamma,
            seuil=seuil,
        )

        Q_history.append(
            Q.copy()
        )

        F_history.append(
            F.copy()
        )

        I_history.append(
            I.copy()
        )

    return {
        "Q": np.asarray(
            Q_history,
            dtype=float,
        ),
        "F": np.asarray(
            F_history,
            dtype=float,
        ),
        "I": np.asarray(
            I_history,
            dtype=float,
        ),
        "B_final": B.copy(),
    }
