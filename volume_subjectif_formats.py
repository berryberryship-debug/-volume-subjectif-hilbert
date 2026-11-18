#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outils de base pour l'apprentissage d'une métrique subjective.

Convention :
    X : matrice d'états (n_states, n_features)
    D : matrice de dissimilarité (n_states, n_states)
    L : projection latente (n_features, k)
    W : métrique (n_features, n_features)

La distance prédite entre deux états xi et xj est :

    d_ij = ||(xi - xj) @ L||_2

et :

    W = L @ L.T
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.optimize import minimize


__all__ = [
    "upper_tri_pairs",
    "ball_volume_factor",
    "similarity_to_dissimilarity",
    "fit_from_pairwise_dissimilarity",
]


def upper_tri_pairs(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne les indices de la partie triangulaire supérieure stricte.
    """
    n = int(n)

    if n < 2:
        raise ValueError("n doit être supérieur ou égal à 2.")

    return np.triu_indices(n, k=1)


def ball_volume_factor(k: int) -> float:
    """
    Volume de la boule unité euclidienne en dimension k.

    V_k = pi^(k/2) / Gamma(k/2 + 1)
    """
    k = int(k)

    if k < 1:
        raise ValueError("La dimension k doit être un entier positif.")

    return float(
        math.pi ** (k / 2.0) / math.gamma(k / 2.0 + 1.0)
    )


def _validate_square_matrix(
    matrix: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Valide et symétrise une matrice carrée.
    """
    arr = np.asarray(matrix, dtype=float)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} doit être une matrice carrée.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contient des valeurs non finies.")

    return 0.5 * (arr + arr.T)


def similarity_to_dissimilarity(
    similarity_matrix: np.ndarray,
    mode: str = "linear_max",
    min_similarity: float | None = None,
    max_similarity: float | None = None,
) -> np.ndarray:
    """
    Convertit une matrice de similarité en matrice de dissimilarité.

    Modes disponibles :

    linear_max
        D = max(S) - S.
        Préserve l'échelle brute des notes.

    linear_range
        D = (max(S) - S) / (max(S) - min(S)).
        Produit une dissimilarité comprise entre 0 et 1.

    one_minus
        D = 1 - S.
        Adapté aux similarités déjà comprises entre 0 et 1.

    Parameters
    ----------
    similarity_matrix
        Matrice carrée de similarité.

    mode
        Mode de conversion.

    min_similarity, max_similarity
        Bornes externes facultatives de l'échelle de similarité.
        Par exemple 1 et 7 pour une échelle de Likert 1-7.

    Returns
    -------
    np.ndarray
        Matrice symétrique de dissimilarité à diagonale nulle.
    """
    S = _validate_square_matrix(
        similarity_matrix,
        "similarity_matrix",
    )

    mode = str(mode).lower()

    if min_similarity is None:
        s_min = float(np.min(S))
    else:
        s_min = float(min_similarity)

    if max_similarity is None:
        s_max = float(np.max(S))
    else:
        s_max = float(max_similarity)

    if not np.isfinite(s_min) or not np.isfinite(s_max):
        raise ValueError(
            "Les bornes de similarité doivent être finies."
        )

    if s_max <= s_min:
        raise ValueError(
            "max_similarity doit être supérieur à min_similarity."
        )

    if mode == "linear_max":
        D = s_max - S

    elif mode == "linear_range":
        D = (s_max - S) / (s_max - s_min)

    elif mode == "one_minus":
        D = 1.0 - S

    else:
        raise ValueError(
            "Mode inconnu. Utiliser 'linear_max', "
            "'linear_range' ou 'one_minus'."
        )

    D = 0.5 * (D + D.T)
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)

    return D


def _classical_mds_initialization(
    D: np.ndarray,
    X: np.ndarray,
    k: int,
    reg: float,
) -> np.ndarray:
    """
    Initialise L par MDS classique suivi d'une régression ridge.
    """
    n = D.shape[0]

    centering = (
        np.eye(n)
        - np.ones((n, n), dtype=float) / float(n)
    )

    gram = -0.5 * centering @ (D ** 2) @ centering
    gram = 0.5 * (gram + gram.T)

    eigvals, eigvecs = np.linalg.eigh(gram)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-10
    Z0 = np.zeros((n, k), dtype=float)

    if np.any(positive):
        selected = np.where(positive)[0][:k]
        n_selected = len(selected)

        Z0[:, :n_selected] = (
            eigvecs[:, selected]
            * np.sqrt(
                np.maximum(eigvals[selected], 0.0)
            )[None, :]
        )

    xtx = X.T @ X
    ridge_matrix = (
        xtx + float(reg) * np.eye(X.shape[1])
    )

    try:
        L0 = np.linalg.solve(
            ridge_matrix,
            X.T @ Z0,
        )
    except np.linalg.LinAlgError:
        L0 = (
            np.linalg.pinv(ridge_matrix)
            @ X.T
            @ Z0
        )

    return L0


def fit_from_pairwise_dissimilarity(
    X_train: np.ndarray,
    D_train: np.ndarray,
    k: int,
    reg: float = 1e-2,
    seed: int = 42,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ajuste une métrique de faible rang à partir d'une RDM.

    Parameters
    ----------
    X_train
        Matrice d'états de forme (n_states, n_features).

        Le scaling doit être réalisé avant l'appel à cette fonction.
        Cette fonction ne recalcule donc ni moyenne ni écart-type.

    D_train
        Matrice de dissimilarité de forme (n_states, n_states).

    k
        Rang latent recherché.

    reg
        Régularisation ridge appliquée à L.

    seed
        Graine conservée dans l'interface pour la reproductibilité.
        L'optimisation utilisée ici est déterministe.

    max_iter
        Nombre maximal d'itérations de l'optimiseur.

    Returns
    -------
    L_hat
        Matrice de projection de forme (n_features, k).

    W_hat
        Matrice métrique :

            W_hat = L_hat @ L_hat.T

    Notes
    -----
    La matrice L n'est pas unique à rotation orthogonale près.
    La quantité principalement interprétable est W_hat.
    """
    del seed

    X = np.asarray(X_train, dtype=float)
    D = _validate_square_matrix(
        D_train,
        "D_train",
    )

    if X.ndim != 2:
        raise ValueError(
            "X_train doit être une matrice à deux dimensions."
        )

    n_states, n_features = X.shape

    if D.shape != (n_states, n_states):
        raise ValueError(
            "D_train doit avoir une dimension correspondant "
            "au nombre d'états de X_train."
        )

    k = int(k)

    if k < 1 or k > n_features:
        raise ValueError(
            "k doit être compris entre 1 et le nombre "
            "de variables de X_train."
        )

    if reg < 0:
        raise ValueError(
            "reg doit être positif ou nul."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X_train contient des valeurs non finies."
        )

    if np.any(D < -1e-12):
        raise ValueError(
            "D_train contient des dissimilarités négatives."
        )

    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)

    i_upper, j_upper = upper_tri_pairs(n_states)
    target_distances = D[i_upper, j_upper]

    if len(target_distances) == 0:
        raise ValueError(
            "Il faut au moins une paire d'états."
        )

    X_centered = (
        X - np.mean(X, axis=0, keepdims=True)
    )

    L0 = _classical_mds_initialization(
        D=D,
        X=X_centered,
        k=k,
        reg=max(float(reg), 1e-12),
    )

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float).reshape(
            n_features,
            k,
        )

    def objective(theta: np.ndarray) -> float:
        L = unpack(theta)

        latent = X_centered @ L
        delta = (
            latent[i_upper]
            - latent[j_upper]
        )

        predicted_distances = np.linalg.norm(
            delta,
            axis=1,
        )

        residual = (
            predicted_distances
            - target_distances
        )

        data_loss = float(
            np.mean(residual ** 2)
        )

        penalty = float(reg) * float(
            np.sum(L ** 2)
        )

        return data_loss + penalty

    result = minimize(
        objective,
        x0=L0.reshape(-1),
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
            "Échec de l'optimisation de la métrique : "
            f"{result.message}"
        )

    L_hat = unpack(result.x)

    W_hat = L_hat @ L_hat.T
    W_hat = 0.5 * (W_hat + W_hat.T)

    if not np.all(np.isfinite(L_hat)):
        raise RuntimeError(
            "L_hat contient des valeurs non finies."
        )

    if not np.all(np.isfinite(W_hat)):
        raise RuntimeError(
            "W_hat contient des valeurs non finies."
        )

    return L_hat, W_hat


if __name__ == "__main__":
    rng = np.random.default_rng(2026)

    n_states = 20
    n_features = 7
    k_rank = 2

    X = rng.standard_normal(
        (n_states, n_features)
    )

    L_true = np.zeros(
        (n_features, k_rank)
    )

    L_true[0, 0] = 1.2
    L_true[2, 0] = -0.8
    L_true[4, 1] = 1.0
    L_true[6, 1] = 0.7

    i_upper, j_upper = upper_tri_pairs(
        n_states
    )

    differences = (
        X[i_upper]
        - X[j_upper]
    ) @ L_true

    d_true = np.linalg.norm(
        differences,
        axis=1,
    )

    similarity = np.zeros(
        (n_states, n_states),
        dtype=float,
    )

    similarity[i_upper, j_upper] = (
        7.0 - d_true
    )

    similarity[j_upper, i_upper] = (
        7.0 - d_true
    )

    np.fill_diagonal(
        similarity,
        7.0,
    )

    D = similarity_to_dissimilarity(
        similarity,
        mode="linear_max",
        min_similarity=1.0,
        max_similarity=7.0,
    )

    mu = np.mean(
        X,
        axis=0,
    )

    sd = np.maximum(
        np.std(X, axis=0, ddof=1),
        1e-8,
    )

    X_std = (
        X - mu
    ) / sd

    L_hat, W_hat = (
        fit_from_pairwise_dissimilarity(
            X_std,
            D,
            k=k_rank,
            reg=1e-2,
            seed=2026,
        )
    )

    print("X_std :", X_std.shape)
    print("D     :", D.shape)
    print("L_hat :", L_hat.shape)
    print("W_hat :", W_hat.shape)
    print(
        "Rang numérique de W_hat :",
        np.linalg.matrix_rank(W_hat),
    )
