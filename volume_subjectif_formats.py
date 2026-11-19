#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outils pour l'apprentissage d'une métrique subjective de faible rang.

Convention
----------
X : matrice d'états          (n_states, n_features)
D : matrice de dissimilarité (n_states, n_states)
L : projection latente       (n_features, k)
W : métrique                 (n_features, n_features)

Distance prédite :
    d_ij = || (x_i - x_j) @ L ||_2

et
    W = L @ L.T
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


__all__ = [
    "upper_tri_pairs",
    "ball_volume_factor",
    "similarity_to_dissimilarity",
    "fit_from_pairwise_dissimilarity",
    "predict_distances",
    "SubjectiveMetric",
]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def upper_tri_pairs(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices de la partie triangulaire supérieure stricte."""
    n = int(n)
    if n < 2:
        raise ValueError("n doit être ≥ 2.")
    return np.triu_indices(n, k=1)


def ball_volume_factor(k: int) -> float:
    """Volume de la boule unité euclidienne en dimension k."""
    k = int(k)
    if k < 1:
        raise ValueError("k doit être un entier positif.")
    return float(math.pi ** (k / 2.0) / math.gamma(k / 2.0 + 1.0))


def _validate_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    """Valide et symétrise une matrice carrée."""
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

    Modes
    -----
    linear_max   : D = max(S) - S
    linear_range : D = (max(S) - S) / (max(S) - min(S))   → [0, 1]
    one_minus    : D = 1 - S
    """
    S = _validate_square_matrix(similarity_matrix, "similarity_matrix")
    mode = str(mode).lower()

    s_min = float(np.min(S)) if min_similarity is None else float(min_similarity)
    s_max = float(np.max(S)) if max_similarity is None else float(max_similarity)

    if not (np.isfinite(s_min) and np.isfinite(s_max)):
        raise ValueError("Les bornes de similarité doivent être finies.")
    if s_max <= s_min:
        raise ValueError("max_similarity doit être strictement supérieur à min_similarity.")

    if mode == "linear_max":
        D = s_max - S
    elif mode == "linear_range":
        D = (s_max - S) / (s_max - s_min)
    elif mode == "one_minus":
        D = 1.0 - S
    else:
        raise ValueError("Mode inconnu. Utiliser 'linear_max', 'linear_range' ou 'one_minus'.")

    D = 0.5 * (D + D.T)
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def predict_distances(X: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice complète des distances prédites.

    Parameters
    ----------
    X : (n, p)
    L : (p, k)

    Returns
    -------
    D_pred : (n, n)
    """
    X = np.asarray(X, dtype=float)
    L = np.asarray(L, dtype=float)
    latent = X @ L
    # distances euclidiennes pairwise
    sq = np.sum(latent**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (latent @ latent.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)


# ---------------------------------------------------------------------------
# Initialisation MDS classique
# ---------------------------------------------------------------------------

def _classical_mds_initialization(
    D: np.ndarray,
    X: np.ndarray,
    k: int,
    reg: float,
) -> np.ndarray:
    """Initialise L par MDS classique + régression ridge."""
    n = D.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (D ** 2) @ centering
    gram = 0.5 * (gram + gram.T)

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    Z0 = np.zeros((n, k), dtype=float)
    positive = eigvals > 1e-10
    if np.any(positive):
        selected = np.where(positive)[0][:k]
        Z0[:, : len(selected)] = (
            eigvecs[:, selected] * np.sqrt(np.maximum(eigvals[selected], 0.0))
        )

    xtx = X.T @ X
    ridge = xtx + max(float(reg), 1e-12) * np.eye(X.shape[1])

    try:
        L0 = np.linalg.solve(ridge, X.T @ Z0)
    except np.linalg.LinAlgError:
        L0 = np.linalg.pinv(ridge) @ X.T @ Z0

    return L0


# ---------------------------------------------------------------------------
# Fonction principale d'ajustement
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    L: np.ndarray
    W: np.ndarray
    success: bool
    message: str
    n_iter: int
    final_loss: float
    data_loss: float
    penalty: float


def fit_from_pairwise_dissimilarity(
    X_train: np.ndarray,
    D_train: np.ndarray,
    k: int,
    reg: float = 1e-2,
    max_iter: int = 2000,
    standardize: bool = False,
    return_details: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | FitResult:
    """
    Ajuste une métrique de faible rang à partir d'une RDM.

    Parameters
    ----------
    X_train : (n_states, n_features)
        Matrice d'états. Si standardize=True, elle sera centrée/réduite
        en interne (moyenne et écart-type sont alors stockés dans le résultat
        si return_details=True).
    D_train : (n_states, n_states)
        Matrice de dissimilarité.
    k : int
        Rang latent.
    reg : float
        Régularisation ridge sur L.
    max_iter : int
        Nombre max d'itérations L-BFGS-B.
    standardize : bool
        Si True, standardise X_train (moyenne 0, variance 1) avant l'ajustement.
    return_details : bool
        Si True, retourne un objet FitResult riche au lieu de (L, W).

    Returns
    -------
    L_hat, W_hat   ou   FitResult
    """
    X = np.asarray(X_train, dtype=float)
    D = _validate_square_matrix(D_train, "D_train")

    if X.ndim != 2:
        raise ValueError("X_train doit être de forme (n_states, n_features).")
    n_states, n_features = X.shape

    if D.shape != (n_states, n_states):
        raise ValueError("D_train doit avoir la même première dimension que X_train.")

    k = int(k)
    if not (1 <= k <= n_features):
        raise ValueError(f"k doit être entre 1 et {n_features}.")
    if reg < 0:
        raise ValueError("reg doit être ≥ 0.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X_train contient des valeurs non finies.")
    if np.any(D < -1e-12):
        raise ValueError("D_train contient des dissimilarités négatives.")

    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)

    # Standardisation optionnelle
    mu = np.zeros(n_features)
    sd = np.ones(n_features)
    if standardize:
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0, ddof=1)
        sd = np.maximum(sd, 1e-8)
        X = (X - mu) / sd

    X_centered = X - np.mean(X, axis=0, keepdims=True)  # pour l'optimisation

    i_upper, j_upper = upper_tri_pairs(n_states)
    target = D[i_upper, j_upper]

    if len(target) == 0:
        raise ValueError("Il faut au moins une paire d'états.")

    # Initialisation
    L0 = _classical_mds_initialization(
        D=D, X=X_centered, k=k, reg=max(reg, 1e-12)
    )

    def unpack(theta: np.ndarray) -> np.ndarray:
        return theta.reshape(n_features, k)

    def objective(theta: np.ndarray) -> float:
        L = unpack(theta)
        latent = X_centered @ L
        delta = latent[i_upper] - latent[j_upper]
        pred = np.linalg.norm(delta, axis=1)
        data_loss = float(np.mean((pred - target) ** 2))
        penalty = float(reg) * float(np.sum(L ** 2))
        return data_loss + penalty

    result = minimize(
        objective,
        x0=L0.ravel(),
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    L_hat = unpack(result.x)
    W_hat = L_hat @ L_hat.T
    W_hat = 0.5 * (W_hat + W_hat.T)

    if not (np.all(np.isfinite(L_hat)) and np.all(np.isfinite(W_hat))):
        raise RuntimeError("L_hat ou W_hat contient des valeurs non finies.")

    # Calcul final des pertes
    final_loss = float(result.fun)
    L_tmp = L_hat
    latent = X_centered @ L_tmp
    delta = latent[i_upper] - latent[j_upper]
    pred = np.linalg.norm(delta, axis=1)
    data_loss = float(np.mean((pred - target) ** 2))
    penalty = float(reg) * float(np.sum(L_tmp ** 2))

    if return_details:
        return FitResult(
            L=L_hat,
            W=W_hat,
            success=bool(result.success),
            message=str(result.message),
            n_iter=int(result.nit),
            final_loss=final_loss,
            data_loss=data_loss,
            penalty=penalty,
        )

    if not result.success:
        # On laisse quand même le résultat, mais on avertit
        import warnings
        warnings.warn(
            f"Optimisation non convergée : {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    return L_hat, W_hat


# ---------------------------------------------------------------------------
# Classe pratique
# ---------------------------------------------------------------------------

class SubjectiveMetric:
    """
    Interface orientée objet pour l'apprentissage et l'utilisation
    d'une métrique subjective de faible rang.
    """

    def __init__(self, k: int = 2, reg: float = 1e-2, standardize: bool = True):
        self.k = int(k)
        self.reg = float(reg)
        self.standardize = bool(standardize)
        self.L_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.fit_result_: Optional[FitResult] = None

    def fit(self, X: np.ndarray, D: np.ndarray) -> "SubjectiveMetric":
        """Ajuste la métrique."""
        X = np.asarray(X, dtype=float)
        if self.standardize:
            self.mu_ = np.mean(X, axis=0)
            self.sd_ = np.maximum(np.std(X, axis=0, ddof=1), 1e-8)
            X_proc = (X - self.mu_) / self.sd_
        else:
            self.mu_ = np.zeros(X.shape[1])
            self.sd_ = np.ones(X.shape[1])
            X_proc = X

        result = fit_from_pairwise_dissimilarity(
            X_proc,
            D,
            k=self.k,
            reg=self.reg,
            standardize=False,          # déjà fait
            return_details=True,
        )
        assert isinstance(result, FitResult)
        self.L_ = result.L
        self.W_ = result.W
        self.fit_result_ = result
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projette les états dans l'espace latent."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        X_proc = (X - self.mu_) / self.sd_
        return X_proc @ self.L_

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Matrice des distances prédites."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        X_proc = (X - self.mu_) / self.sd_
        return predict_distances(X_proc, self.L_)

    def _check_is_fitted(self) -> None:
        if self.L_ is None:
            raise RuntimeError("L'estimateur n'a pas encore été ajusté. Appelez fit().")


# ---------------------------------------------------------------------------
# Démonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(2026)

    n_states = 20
    n_features = 7
    k_rank = 2

    X = rng.standard_normal((n_states, n_features))

    L_true = np.zeros((n_features, k_rank))
    L_true[0, 0] = 1.2
    L_true[2, 0] = -0.8
    L_true[4, 1] = 1.0
    L_true[6, 1] = 0.7

    i_upper, j_upper = upper_tri_pairs(n_states)
    differences = (X[i_upper] - X[j_upper]) @ L_true
    d_true = np.linalg.norm(differences, axis=1)

    similarity = np.zeros((n_states, n_states))
    similarity[i_upper, j_upper] = 7.0 - d_true
    similarity[j_upper, i_upper] = 7.0 - d_true
    np.fill_diagonal(similarity, 7.0)

    D = similarity_to_dissimilarity(
        similarity,
        mode="linear_max",
        min_similarity=1.0,
        max_similarity=7.0,
    )

    # Version fonctionnelle
    L_hat, W_hat = fit_from_pairwise_dissimilarity(
        X, D, k=k_rank, reg=1e-2, standardize=True
    )

    print("=== Version fonctionnelle ===")
    print("L_hat shape :", L_hat.shape)
    print("W_hat shape :", W_hat.shape)
    print("Rang de W   :", np.linalg.matrix_rank(W_hat, tol=1e-8))

    # Version orientée objet
    metric = SubjectiveMetric(k=k_rank, reg=1e-2, standardize=True)
    metric.fit(X, D)

    print("\n=== Version orientée objet ===")
    print("Succès       :", metric.fit_result_.success)
    print("Itérations   :", metric.fit_result_.n_iter)
    print("Loss finale  :", f"{metric.fit_result_.final_loss:.6f}")
    print("Distances prédites shape :", metric.pairwise_distances(X).shape)
