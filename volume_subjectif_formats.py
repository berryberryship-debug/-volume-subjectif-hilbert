#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outils pour l'apprentissage d'une métrique subjective de faible rang
avec sélection automatique du rang k par validation croisée.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize


__all__ = [
    "upper_tri_pairs",
    "ball_volume_factor",
    "similarity_to_dissimilarity",
    "predict_distances",
    "fit_from_pairwise_dissimilarity",
    "select_k_by_cross_validation",
    "SubjectiveMetric",
    "FitResult",
    "CVResult",
]


# ---------------------------------------------------------------------------
# Utilitaires de base
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
    """Convertit une matrice de similarité en dissimilarité."""
    S = _validate_square_matrix(similarity_matrix, "similarity_matrix")
    mode = str(mode).lower()

    s_min = float(np.min(S)) if min_similarity is None else float(min_similarity)
    s_max = float(np.max(S)) if max_similarity is None else float(max_similarity)

    if not (np.isfinite(s_min) and np.isfinite(s_max)):
        raise ValueError("Les bornes de similarité doivent être finies.")
    if s_max <= s_min:
        raise ValueError("max_similarity doit être > min_similarity.")

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
    """Matrice complète des distances prédites."""
    X = np.asarray(X, dtype=float)
    L = np.asarray(L, dtype=float)
    latent = X @ L
    sq = np.sum(latent**2, axis=1, keepdims=True)
    D2 = np.maximum(sq + sq.T - 2.0 * (latent @ latent.T), 0.0)
    return np.sqrt(D2)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _classical_mds_initialization(
    D: np.ndarray,
    X: np.ndarray,
    k: int,
    reg: float,
) -> np.ndarray:
    n = D.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (D ** 2) @ centering
    gram = 0.5 * (gram + gram.T)

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    Z0 = np.zeros((n, k), dtype=float)
    positive = eigvals > 1e-10
    if np.any(positive):
        selected = np.where(positive)[0][:k]
        Z0[:, :len(selected)] = (
            eigvecs[:, selected] * np.sqrt(np.maximum(eigvals[selected], 0.0))
        )

    ridge = X.T @ X + max(float(reg), 1e-12) * np.eye(X.shape[1])
    try:
        return np.linalg.solve(ridge, X.T @ Z0)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(ridge) @ X.T @ Z0


# ---------------------------------------------------------------------------
# Résultats
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


@dataclass
class CVResult:
    best_k: int
    scores: dict          # k -> mean validation MSE
    std_scores: dict      # k -> std validation MSE
    all_fold_scores: dict # k -> list of fold scores


# ---------------------------------------------------------------------------
# Ajustement (avec support de sous-ensemble de paires)
# ---------------------------------------------------------------------------

def fit_from_pairwise_dissimilarity(
    X_train: np.ndarray,
    D_train: np.ndarray,
    k: int,
    reg: float = 1e-2,
    max_iter: int = 2000,
    standardize: bool = False,
    pair_mask: Optional[np.ndarray] = None,
    return_details: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], FitResult]:
    """
    Ajuste une métrique de faible rang.

    Parameters
    ----------
    pair_mask : array bool de forme (n_pairs,), optionnel
        Si fourni, n'utilise que les paires True pour l'ajustement
        (nécessaire pour la validation croisée).
    """
    X = np.asarray(X_train, dtype=float)
    D = _validate_square_matrix(D_train, "D_train")

    if X.ndim != 2:
        raise ValueError("X_train doit être de forme (n_states, n_features).")
    n_states, n_features = X.shape

    if D.shape != (n_states, n_states):
        raise ValueError("D_train doit correspondre au nombre d'états de X_train.")

    k = int(k)
    if not (1 <= k <= n_features):
        raise ValueError(f"k doit être entre 1 et {n_features}.")
    if reg < 0:
        raise ValueError("reg doit être ≥ 0.")

    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)

    # Standardisation
    if standardize:
        mu = np.mean(X, axis=0)
        sd = np.maximum(np.std(X, axis=0, ddof=1), 1e-8)
        X = (X - mu) / sd

    X_centered = X - np.mean(X, axis=0, keepdims=True)

    i_all, j_all = upper_tri_pairs(n_states)
    n_pairs = len(i_all)

    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != (n_pairs,):
            raise ValueError(f"pair_mask doit être de forme ({n_pairs},)")
        i_upper = i_all[pair_mask]
        j_upper = j_all[pair_mask]
    else:
        i_upper, j_upper = i_all, j_all

    target = D[i_upper, j_upper]
    if len(target) == 0:
        raise ValueError("Aucune paire disponible pour l'ajustement.")

    # Initialisation
    L0 = _classical_mds_initialization(D, X_centered, k, max(reg, 1e-12))

    def unpack(theta):
        return theta.reshape(n_features, k)

    def objective(theta):
        L = unpack(theta)
        latent = X_centered @ L
        delta = latent[i_upper] - latent[j_upper]
        pred = np.linalg.norm(delta, axis=1)
        data_loss = np.mean((pred - target) ** 2)
        penalty = reg * np.sum(L ** 2)
        return float(data_loss + penalty)

    result = minimize(
        objective,
        x0=L0.ravel(),
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )

    L_hat = unpack(result.x)
    W_hat = 0.5 * (L_hat @ L_hat.T + (L_hat @ L_hat.T).T)

    # Pertes finales
    latent = X_centered @ L_hat
    delta = latent[i_upper] - latent[j_upper]
    pred = np.linalg.norm(delta, axis=1)
    data_loss = float(np.mean((pred - target) ** 2))
    penalty = float(reg * np.sum(L_hat ** 2))
    final_loss = data_loss + penalty

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
        warnings.warn(f"Optimisation non convergée : {result.message}", RuntimeWarning)

    return L_hat, W_hat


# ---------------------------------------------------------------------------
# Validation croisée pour k
# ---------------------------------------------------------------------------

def select_k_by_cross_validation(
    X: np.ndarray,
    D: np.ndarray,
    k_candidates: Sequence[int] = (1, 2, 3, 4, 5),
    n_splits: int = 5,
    reg: float = 1e-2,
    standardize: bool = True,
    max_iter: int = 1500,
    random_state: int = 42,
    verbose: bool = True,
) -> CVResult:
    """
    Sélectionne le meilleur rang k par validation croisée sur les paires.

    Stratégie
    ---------
    On découpe les paires de la triangulaire supérieure en n_splits folds.
    Pour chaque k et chaque fold :
      - on ajuste L uniquement sur les paires d'entraînement
      - on évalue le MSE des distances sur les paires de validation

    Parameters
    ----------
    k_candidates : liste des rangs à tester
    n_splits     : nombre de folds
    """
    X = np.asarray(X, dtype=float)
    D = _validate_square_matrix(D, "D")
    n_states = X.shape[0]
    n_features = X.shape[1]

    i_all, j_all = upper_tri_pairs(n_states)
    n_pairs = len(i_all)

    if n_pairs < n_splits:
        raise ValueError(f"Pas assez de paires ({n_pairs}) pour {n_splits} folds.")

    # Filtrage des k valides
    k_candidates = sorted({int(k) for k in k_candidates if 1 <= int(k) <= n_features})
    if not k_candidates:
        raise ValueError("Aucun k valide dans k_candidates.")

    # Génération des folds (mélange des paires)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_pairs)
    fold_sizes = np.full(n_splits, n_pairs // n_splits, dtype=int)
    fold_sizes[: n_pairs % n_splits] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size

    scores = {k: [] for k in k_candidates}

    for fold_idx, val_idx in enumerate(folds):
        train_mask = np.ones(n_pairs, dtype=bool)
        train_mask[val_idx] = False
        val_mask = \~train_mask

        # Préparation des indices de validation
        i_val = i_all[val_mask]
        j_val = j_all[val_mask]
        target_val = D[i_val, j_val]

        for k in k_candidates:
            # Ajustement sur les paires d'entraînement uniquement
            L_hat, _ = fit_from_pairwise_dissimilarity(
                X,
                D,
                k=k,
                reg=reg,
                max_iter=max_iter,
                standardize=standardize,
                pair_mask=train_mask,
                return_details=False,
            )

            # Évaluation sur les paires de validation
            if standardize:
                mu = np.mean(X, axis=0)
                sd = np.maximum(np.std(X, axis=0, ddof=1), 1e-8)
                X_proc = (X - mu) / sd
            else:
                X_proc = X

            X_cent = X_proc - np.mean(X_proc, axis=0, keepdims=True)
            latent = X_cent @ L_hat
            delta = latent[i_val] - latent[j_val]
            pred = np.linalg.norm(delta, axis=1)
            mse = float(np.mean((pred - target_val) ** 2))
            scores[k].append(mse)

        if verbose:
            print(f"Fold {fold_idx + 1}/{n_splits} terminé.")

    # Agrégation
    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    std_scores = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in scores.items()}

    best_k = min(mean_scores, key=mean_scores.get)

    if verbose:
        print("\nRésultats de la validation croisée (MSE validation) :")
        for k in k_candidates:
            print(f"  k = {k:2d}  →  {mean_scores[k]:.6f}  ±  {std_scores[k]:.6f}")
        print(f"\nMeilleur k sélectionné : {best_k}")

    return CVResult(
        best_k=best_k,
        scores=mean_scores,
        std_scores=std_scores,
        all_fold_scores=scores,
    )


# ---------------------------------------------------------------------------
# Classe pratique
# ---------------------------------------------------------------------------

class SubjectiveMetric:
    """
    Apprentissage d'une métrique subjective de faible rang
    avec sélection automatique de k par validation croisée.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        reg: float = 1e-2,
        standardize: bool = True,
        k_candidates: Sequence[int] = (1, 2, 3, 4, 5),
        n_splits: int = 5,
    ):
        self.k = k
        self.reg = float(reg)
        self.standardize = bool(standardize)
        self.k_candidates = list(k_candidates)
        self.n_splits = int(n_splits)

        self.L_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.fit_result_: Optional[FitResult] = None
        self.cv_result_: Optional[CVResult] = None

    def select_k(
        self,
        X: np.ndarray,
        D: np.ndarray,
        k_candidates: Optional[Sequence[int]] = None,
        n_splits: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = True,
    ) -> int:
        """Lance la validation croisée et stocke le meilleur k."""
        candidates = k_candidates if k_candidates is not None else self.k_candidates
        splits = n_splits if n_splits is not None else self.n_splits

        self.cv_result_ = select_k_by_cross_validation(
            X,
            D,
            k_candidates=candidates,
            n_splits=splits,
            reg=self.reg,
            standardize=self.standardize,
            random_state=random_state,
            verbose=verbose,
        )
        self.k = self.cv_result_.best_k
        return self.k

    def fit(self, X: np.ndarray, D: np.ndarray, select_k: bool = False) -> "SubjectiveMetric":
        """
        Ajuste la métrique.

        Si select_k=True et que self.k est None, lance automatiquement
        la validation croisée avant l'ajustement final.
        """
        X = np.asarray(X, dtype=float)

        if select_k or self.k is None:
            if self.k is None:
                self.select_k(X, D, verbose=True)
            else:
                # on force quand même une CV si demandé
                self.select_k(X, D, verbose=True)

        if self.k is None:
            raise RuntimeError("k n'est pas défini. Appelez select_k() ou passez k à l'init.")

        # Standardisation
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
            standardize=False,
            return_details=True,
        )
        assert isinstance(result, FitResult)

        self.L_ = result.L
        self.W_ = result.W
        self.fit_result_ = result
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        return ((X - self.mu_) / self.sd_) @ self.L_

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        X_proc = (X - self.mu_) / self.sd_
        return predict_distances(X_proc, self.L_)

    def _check_is_fitted(self) -> None:
        if self.L_ is None:
            raise RuntimeError("L'estimateur n'est pas encore ajusté. Appelez fit().")


# ---------------------------------------------------------------------------
# Démonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(2026)

    n_states, n_features, k_true = 30, 8, 3
    X = rng.standard_normal((n_states, n_features))

    L_true = np.zeros((n_features, k_true))
    L_true[0, 0] = 1.4
    L_true[2, 0] = -0.9
    L_true[1, 1] = 1.1
    L_true[4, 1] = 0.7
    L_true[3, 2] = 1.0
    L_true[6, 2] = -0.6

    i_u, j_u = upper_tri_pairs(n_states)
    d_true = np.linalg.norm((X[i_u] - X[j_u]) @ L_true, axis=1)

    similarity = np.zeros((n_states, n_states))
    similarity[i_u, j_u] = similarity[j_u, i_u] = 7.0 - d_true
    np.fill_diagonal(similarity, 7.0)

    D = similarity_to_dissimilarity(similarity, mode="linear_max", min_similarity=1.0, max_similarity=7.0)

    print("=== Sélection de k par validation croisée ===")
    metric = SubjectiveMetric(
        k=None,                    # on laisse la CV choisir
        reg=1e-2,
        standardize=True,
        k_candidates=[1, 2, 3, 4, 5],
        n_splits=5,
    )
    metric.fit(X, D, select_k=True)

    print(f"\nRang sélectionné : {metric.k}")
    print(f"Succès optimisation finale : {metric.fit_result_.success}")
    print(f"Loss finale : {metric.fit_result_.final_loss:.6f}")
    print(f"Shape de L  : {metric.L_.shape}")
