#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apprentissage d'une métrique subjective de faible rang.

Deux protocoles sont disponibles :

- select_k_by_pair_cv :
  validation de nouvelles paires entre états connus ;

- select_k_by_state_cv :
  validation d'états entièrement tenus à l'écart.

Le modèle apprend une application :

    Z = (X - mu_train) / sd_train - center_train
    Z_latent = Z @ L

où L est de forme (n_features, k).

Le volume est calculé dans l'espace latent de dimension k.
Il ne faut pas utiliser det(L @ L.T) dans l'espace initial lorsque
k < n_features.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize


__all__ = [
    "upper_tri_pairs",
    "ball_volume_factor",
    "similarity_to_dissimilarity",
    "predict_distances",
    "latent_volume",
    "fit_from_pairwise_dissimilarity",
    "select_k_by_pair_cv",
    "select_k_by_state_cv",
    "SubjectiveMetric",
    "FitResult",
    "CVResult",
]


# ============================================================================
# VALIDATION ET UTILITAIRES
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
            f"{name} doit avoir {ndim} dimension(s), "
            f"reçu : {array.ndim}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contient des valeurs NaN ou infinies."
        )

    return array


def _validate_square_matrix(
    matrix: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Vérifie une matrice carrée et la symétrise explicitement.
    """
    array = _as_finite_array(
        matrix,
        name,
        ndim=2,
    )

    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} doit être carrée."
        )

    return 0.5 * (
        array
        + array.T
    )


def _validate_features(
    X: np.ndarray,
    name: str = "X",
) -> np.ndarray:
    """Vérifie une matrice d'états et de caractéristiques."""
    array = _as_finite_array(
        X,
        name,
        ndim=2,
    )

    if array.shape[0] < 2:
        raise ValueError(
            f"{name} doit contenir au moins deux états."
        )

    if array.shape[1] < 1:
        raise ValueError(
            f"{name} doit contenir au moins une caractéristique."
        )

    return array


def _validate_xy(
    X: np.ndarray,
    D: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vérifie la compatibilité entre X et D."""
    features = _validate_features(
        X,
        "X",
    )

    distances = _validate_square_matrix(
        D,
        "D",
    )

    if distances.shape != (
        features.shape[0],
        features.shape[0],
    ):
        raise ValueError(
            "D doit avoir une ligne et une colonne par état de X."
        )

    return features, distances


def _positive_integer(
    value: Any,
    name: str,
) -> int:
    """Valide un entier strictement positif."""
    if isinstance(
        value,
        (bool, np.bool_),
    ) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(
            f"{name} doit être un entier positif."
        )

    if value < 1:
        raise ValueError(
            f"{name} doit être supérieur ou égal à 1."
        )

    return int(value)


def _validate_pair_mask(
    pair_mask: Optional[np.ndarray],
    n_pairs: int,
) -> Optional[np.ndarray]:
    """Valide un masque de paires."""
    if pair_mask is None:
        return None

    mask = np.asarray(
        pair_mask,
        dtype=bool,
    )

    if mask.shape != (
        n_pairs,
    ):
        raise ValueError(
            f"pair_mask doit avoir la forme ({n_pairs},)."
        )

    if not np.any(mask):
        raise ValueError(
            "pair_mask ne sélectionne aucune paire."
        )

    return mask


def upper_tri_pairs(
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne les indices des paires de la triangulaire supérieure."""
    n = _positive_integer(
        n,
        "n",
    )

    if n < 2:
        raise ValueError(
            "n doit être supérieur ou égal à 2."
        )

    return np.triu_indices(
        n,
        k=1,
    )


def ball_volume_factor(
    k: int,
) -> float:
    """Volume de la boule unité euclidienne en dimension k."""
    k = _positive_integer(
        k,
        "k",
    )

    return float(
        math.pi ** (k / 2.0)
        / math.gamma(k / 2.0 + 1.0)
    )


# ============================================================================
# CONVERSION SIMILARITÉ → DISSIMILARITÉ
# ============================================================================

def similarity_to_dissimilarity(
    similarity_matrix: np.ndarray,
    mode: str = "linear_max",
    min_similarity: float | None = None,
    max_similarity: float | None = None,
) -> np.ndarray:
    """
    Convertit une matrice de similarité en dissimilarité non négative.

    Modes :

    - linear_max :
        D = max_similarity - S ;

    - linear_range :
        D = (max_similarity - S)
            / (max_similarity - min_similarity) ;

    - one_minus :
        D = 1 - S.
    """
    similarity = _validate_square_matrix(
        similarity_matrix,
        "similarity_matrix",
    )

    mode = str(
        mode
    ).lower()

    if mode == "one_minus":
        dissimilarity = 1.0 - similarity

    else:
        s_min = (
            float(np.min(similarity))
            if min_similarity is None
            else float(min_similarity)
        )

        s_max = (
            float(np.max(similarity))
            if max_similarity is None
            else float(max_similarity)
        )

        if not (
            np.isfinite(s_min)
            and np.isfinite(s_max)
        ):
            raise ValueError(
                "Les bornes de similarité doivent être finies."
            )

        if s_max <= s_min:
            raise ValueError(
                "max_similarity doit être supérieur à min_similarity."
            )

        if mode == "linear_max":
            dissimilarity = (
                s_max
                - similarity
            )

        elif mode == "linear_range":
            dissimilarity = (
                s_max
                - similarity
            ) / (
                s_max
                - s_min
            )

        else:
            raise ValueError(
                "Mode inconnu : "
                "'linear_max', 'linear_range' ou 'one_minus'."
            )

    dissimilarity = 0.5 * (
        dissimilarity
        + dissimilarity.T
    )

    dissimilarity = np.maximum(
        dissimilarity,
        0.0,
    )

    np.fill_diagonal(
        dissimilarity,
        0.0,
    )

    return dissimilarity


# ============================================================================
# PRÉTRAITEMENT
# ============================================================================

def _fit_preprocessing(
    X: np.ndarray,
    standardize: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Estime le prétraitement sur un ensemble d'entraînement.

    Retourne :

    - mu : moyenne brute d'entraînement ;
    - sd : écart-type d'entraînement ;
    - center : moyenne après mise à l'échelle ;
    - X_centered : données transformées et centrées.
    """
    features = _validate_features(
        X,
        "X",
    )

    mu = np.mean(
        features,
        axis=0,
    )

    if standardize:
        sd = np.std(
            features,
            axis=0,
            ddof=1,
        )
        sd = np.maximum(
            sd,
            1e-8,
        )
    else:
        sd = np.ones(
            features.shape[1],
            dtype=float,
        )

    X_scaled = (
        features
        - mu
    ) / sd

    center = np.mean(
        X_scaled,
        axis=0,
    )

    X_centered = (
        X_scaled
        - center
    )

    return (
        mu,
        sd,
        center,
        X_centered,
    )


def _apply_preprocessing(
    X: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """
    Applique un prétraitement déjà estimé.
    """
    features = _validate_features(
        X,
        "X",
    )

    mu = _as_finite_array(
        mu,
        "mu",
        ndim=1,
    )

    sd = _as_finite_array(
        sd,
        "sd",
        ndim=1,
    )

    center = _as_finite_array(
        center,
        "center",
        ndim=1,
    )

    if features.shape[1] != len(mu):
        raise ValueError(
            "Le nombre de caractéristiques de X et mu diffère."
        )

    if not (
        len(mu)
        == len(sd)
        == len(center)
    ):
        raise ValueError(
            "mu, sd et center doivent avoir la même longueur."
        )

    if np.any(sd <= 0):
        raise ValueError(
            "sd doit contenir uniquement des valeurs positives."
        )

    return (
        (
            features
            - mu
        ) / sd
        - center
    )


# ============================================================================
# DISTANCES ET VOLUME LATENT
# ============================================================================

def predict_distances(
    X_centered: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """
    Calcule la matrice complète des distances latentes.

    X_centered doit avoir été transformé avec le prétraitement
    d'entraînement.
    """
    X = _validate_features(
        X_centered,
        "X_centered",
    )

    projection = _as_finite_array(
        L,
        "L",
        ndim=2,
    )

    if X.shape[1] != projection.shape[0]:
        raise ValueError(
            "X_centered et L ont des dimensions incompatibles."
        )

    latent = X @ projection

    squared_norms = np.sum(
        latent ** 2,
        axis=1,
        keepdims=True,
    )

    squared_distances = (
        squared_norms
        + squared_norms.T
        - 2.0 * (
            latent
            @ latent.T
        )
    )

    squared_distances = np.maximum(
        squared_distances,
        0.0,
    )

    return np.sqrt(
        squared_distances
    )


def latent_volume(
    X_centered: np.ndarray,
    L: np.ndarray,
) -> float:
    """
    Estime une mesure de volume ellipsoïdal dans l'espace latent.

        Z = X_centered @ L

        V = B_k * sqrt(det(Cov(Z)))

    où B_k est le volume de la boule unité en dimension k.

    Cette quantité est une mesure de dispersion géométrique latente.
    Elle n'est pas automatiquement le volume d'une région physique.
    """
    X = _validate_features(
        X_centered,
        "X_centered",
    )

    projection = _as_finite_array(
        L,
        "L",
        ndim=2,
    )

    if X.shape[1] != projection.shape[0]:
        raise ValueError(
            "X_centered et L ont des dimensions incompatibles."
        )

    latent = X @ projection
    n_states, k = latent.shape

    if n_states <= k:
        raise ValueError(
            f"Il faut n_states > k ; reçu n_states={n_states}, k={k}."
        )

    if np.linalg.matrix_rank(
        latent
        - np.mean(
            latent,
            axis=0,
            keepdims=True,
        )
    ) < k:
        return 0.0

    covariance = np.cov(
        latent,
        rowvar=False,
        ddof=1,
    )

    covariance = np.atleast_2d(
        covariance
    )

    sign, logdet = np.linalg.slogdet(
        covariance
    )

    if sign <= 0 or not np.isfinite(logdet):
        return 0.0

    return float(
        ball_volume_factor(k)
        * math.exp(
            0.5 * logdet
        )
    )


# ============================================================================
# INITIALISATIONS
# ============================================================================

def _mds_initialization(
    D: np.ndarray,
    X_centered: np.ndarray,
    k: int,
    reg: float,
) -> np.ndarray:
    """
    Initialisation MDS classique.

    Cette fonction ne doit être appelée que lorsque D est complète
    et qu'aucune paire n'est tenue à l'écart.
    """
    distances = _validate_square_matrix(
        D,
        "D",
    )

    features = _validate_features(
        X_centered,
        "X_centered",
    )

    n_states = distances.shape[0]

    if features.shape[0] != n_states:
        raise ValueError(
            "D et X_centered doivent contenir le même nombre d'états."
        )

    k = int(k)

    if not (
        1 <= k <= features.shape[1]
    ):
        raise ValueError(
            "k est incompatible avec X_centered."
        )

    centering = (
        np.eye(n_states)
        - np.ones(
            (
                n_states,
                n_states,
            )
        ) / n_states
    )

    gram = -0.5 * (
        centering
        @ (
            distances ** 2
        )
        @ centering
    )

    gram = 0.5 * (
        gram
        + gram.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        gram
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    latent_initial = np.zeros(
        (
            n_states,
            k,
        ),
        dtype=float,
    )

    positive = eigenvalues > 1e-10

    if np.any(positive):
        selected = np.where(
            positive
        )[0][
            :k
        ]

        latent_initial[
            :,
            :len(selected)
        ] = (
            eigenvectors[
                :,
                selected
            ]
            * np.sqrt(
                np.maximum(
                    eigenvalues[
                        selected
                    ],
                    0.0,
                )
            )
        )

    regularized_gram = (
        features.T
        @ features
        + max(
            float(reg),
            1e-12,
        )
        * np.eye(
            features.shape[1]
        )
    )

    try:
        return np.linalg.solve(
            regularized_gram,
            features.T
            @ latent_initial,
        )

    except np.linalg.LinAlgError:
        return np.linalg.pinv(
            regularized_gram
        ) @ (
            features.T
            @ latent_initial
        )


def _random_initialization(
    n_features: int,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    Initialisation aléatoire indépendante des distances.
    """
    rng = np.random.default_rng(
        int(seed)
    )

    scale = 1.0 / math.sqrt(
        max(
            n_features,
            1,
        )
    )

    return rng.standard_normal(
        (
            n_features,
            k,
        )
    ) * scale


# ============================================================================
# RÉSULTATS
# ============================================================================

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
    protocol: str
    best_k: int
    scores: dict
    std_scores: dict
    all_fold_scores: dict


# ============================================================================
# AJUSTEMENT
# ============================================================================

def fit_from_pairwise_dissimilarity(
    X_centered: np.ndarray,
    D_train: np.ndarray,
    k: int,
    reg: float = 1e-2,
    max_iter: int = 2000,
    pair_mask: Optional[np.ndarray] = None,
    init: str = "mds",
    init_seed: int = 0,
    return_details: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    FitResult,
]:
    """
    Ajuste une métrique de faible rang.

    X_centered doit déjà être centré avec les paramètres d'entraînement.

    init="mds" est autorisé uniquement si aucune paire n'est exclue.
    En validation croisée, utiliser init="random".
    """
    X = _validate_features(
        X_centered,
        "X_centered",
    )

    D = _validate_square_matrix(
        D_train,
        "D_train",
    )

    n_states, n_features = X.shape

    if D.shape != (
        n_states,
        n_states,
    ):
        raise ValueError(
            "D_train doit correspondre au nombre d'états de X_centered."
        )

    k = int(k)

    if not (
        1 <= k <= n_features
    ):
        raise ValueError(
            f"k doit être compris entre 1 et {n_features}."
        )

    reg = float(reg)

    if not np.isfinite(reg) or reg < 0:
        raise ValueError(
            "reg doit être fini et non négatif."
        )

    max_iter = _positive_integer(
        max_iter,
        "max_iter",
    )

    D = np.maximum(
        D,
        0.0,
    )

    np.fill_diagonal(
        D,
        0.0,
    )

    i_all, j_all = upper_tri_pairs(
        n_states
    )

    pair_mask = _validate_pair_mask(
        pair_mask,
        len(i_all),
    )

    if pair_mask is None:
        i_upper = i_all
        j_upper = j_all
        all_pairs_used = True
    else:
        i_upper = i_all[
            pair_mask
        ]
        j_upper = j_all[
            pair_mask
        ]
        all_pairs_used = bool(
            np.all(pair_mask)
        )

    target = D[
        i_upper,
        j_upper
    ]

    if len(target) == 0:
        raise ValueError(
            "Aucune paire disponible pour l'ajustement."
        )

    init = str(
        init
    ).lower()

    if init == "mds":
        if not all_pairs_used:
            raise ValueError(
                "init='mds' est interdit lorsque des paires sont exclues. "
                "Utiliser init='random' pour éviter la fuite."
            )

        L0 = _mds_initialization(
            D,
            X,
            k,
            max(
                reg,
                1e-12,
            ),
        )

    elif init == "random":
        L0 = _random_initialization(
            n_features,
            k,
            init_seed,
        )

    else:
        raise ValueError(
            "init doit être 'mds' ou 'random'."
        )

    def unpack(
        theta: np.ndarray,
    ) -> np.ndarray:
        return theta.reshape(
            n_features,
            k,
        )

    def objective(
        theta: np.ndarray,
    ) -> float:
        L = unpack(
            theta
        )

        latent = X @ L

        delta = (
            latent[
                i_upper
            ]
            - latent[
                j_upper
            ]
        )

        predicted = np.linalg.norm(
            delta,
            axis=1,
        )

        data_loss = np.mean(
            (
                predicted
                - target
            ) ** 2
        )

        penalty = reg * np.sum(
            L ** 2
        )

        return float(
            data_loss
            + penalty
        )

    result = minimize(
        objective,
        x0=L0.ravel(),
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    L_hat = unpack(
        result.x
    )

    W_hat = L_hat @ L_hat.T

    latent = X @ L_hat

    delta = (
        latent[
            i_upper
        ]
        - latent[
            j_upper
        ]
    )

    predicted = np.linalg.norm(
        delta,
        axis=1,
    )

    data_loss = float(
        np.mean(
            (
                predicted
                - target
            ) ** 2
        )
    )

    penalty = float(
        reg
        * np.sum(
            L_hat ** 2
        )
    )

    final_loss = float(
        data_loss
        + penalty
    )

    if not result.success:
        warnings.warn(
            "Optimisation non convergée : "
            f"{result.message}",
            RuntimeWarning,
        )

    if return_details:
        return FitResult(
            L=L_hat,
            W=W_hat,
            success=bool(
                result.success
            ),
            message=str(
                result.message
            ),
            n_iter=int(
                result.nit
            ),
            final_loss=final_loss,
            data_loss=data_loss,
            penalty=penalty,
        )

    return L_hat, W_hat


# ============================================================================
# VALIDATION CROISÉE PAR PAIRES
# ============================================================================

def select_k_by_pair_cv(
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
    Sélectionne k par validation croisée de paires.

    Les mêmes états apparaissent en entraînement et en validation.
    Le protocole mesure l'interpolation de nouvelles paires entre états
    connus, et non la généralisation à de nouveaux états.
    """
    features, distances = _validate_xy(
        X,
        D,
    )

    n_states, n_features = features.shape

    n_splits = _positive_integer(
        n_splits,
        "n_splits",
    )

    i_all, j_all = upper_tri_pairs(
        n_states
    )

    n_pairs = len(i_all)

    if n_splits > n_pairs:
        raise ValueError(
            "Le nombre de folds dépasse le nombre de paires."
        )

    candidates = sorted(
        {
            int(k)
            for k in k_candidates
            if 1 <= int(k) <= n_features
        }
    )

    if not candidates:
        raise ValueError(
            "Aucun rang k valide."
        )

    _, _, _, X_centered = _fit_preprocessing(
        features,
        standardize,
    )

    rng = np.random.default_rng(
        random_state
    )

    shuffled_pairs = rng.permutation(
        n_pairs
    )

    fold_sizes = np.full(
        n_splits,
        n_pairs // n_splits,
        dtype=int,
    )

    fold_sizes[
        : n_pairs % n_splits
    ] += 1

    folds = []
    start = 0

    for size in fold_sizes:
        folds.append(
            shuffled_pairs[
                start : start + size
            ]
        )
        start += size

    scores = {
        k: []
        for k in candidates
    }

    for fold_index, validation_indices in enumerate(
        folds
    ):
        train_mask = np.ones(
            n_pairs,
            dtype=bool,
        )

        train_mask[
            validation_indices
        ] = False

        validation_mask = ~train_mask

        validation_i = i_all[
            validation_mask
        ]

        validation_j = j_all[
            validation_mask
        ]

        validation_target = distances[
            validation_i,
            validation_j
        ]

        for k in candidates:
            L_hat, _ = fit_from_pairwise_dissimilarity(
                X_centered,
                distances,
                k=k,
                reg=reg,
                max_iter=max_iter,
                pair_mask=train_mask,
                init="random",
                init_seed=random_state
                + 10_000 * fold_index
                + k,
            )

            latent = X_centered @ L_hat

            delta = (
                latent[
                    validation_i
                ]
                - latent[
                    validation_j
                ]
            )

            predicted = np.linalg.norm(
                delta,
                axis=1,
            )

            mse = float(
                np.mean(
                    (
                        predicted
                        - validation_target
                    ) ** 2
                )
            )

            scores[k].append(
                mse
            )

        if verbose:
            print(
                f"[pair-cv] fold "
                f"{fold_index + 1}/{n_splits} terminé."
            )

    mean_scores = {
        k: float(
            np.mean(
                values
            )
        )
        for k, values in scores.items()
    }

    std_scores = {
        k: float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0
        for k, values in scores.items()
    }

    best_k = min(
        mean_scores,
        key=mean_scores.get,
    )

    if verbose:
        print(
            "\n[pair-cv] MSE de validation "
            "entre états connus :"
        )

        for k in candidates:
            print(
                f"  k={k}: "
                f"{mean_scores[k]:.6f} "
                f"+/- {std_scores[k]:.6f}"
            )

        print(
            f"[pair-cv] meilleur k : {best_k}"
        )

    return CVResult(
        protocol="pair",
        best_k=best_k,
        scores=mean_scores,
        std_scores=std_scores,
        all_fold_scores=scores,
    )


# ============================================================================
# VALIDATION CROISÉE PAR ÉTATS
# ============================================================================

def select_k_by_state_cv(
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
    Sélectionne k par validation d'états tenus à l'écart.

    Les paramètres de prétraitement sont estimés uniquement sur les états
    d'entraînement de chaque fold, puis appliqués aux états de validation.
    """
    features, distances = _validate_xy(
        X,
        D,
    )

    n_states, n_features = features.shape

    n_splits = _positive_integer(
        n_splits,
        "n_splits",
    )

    if n_splits > n_states:
        raise ValueError(
            "Le nombre de folds dépasse le nombre d'états."
        )

    if n_states - math.ceil(
        n_states / n_splits
    ) < 2:
        raise ValueError(
            "Chaque fold doit laisser au moins deux états d'entraînement."
        )

    candidates = sorted(
        {
            int(k)
            for k in k_candidates
            if 1 <= int(k) <= n_features
        }
    )

    if not candidates:
        raise ValueError(
            "Aucun rang k valide."
        )

    rng = np.random.default_rng(
        random_state
    )

    shuffled_states = rng.permutation(
        n_states
    )

    fold_sizes = np.full(
        n_splits,
        n_states // n_splits,
        dtype=int,
    )

    fold_sizes[
        : n_states % n_splits
    ] += 1

    folds = []
    start = 0

    for size in fold_sizes:
        folds.append(
            np.sort(
                shuffled_states[
                    start : start + size
                ]
            )
        )
        start += size

    i_all, j_all = upper_tri_pairs(
        n_states
    )

    scores = {
        k: []
        for k in candidates
    }

    for fold_index, validation_states in enumerate(
        folds
    ):
        validation_state_mask = np.zeros(
            n_states,
            dtype=bool,
        )

        validation_state_mask[
            validation_states
        ] = True

        training_states = np.where(
            ~validation_state_mask
        )[0]

        if len(training_states) <= max(
            candidates
        ):
            raise ValueError(
                "Trop peu d'états d'entraînement "
                "pour le rang maximal demandé."
            )

        X_training = features[
            training_states
        ]

        X_validation = features[
            validation_states
        ]

        mu, sd, center, X_training_centered = (
            _fit_preprocessing(
                X_training,
                standardize,
            )
        )

        X_all_centered = _apply_preprocessing(
            features,
            mu,
            sd,
            center,
        )

        D_training = distances[
            np.ix_(
                training_states,
                training_states,
            )
        ]

        # Toutes les paires impliquant au moins un état de validation.
        validation_pair_mask = (
            validation_state_mask[
                i_all
            ]
            | validation_state_mask[
                j_all
            ]
        )

        validation_i = i_all[
            validation_pair_mask
        ]

        validation_j = j_all[
            validation_pair_mask
        ]

        validation_target = distances[
            validation_i,
            validation_j
        ]

        for k in candidates:
            L_hat, _ = fit_from_pairwise_dissimilarity(
                X_training_centered,
                D_training,
                k=k,
                reg=reg,
                max_iter=max_iter,
                pair_mask=None,
                init="random",
                init_seed=random_state
                + 10_000 * fold_index
                + k,
            )

            latent_all = X_all_centered @ L_hat

            delta = (
                latent_all[
                    validation_i
                ]
                - latent_all[
                    validation_j
                ]
            )

            predicted = np.linalg.norm(
                delta,
                axis=1,
            )

            mse = float(
                np.mean(
                    (
                        predicted
                        - validation_target
                    ) ** 2
                )
            )

            scores[k].append(
                mse
            )

        if verbose:
            print(
                f"[state-cv] fold "
                f"{fold_index + 1}/{n_splits} terminé ; "
                f"{len(validation_states)} état(s) tenu(s) à l'écart."
            )

    mean_scores = {
        k: float(
            np.mean(
                values
            )
        )
        for k, values in scores.items()
    }

    std_scores = {
        k: float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0
        for k, values in scores.items()
    }

    best_k = min(
        mean_scores,
        key=mean_scores.get,
    )

    if verbose:
        print(
            "\n[state-cv] MSE de validation "
            "sur états tenus à l'écart :"
        )

        for k in candidates:
            print(
                f"  k={k}: "
                f"{mean_scores[k]:.6f} "
                f"+/- {std_scores[k]:.6f}"
            )

        print(
            f"[state-cv] meilleur k : {best_k}"
        )

    return CVResult(
        protocol="state",
        best_k=best_k,
        scores=mean_scores,
        std_scores=std_scores,
        all_fold_scores=scores,
    )


# ============================================================================
# CLASSE PRATIQUE
# ============================================================================

class SubjectiveMetric:
    """
    Métrique subjective de faible rang.

    Le prétraitement d'entraînement est conservé dans :

    - mu_ ;
    - sd_ ;
    - center_.

    Il est réutilisé sans recentrage dépendant du lot lors de la projection
    de nouveaux états.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        reg: float = 1e-2,
        standardize: bool = True,
        k_candidates: Sequence[int] = (1, 2, 3, 4, 5),
        n_splits: int = 5,
        cv_protocol: str = "pair",
    ):
        self.k = k
        self.reg = float(reg)
        self.standardize = bool(
            standardize
        )
        self.k_candidates = list(
            k_candidates
        )
        self.n_splits = int(
            n_splits
        )

        self.cv_protocol = str(
            cv_protocol
        ).lower()

        if self.cv_protocol not in (
            "pair",
            "state",
        ):
            raise ValueError(
                "cv_protocol doit être 'pair' ou 'state'."
            )

        self.L_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.center_: Optional[np.ndarray] = None
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
        """Sélectionne k selon le protocole choisi."""
        candidates = (
            self.k_candidates
            if k_candidates is None
            else k_candidates
        )

        splits = (
            self.n_splits
            if n_splits is None
            else n_splits
        )

        if self.cv_protocol == "pair":
            selector = select_k_by_pair_cv
        else:
            selector = select_k_by_state_cv

        self.cv_result_ = selector(
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

    def fit(
        self,
        X: np.ndarray,
        D: np.ndarray,
        select_k: bool = False,
    ) -> "SubjectiveMetric":
        """
        Ajuste la métrique.

        Si select_k=True ou si k=None, la sélection croisée est exécutée.
        """
        features, distances = _validate_xy(
            X,
            D,
        )

        if select_k or self.k is None:
            self.select_k(
                features,
                distances,
                verbose=True,
            )

        if self.k is None:
            raise RuntimeError(
                "k n'est pas défini."
            )

        mu, sd, center, X_centered = (
            _fit_preprocessing(
                features,
                self.standardize,
            )
        )

        self.mu_ = mu
        self.sd_ = sd
        self.center_ = center

        result = fit_from_pairwise_dissimilarity(
            X_centered,
            distances,
            k=self.k,
            reg=self.reg,
            pair_mask=None,
            init="mds",
            return_details=True,
        )

        if not isinstance(
            result,
            FitResult,
        ):
            raise RuntimeError(
                "Résultat d'ajustement inattendu."
            )

        self.L_ = result.L
        self.W_ = result.W
        self.fit_result_ = result

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Projette des états dans le référentiel latent entraîné."""
        self._check_is_fitted()

        if self.mu_ is None:
            raise RuntimeError(
                "Les paramètres de centrage sont absents."
            )

        return _apply_preprocessing(
            X,
            self.mu_,
            self.sd_,
            self.center_,
        ) @ self.L_

    def pairwise_distances(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Calcule les distances latentes dans le référentiel d'entraînement."""
        self._check_is_fitted()

        X_centered = _apply_preprocessing(
            X,
            self.mu_,
            self.sd_,
            self.center_,
        )

        return predict_distances(
            X_centered,
            self.L_,
        )

    def latent_volume(
        self,
        X: np.ndarray,
    ) -> float:
        """Calcule la dispersion volumique latente dans R^k."""
        self._check_is_fitted()

        X_centered = _apply_preprocessing(
            X,
            self.mu_,
            self.sd_,
            self.center_,
        )

        return latent_volume(
            X_centered,
            self.L_,
        )

    def _check_is_fitted(
        self,
    ) -> None:
        if (
            self.L_ is None
            or self.mu_ is None
            or self.sd_ is None
            or self.center_ is None
        ):
            raise RuntimeError(
                "Estimateur non ajusté. Appelez fit()."
            )


# ============================================================================
# DÉMONSTRATION
# ============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(
        2026
    )

    n_states = 30
    n_features = 8
    k_true = 3

    X = rng.standard_normal(
        (
            n_states,
            n_features,
        )
    )

    L_true = np.zeros(
        (
            n_features,
            k_true,
        )
    )

    L_true[0, 0] = 1.4
    L_true[2, 0] = -0.9
    L_true[1, 1] = 1.1
    L_true[4, 1] = 0.7
    L_true[3, 2] = 1.0
    L_true[6, 2] = -0.6

    i_upper, j_upper = upper_tri_pairs(
        n_states
    )

    true_distances = np.linalg.norm(
        (
            X[i_upper]
            - X[j_upper]
        ) @ L_true,
        axis=1,
    )

    similarity = np.zeros(
        (
            n_states,
            n_states,
        )
    )

    similarity[
        i_upper,
        j_upper
    ] = 7.0 - true_distances

    similarity[
        j_upper,
        i_upper
    ] = 7.0 - true_distances

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

    print(
        "=== Validation croisée par paires ==="
    )

    pair_metric = SubjectiveMetric(
        k=None,
        reg=1e-2,
        standardize=True,
        k_candidates=(1, 2, 3, 4, 5),
        n_splits=5,
        cv_protocol="pair",
    )

    pair_metric.fit(
        X,
        D,
        select_k=True,
    )

    print(
        f"Rang sélectionné : {pair_metric.k}"
    )

    print(
        f"Succès optimisation : "
        f"{pair_metric.fit_result_.success}"
    )

    print(
        f"Volume latent : "
        f"{pair_metric.latent_volume(X):.6f}"
    )

    print()

    print(
        "=== Validation croisée par états ==="
    )

    state_metric = SubjectiveMetric(
        k=None,
        reg=1e-2,
        standardize=True,
        k_candidates=(1, 2, 3, 4, 5),
        n_splits=5,
        cv_protocol="state",
    )

    state_metric.fit(
        X,
        D,
        select_k=True,
    )

    print(
        f"Rang sélectionné : {state_metric.k}"
    )

    print(
        f"Succès optimisation : "
        f"{state_metric.fit_result_.success}"
    )

    print(
        f"Volume latent : "
        f"{state_metric.latent_volume(X):.6f}"
)
