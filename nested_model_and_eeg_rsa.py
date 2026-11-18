#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from scipy.stats import spearmanr


def fit_frozen_nested_models(C_train, V_psi_train, Q_train, Y_train):
    """
    Ajuste uniquement sur l'échantillon d'entraînement :
      1. Les scalers (mu, sd) de C, V_psi et Q.
      2. La matrice de projection résidualisant Q par rapport à C
         dans le pli d'entraînement uniquement.
      3. Les poids figés beta_M0, beta_M1 et beta_M2
         (Y de forme (n_train, n_windows)).
    """
    C_tr = np.asarray(C_train, dtype=float)
    v_tr = np.asarray(V_psi_train, dtype=float).reshape(-1, 1)
    Q_tr = np.asarray(Q_train, dtype=float)
    Y_tr = np.asarray(Y_train, dtype=float)

    if Y_tr.ndim == 1:
        Y_tr = Y_tr[:, None]

    mu_C = np.mean(
        C_tr,
        axis=0,
        keepdims=True,
    )

    sd_C = np.maximum(
        np.std(
            C_tr,
            axis=0,
            ddof=1,
            keepdims=True,
        ),
        1e-8,
    )

    C_std = (
        C_tr - mu_C
    ) / sd_C

    mu_V = float(
        np.mean(v_tr)
    )

    sd_V = max(
        float(
            np.std(
                v_tr,
                ddof=1,
            )
        ),
        1e-8,
    )

    v_std = (
        v_tr - mu_V
    ) / sd_V

    mu_Q = np.mean(
        Q_tr,
        axis=0,
        keepdims=True,
    )

    sd_Q = np.maximum(
        np.std(
            Q_tr,
            axis=0,
            ddof=1,
            keepdims=True,
        ),
        1e-8,
    )

    Q_std = (
        Q_tr - mu_Q
    ) / sd_Q

    # Résidualisation de Q par rapport à [1, C_std]
    # dans le pli d'entraînement uniquement.
    X_cov_int = np.hstack(
        [
            np.ones(
                (
                    len(C_std),
                    1,
                )
            ),
            C_std,
        ]
    )

    W_q_on_c, _, _, _ = np.linalg.lstsq(
        X_cov_int,
        Q_std,
        rcond=None,
    )

    Q_res = (
        Q_std
        - X_cov_int @ W_q_on_c
    )

    X_M0 = X_cov_int

    X_M1 = np.hstack(
        [
            X_M0,
            v_std,
        ]
    )

    X_M2 = np.hstack(
        [
            X_M1,
            Q_res,
        ]
    )

    beta_M0, _, _, _ = np.linalg.lstsq(
        X_M0,
        Y_tr,
        rcond=None,
    )

    beta_M1, _, _, _ = np.linalg.lstsq(
        X_M1,
        Y_tr,
        rcond=None,
    )

    beta_M2, _, _, _ = np.linalg.lstsq(
        X_M2,
        Y_tr,
        rcond=None,
    )

    return {
        "mu_C": mu_C,
        "sd_C": sd_C,
        "mu_V": mu_V,
        "sd_V": sd_V,
        "mu_Q": mu_Q,
        "sd_Q": sd_Q,
        "W_q_on_c": W_q_on_c,
        "beta_M0": beta_M0,
        "beta_M1": beta_M1,
        "beta_M2": beta_M2,
    }


def _transform_test_design(
    C_test,
    V_psi_test,
    Q_test,
    frozen_fit,
):
    C_te = (
        np.asarray(
            C_test,
            dtype=float,
        )
        - frozen_fit["mu_C"]
    ) / frozen_fit["sd_C"]

    v_te = (
        np.asarray(
            V_psi_test,
            dtype=float,
        ).reshape(-1, 1)
        - frozen_fit["mu_V"]
    ) / frozen_fit["sd_V"]

    Q_te = (
        np.asarray(
            Q_test,
            dtype=float,
        )
        - frozen_fit["mu_Q"]
    ) / frozen_fit["sd_Q"]

    X_cov_int = np.hstack(
        [
            np.ones(
                (
                    len(C_te),
                    1,
                )
            ),
            C_te,
        ]
    )

    Q_res_te = (
        Q_te
        - X_cov_int @ frozen_fit["W_q_on_c"]
    )

    X_M0 = X_cov_int

    X_M1 = np.hstack(
        [
            X_M0,
            v_te,
        ]
    )

    X_M2 = np.hstack(
        [
            X_M1,
            Q_res_te,
        ]
    )

    return X_M0, X_M1, X_M2


def evaluate_nested_oos_with_max_t(
    C_test,
    V_psi_test,
    Q_test,
    Y_test,
    frozen_fit,
    pair_ids,
    n_perm=499,
    seed=42,
):
    """
    Évalue hors échantillon sans réestimation :

      - Delta_S_V = S_test(M1) - S_test(M0)
      - Delta_S_q = S_test(M2) - S_test(M1)

    sur une ou plusieurs fenêtres temporelles Y_test
    (ex. RDM EEG/MEG à t_1..t_T), avec p-value corrigée
    max-T par échange intra-paire sur pair_ids.
    """
    Y_te = np.asarray(
        Y_test,
        dtype=float,
    )

    if Y_te.ndim == 1:
        Y_te = Y_te[:, None]

    n_obs, n_win = Y_te.shape

    X_M0, X_M1, X_M2 = _transform_test_design(
        C_test,
        V_psi_test,
        Q_test,
        frozen_fit,
    )

    pred_0 = (
        X_M0
        @ frozen_fit["beta_M0"]
    )

    pred_1 = (
        X_M1
        @ frozen_fit["beta_M1"]
    )

    pred_2 = (
        X_M2
        @ frozen_fit["beta_M2"]
    )

    mse_0 = np.mean(
        (
            Y_te - pred_0
        ) ** 2,
        axis=0,
    )

    mse_1 = np.mean(
        (
            Y_te - pred_1
        ) ** 2,
        axis=0,
    )

    mse_2 = np.mean(
        (
            Y_te - pred_2
        ) ** 2,
        axis=0,
    )

    # Score S = -MSE :
    # Delta_S > 0 signifie une réduction d'erreur hors échantillon.
    delta_S_V_obs = (
        mse_0 - mse_1
    )

    delta_S_q_obs = (
        mse_1 - mse_2
    )

    p_arr = np.asarray(
        pair_ids
    )

    if len(p_arr) != n_obs:
        raise ValueError(
            "`pair_ids` doit avoir la même longueur que `Y_test`."
        )

    pair_indices = []

    for pid in np.unique(p_arr):
        idx = np.where(
            p_arr == pid
        )[0]

        if len(idx) != 2:
            raise ValueError(
                f"La paire {pid!r} doit contenir exactement "
                "2 observations échangeables."
            )

        pair_indices.append(
            idx
        )

    rng = np.random.default_rng(
        seed
    )

    null_max_V = np.empty(
        int(n_perm),
        dtype=float,
    )

    null_max_q = np.empty(
        int(n_perm),
        dtype=float,
    )

    v_col = X_M1[:, [-1]]

    q_cols = X_M2[
        :,
        X_M1.shape[1]:,
    ]

    for p_idx in range(
        int(n_perm)
    ):
        v_perm = v_col.copy()
        q_perm = q_cols.copy()

        for idx_pair in pair_indices:
            if rng.integers(
                0,
                2,
            ) == 1:
                v_perm[
                    idx_pair[0]
                ], v_perm[
                    idx_pair[1]
                ] = (
                    v_perm[
                        idx_pair[1]
                    ].copy(),
                    v_perm[
                        idx_pair[0]
                    ].copy(),
                )

                q_perm[
                    idx_pair[0]
                ], q_perm[
                    idx_pair[1]
                ] = (
                    q_perm[
                        idx_pair[1]
                    ].copy(),
                    q_perm[
                        idx_pair[0]
                    ].copy(),
                )

        X_M1_p = np.hstack(
            [
                X_M0,
                v_perm,
            ]
        )

        X_M2_p = np.hstack(
            [
                X_M1,
                q_perm,
            ]
        )

        mse_1_p = np.mean(
            (
                Y_te
                - X_M1_p
                @ frozen_fit["beta_M1"]
            ) ** 2,
            axis=0,
        )

        mse_2_p = np.mean(
            (
                Y_te
                - X_M2_p
                @ frozen_fit["beta_M2"]
            ) ** 2,
            axis=0,
        )

        null_max_V[p_idx] = float(
            np.max(
                mse_0 - mse_1_p
            )
        )

        null_max_q[p_idx] = float(
            np.max(
                mse_1 - mse_2_p
            )
        )

    p_max_t_V = [
        round(
            (
                1.0
                + float(
                    np.sum(
                        null_max_V
                        >= obs_val
                    )
                )
            )
            / (
                int(n_perm)
                + 1.0
            ),
            4,
        )
        for obs_val in delta_S_V_obs
    ]

    p_max_t_q = [
        round(
            (
                1.0
                + float(
                    np.sum(
                        null_max_q
                        >= obs_val
                    )
                )
            )
            / (
                int(n_perm)
                + 1.0
            ),
            4,
        )
        for obs_val in delta_S_q_obs
    ]

    # Corrélation RSA temporelle :
    # Spearman entre Y_test et prédictions figées par fenêtre.
    rsa_rho_M1 = [
        round(
            float(
                spearmanr(
                    Y_te[:, w],
                    pred_1[:, w],
                ).statistic
            ),
            4,
        )
        for w in range(n_win)
    ]

    rsa_rho_M2 = [
        round(
            float(
                spearmanr(
                    Y_te[:, w],
                    pred_2[:, w],
                ).statistic
            ),
            4,
        )
        for w in range(n_win)
    ]

    return {
        "n_windows": int(n_win),
        "delta_S_V_M1_minus_M0": [
            round(
                float(x),
                4,
            )
            for x in delta_S_V_obs
        ],
        "p_value_maxT_V": p_max_t_V,
        "delta_S_q_M2_minus_M1": [
            round(
                float(x),
                4,
            )
            for x in delta_S_q_obs
        ],
        "p_value_maxT_q": p_max_t_q,
        "rsa_spearman_M1": rsa_rho_M1,
        "rsa_spearman_M2": rsa_rho_M2,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(
        2026
    )

    n_pairs_tr = 30
    n_pairs_te = 30
    n_win = 3

    def sim_split(n_p):
        C = []
        V = []
        Q = []
        Y = []
        pids = []

        for p in range(n_p):
            c_base = rng.standard_normal(
                2
            )

            for cond in [0, 1]:
                v_val = (
                    0.8 * cond
                    + 0.3
                    * rng.standard_normal()
                )

                q_val = rng.standard_normal(
                    3
                )

                # Fenêtre 1, par exemple 200–300 ms,
                # porte un signal lié à V_psi et q[0].
                # Les fenêtres 0 et 2 sont nulles.

                y_w0 = (
                    0.6 * c_base[0]
                    - 0.4 * c_base[1]
                    + 0.5
                    * rng.standard_normal()
                )

                y_w1 = (
                    0.6 * c_base[0]
                    - 0.4 * c_base[1]
                    + 0.55 * v_val
                    + 0.45 * q_val[0]
                    + 0.45
                    * rng.standard_normal()
                )

                y_w2 = (
                    0.6 * c_base[0]
                    - 0.4 * c_base[1]
                    + 0.5
                    * rng.standard_normal()
                )

                C.append(
                    c_base
                )

                V.append(
                    v_val
                )

                Q.append(
                    q_val
                )

                Y.append(
                    [
                        y_w0,
                        y_w1,
                        y_w2,
                    ]
                )

                pids.append(
                    p
                )

        return (
            np.asarray(C),
            np.asarray(V),
            np.asarray(Q),
            np.asarray(Y),
            np.asarray(pids),
        )

    C_tr, V_tr, Q_tr, Y_tr, _ = sim_split(
        n_pairs_tr
    )

    C_te, V_te, Q_te, Y_te, pids_te = sim_split(
        n_pairs_te
    )

    frozen = fit_frozen_nested_models(
        C_tr,
        V_tr,
        Q_tr,
        Y_tr,
    )

    oos_res = evaluate_nested_oos_with_max_t(
        C_te,
        V_te,
        Q_te,
        Y_te,
        frozen,
        pids_te,
        n_perm=299,
        seed=2026,
    )

    print(
        json.dumps(
            oos_res,
            indent=2,
            ensure_ascii=False,
        )
    )
