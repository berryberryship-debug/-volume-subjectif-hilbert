#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. Géométrie log-euclidienne pour matrices symétriques définies positives
# ---------------------------------------------------------------------------

def spd_logm(G, floor=1e-12):
    """Logarithme matriciel exact d'une matrice SPD via décomposition spectrale."""
    S = 0.5 * (np.asarray(G, dtype=float) + np.asarray(G, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, float(floor))
    return (eigvecs * np.log(eigvals)[None, :]) @ eigvecs.T


def spd_expm(L):
    """Exponentielle matricielle d'une matrice symétrique (garantit G SPD)."""
    S = 0.5 * (np.asarray(L, dtype=float) + np.asarray(L, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(S)
    return (eigvecs * np.exp(eigvals)[None, :]) @ eigvecs.T


def log_euclidean_distance(G1, G2):
    """Distance log-euclidienne ||log(G1) - log(G2)||_F."""
    diff = spd_logm(G1) - spd_logm(G2)
    return float(np.linalg.norm(diff, ord="fro"))


def ball_volume_factor(k):
    return (math.pi ** (0.5 * k)) / math.gamma(0.5 * k + 1.0)


def wilson_ci_95(k_success, n_total):
    """Intervalle de confiance de Wilson à 95 % pour une proportion."""
    n = max(int(n_total), 1)
    p = float(k_success) / n
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p)) / n + z2 / (4.0 * n * n))) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


# ---------------------------------------------------------------------------
# 2. Rééchantillonnage par grappes et calcul de l'ICC interséance
# ---------------------------------------------------------------------------

def _cluster_bootstrap_indices(cluster_ids, rng):
    c_arr = np.asarray(cluster_ids)
    unique_c = np.unique(c_arr)
    sampled_c = rng.choice(unique_c, size=len(unique_c), replace=True)
    idx_list = [np.where(c_arr == c)[0] for c in sampled_c]
    return np.concatenate(idx_list)


def compute_session_icc_from_logs(logs_psgm):
    """
    Calcule ICC_seance = sigma^2_participant / (sigma^2_participant + sigma^2_seance + sigma^2_erreur)
    à partir du tenseur `logs_psgm` de forme (n_participants, n_sessions, n_grid, n_sym_entries)
    selon la décomposition Y_{isgm} = mu_{gm} + u_{igm} + v_{isgm} + e_{isgm}.
    """
    arr = np.asarray(logs_psgm, dtype=float)
    n_p, n_s, n_g, _ = arr.shape
    if n_p < 2:
        return 0.0
    if n_s < 2:
        mean_by_grid = np.mean(arr[:, 0], axis=0)
        vb = float(np.mean(np.var(mean_by_grid, axis=0, ddof=1))) if n_g > 1 else 0.0
        vw = float(np.mean(np.var(arr[:, 0], axis=0, ddof=1)))
        return vb / max(vb + vw, 1e-12)

    grand_mean = np.mean(arr, axis=(0, 1), keepdims=True)
    part_mean = np.mean(arr, axis=1, keepdims=True)

    ms_part = (n_s / max(n_p - 1, 1)) * np.sum((part_mean - grand_mean) ** 2, axis=(0, 1))
    ms_within = np.sum((arr - part_mean) ** 2, axis=(0, 1)) / max(n_p * (n_s - 1), 1)

    var_part = np.maximum((ms_part - ms_within) / n_s, 0.0)
    var_sess_err = np.maximum(ms_within, 0.0)

    num = float(np.sum(var_part))
    den = float(np.sum(var_part + var_sess_err))
    return num / max(den, 1e-12)


# ---------------------------------------------------------------------------
# 3. Les trois estimands indépendants et le critère conjonctif
# ---------------------------------------------------------------------------

def evaluate_jnd_equivalence(jnd_pred, jnd_obs, cluster_ids, delta_jnd, n_boot=400, seed=42):
    """
    Volet A : Prédiction des seuils JND indépendants.
    Critère préenregistré : CI_95%(Bias_JND) inclus dans [-delta_jnd, +delta_jnd].
    """
    pred = np.asarray(jnd_pred, dtype=float)
    obs = np.asarray(jnd_obs, dtype=float)
    err = pred - obs

    bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))

    rng = np.random.default_rng(seed)
    boot_biases = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = _cluster_bootstrap_indices(cluster_ids, rng)
        boot_biases[b] = float(np.mean(err[idx]))

    ci_lo, ci_hi = np.quantile(boot_biases, [0.025, 0.975])
    passed = bool((ci_lo >= -float(delta_jnd)) and (ci_hi <= float(delta_jnd)))

    return {
        "I_J": int(passed),
        "bootstrap_equivalence_pass": passed,
        "bias_jnd": round(bias, 4),
        "mae_jnd": round(mae, 4),
        "ci95_bias": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "delta_jnd_margin": float(delta_jnd),
    }


def evaluate_field_stability(
    G_disc_grid,
    G_rep_by_participant_session,
    delta_G_margin,
    rho_G,
    icc_criterion="global",
    n_boot=400,
    seed=42,
):
    """
    Volet B : Stabilité log-euclidienne de G(x) sur une grille fixe de points-ponts
    et ICC interséance estimé sur `G_rep_by_participant_session` (participant, seance, grille, k, k).
    Décision préenregistrée fondée sur les bornes d'intervalle bootstrap à 95 % :
      - borne supérieure de CI_95%(Q_0.95(d_G)) <= delta_G_margin
      - borne inférieure de CI_95%(ICC_G) >= rho_G
    """
    G_d = np.asarray(G_disc_grid, dtype=float)
    G_ps = np.asarray(G_rep_by_participant_session, dtype=float)
    if G_ps.ndim != 5:
        raise ValueError("`G_rep_by_participant_session` doit avoir la forme (participant, seance, grille, k, k).")

    n_p, n_s, n_grid, k, _ = G_ps.shape
    triu_i, triu_j = np.triu_indices(k)

    logs_full = np.zeros((n_p, n_s, n_grid, k, k), dtype=float)
    logs_sym = np.zeros((n_p, n_s, n_grid, len(triu_i)), dtype=float)
    for i in range(n_p):
        for s in range(n_s):
            for g in range(n_grid):
                L = spd_logm(G_ps[i, s, g])
                logs_full[i, s, g] = L
                logs_sym[i, s, g] = L[triu_i, triu_j]

    mean_log_rep = np.mean(logs_full, axis=(0, 1))
    log_disc = np.array([spd_logm(G_d[g]) for g in range(n_grid)], dtype=float)

    dists = np.array(
        [float(np.linalg.norm(mean_log_rep[g] - log_disc[g], ord="fro")) for g in range(n_grid)],
        dtype=float,
    )
    q95_dG = float(np.quantile(dists, 0.95))

    icc_global = compute_session_icc_from_logs(logs_sym)
    icc_by_grid = [compute_session_icc_from_logs(logs_sym[:, :, [g], :]) for g in range(n_grid)]
    icc_q05 = float(np.quantile(icc_by_grid, 0.05))
    icc_min = float(np.min(icc_by_grid))

    rng = np.random.default_rng(seed)
    boot_q95 = np.empty(int(n_boot), dtype=float)
    boot_icc = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        p_idx = rng.choice(n_p, size=n_p, replace=True)
        sub_full = logs_full[p_idx]
        sub_sym = logs_sym[p_idx]

        mean_sub = np.mean(sub_full, axis=(0, 1))
        b_dists = [float(np.linalg.norm(mean_sub[g] - log_disc[g], ord="fro")) for g in range(n_grid)]
        boot_q95[b] = float(np.quantile(b_dists, 0.95))

        if icc_criterion == "q05":
            b_grid_iccs = [compute_session_icc_from_logs(sub_sym[:, :, [g], :]) for g in range(n_grid)]
            boot_icc[b] = float(np.quantile(b_grid_iccs, 0.05))
        elif icc_criterion == "min":
            b_grid_iccs = [compute_session_icc_from_logs(sub_sym[:, :, [g], :]) for g in range(n_grid)]
            boot_icc[b] = float(np.min(b_grid_iccs))
        else:
            boot_icc[b] = compute_session_icc_from_logs(sub_sym)

    ci_q95 = np.quantile(boot_q95, [0.025, 0.975])
    ci_icc = np.quantile(boot_icc, [0.025, 0.975])

    passed = bool(
        (ci_q95[1] <= float(delta_G_margin))
        and (ci_icc[0] >= float(rho_G))
    )

    return {
        "I_G": int(passed),
        "bootstrap_criteria_pass": passed,
        "icc_criterion_used": icc_criterion,
        "q95_dG": round(q95_dG, 4),
        "ci95_q95_dG": [round(float(ci_q95[0]), 4), round(float(ci_q95[1]), 4)],
        "icc_session_G_global": round(icc_global, 4),
        "icc_session_G_q05": round(icc_q05, 4),
        "icc_session_G_min": round(icc_min, 4),
        "icc_by_grid": [round(float(v), 4) for v in icc_by_grid],
        "ci95_icc_target": [round(float(ci_icc[0]), 4), round(float(ci_icc[1]), 4)],
        "delta_G_margin": float(delta_G_margin),
        "rho_G_threshold": float(rho_G),
    }


def evaluate_volume_oos_gain(
    y_test,
    X_M0_test,
    V_psi_test,
    beta_M0_frozen,
    beta_M1_frozen,
    participant_ids,
    pair_ids,
    delta_V=0.0,
    n_boot=400,
    n_perm=299,
    seed=42,
):
    """
    Volet C : Gain prédictif hors échantillon Delta_S_V = S_test(M0 + V_psi) - S_test(M0)
    avec paramètres figés depuis la découverte et échange strict intra-paire sur `pair_ids`.
    """
    y = np.asarray(y_test, dtype=float)
    X0 = np.asarray(X_M0_test, dtype=float)
    v_col = np.asarray(V_psi_test, dtype=float).reshape(-1, 1)
    X1 = np.hstack([X0, v_col])

    pred_0 = X0 @ np.asarray(beta_M0_frozen, dtype=float)
    pred_1 = X1 @ np.asarray(beta_M1_frozen, dtype=float)

    loss_0 = (y - pred_0) ** 2
    loss_1 = (y - pred_1) ** 2
    pointwise_gain = loss_0 - loss_1
    delta_S_obs = float(np.mean(pointwise_gain))

    rng = np.random.default_rng(seed)
    boot_gains = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = _cluster_bootstrap_indices(participant_ids, rng)
        boot_gains[b] = float(np.mean(pointwise_gain[idx]))

    ci_lo, ci_hi = np.quantile(boot_gains, [0.025, 0.975])
    passed = bool(ci_lo > float(delta_V))

    p_arr = np.asarray(pair_ids)
    unique_pairs = np.unique(p_arr)
    pair_indices = []
    for pid in unique_pairs:
        idx = np.where(p_arr == pid)[0]
        if len(idx) != 2:
            raise ValueError(f"L'unité échangeable {pid!r} doit contenir exactement 2 observations appariées.")
        pair_indices.append(idx)

    ge_perm = 0
    for _ in range(int(n_perm)):
        v_perm = v_col.copy()
        for idx_pair in pair_indices:
            if rng.integers(0, 2) == 1:
                v_perm[idx_pair[0]], v_perm[idx_pair[1]] = v_perm[idx_pair[1]].copy(), v_perm[idx_pair[0]].copy()
        X1_p = np.hstack([X0, v_perm])
        pred_1_p = X1_p @ np.asarray(beta_M1_frozen, dtype=float)
        gain_p = float(np.mean(loss_0 - (y - pred_1_p) ** 2))
        if gain_p >= delta_S_obs:
            ge_perm += 1

    p_V = (1.0 + ge_perm) / (int(n_perm) + 1.0)

    return {
        "I_V": int(passed),
        "delta_S_V": round(delta_S_obs, 4),
        "ci95_delta_S_V": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "delta_V_threshold": float(delta_V),
        "p_value_V_paired_perm": round(float(p_V), 4),
    }


def evaluate_conjunctive_criterion(res_jnd, res_G, res_V):
    """Combine les trois volets de façon conjonctive : I_conjoint = I_J * I_G * I_V."""
    i_j = int(res_jnd["I_J"])
    i_g = int(res_G["I_G"])
    i_v = int(res_V["I_V"])
    i_conj = int(i_j * i_g * i_v)

    failed_parts = []
    if i_j == 0:
        failed_parts.append("JND")
    if i_g == 0:
        failed_parts.append("G")
    if i_v == 0:
        failed_parts.append("volume")

    return {
        "I_conjoint": i_conj,
        "confirmed": bool(i_conj == 1),
        "failed_components": failed_parts,
    }


# ---------------------------------------------------------------------------
# 4. Intégration d'un volume conditionnel par ellipsoïde et ajustement JND
# ---------------------------------------------------------------------------

def integrate_ellipsoid_volume_log_euclidean(
    center,
    cov_matrix,
    unit_ball_mc,
    grid_coords,
    log_G_grid,
    bandwidth=1.2,
    eig_floor=1e-8,
):
    """
    Calcule V_psi = V_euclidien * (1/N) * sum_n sqrt(det G(z_n)) sur des points
    communs de la boule unité projetés dans l'ellipsoïde conditionnel.
    """
    c = np.asarray(center, dtype=float)
    cov = 0.5 * (np.asarray(cov_matrix, dtype=float) + np.asarray(cov_matrix, dtype=float).T)
    k = len(c)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, float(eig_floor))
    radii = np.sqrt(eigvals)
    vol_euclid = ball_volume_factor(k) * float(np.prod(radii))

    z_samples = c[None, :] + (unit_ball_mc * radii[None, :]) @ eigvecs.T
    dists_sq = np.sum((z_samples[:, None, :] - grid_coords[None, :, :]) ** 2, axis=-1)
    weights = np.exp(-0.5 * dists_sq / max(float(bandwidth) ** 2, 1e-12))
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)

    log_G_z = np.einsum("ng,gij->nij", weights, log_G_grid)
    traces = np.trace(log_G_z, axis1=1, axis2=2)
    sqrt_det_G = np.exp(0.5 * traces)

    return float(vol_euclid * float(np.mean(sqrt_det_G)))


def _fit_psychometric_jnd(stim_levels, correct_responses):
    """
    Ajuste une fonction psychométrique cumulative normale 2-AFC :
      P(correct | x) = 0.5 + 0.5 * Phi((x - mu_thr) / sig).
    À x = mu_thr, P(correct) = 0.75 exactement : mu_hat est donc le seuil JND à 75 %.
    """
    x = np.asarray(stim_levels, dtype=float)
    y = np.asarray(correct_responses, dtype=float)

    def neg_ll(params):
        mu_thr, log_sig = params
        sig = math.exp(np.clip(log_sig, -5.0, 3.0))
        p_corr = 0.5 + 0.5 * norm.cdf((x - mu_thr) / sig)
        p_corr = np.clip(p_corr, 1e-6, 1.0 - 1e-6)
        return -float(np.sum(y * np.log(p_corr) + (1.0 - y) * np.log(1.0 - p_corr)))

    res = minimize(neg_ll, x0=[float(np.median(x)), math.log(0.15)], method="Nelder-Mead")
    if not res.success and not np.isfinite(res.fun):
        return float("nan"), False
    mu_hat = float(res.x[0])
    if not np.isfinite(mu_hat) or mu_hat <= 1e-5:
        return float("nan"), False
    return mu_hat, True


# ---------------------------------------------------------------------------
# 5. Simulation de puissance conjointe
# ---------------------------------------------------------------------------

def run_joint_power_simulation(
    scenario_label="1_stabilite_pure_et_deformation",
    n_participants=60,
    n_sessions=2,
    n_trials_per_jnd=120,
    n_replications=200,
    true_cohort_drift_G=0.0,
    true_cond_deformation_G=0.25,
    true_ellipsoid_expansion=0.45,
    true_beta_V=0.50,
    delta_jnd_margin=0.06,
    delta_G_margin=0.25,
    rho_G=0.65,
    icc_criterion="global",
    delta_V=0.0,
    seed=2026,
):
    """
    Simule l'intégralité du pipeline en séparant :
      1. `true_cohort_drift_G` : dérive intercohorte sur le champ de référence G_A(x)
         (0.0 = stabilité pure entre découverte et réplication).
      2. `true_cond_deformation_G` : déformation théorique Delta_G(x) = log G_B(x) - log G_A(x)
         entre les conditions A et B à l'intérieur de chaque participant-séance.
      3. `true_ellipsoid_expansion` et `true_beta_V` : dilatation géométrique de l'ellipsoïde
         et lien prédictif hors échantillon de V_psi au-delà de M0.
    """
    rng = np.random.default_rng(seed)
    n_grid = 6
    k_rank = 2

    grid_coords = np.array(
        [[-1.2, -0.8], [-0.4, 0.9], [0.0, -0.3], [0.5, 0.7], [1.0, -0.6], [1.3, 0.4]],
        dtype=float,
    )
    log_G0 = np.array([np.diag([0.35 + 0.12 * g, -0.15 + 0.08 * g]) for g in range(n_grid)], dtype=float)
    drift_G_mat = np.array(
        [np.diag([float(true_cohort_drift_G), -0.5 * float(true_cohort_drift_G)]) for _ in range(n_grid)],
        dtype=float,
    )
    cond_def_mat = np.array(
        [np.diag([float(true_cond_deformation_G), 0.6 * float(true_cond_deformation_G)]) for _ in range(n_grid)],
        dtype=float,
    )

    dirs_mc = rng.standard_normal((120, k_rank))
    dirs_mc /= np.maximum(np.linalg.norm(dirs_mc, axis=1, keepdims=True), 1e-12)
    unit_ball_mc = dirs_mc * (rng.uniform(0.0, 1.0, size=(120, 1)) ** (1.0 / k_rank))

    pass_j, pass_g, pass_v, pass_joint = 0, 0, 0, 0
    cov_j_hits, cov_g_hits = 0, 0
    nonconv = 0
    cond_def_norms = []

    n_disc = n_participants // 2
    n_test = n_participants - n_disc
    stim_levels = np.linspace(0.25, 1.15, 6)
    trials_per_level = max(int(n_trials_per_jnd) // len(stim_levels), 5)

    for rep in range(int(n_replications)):
        # 1. Génération hiérarchique des champs G_A (référence) et G_B (condition déformée)
        G_A_all = np.zeros((n_participants, n_sessions, n_grid, k_rank, k_rank), dtype=float)
        log_G_A_all = np.zeros((n_participants, n_sessions, n_grid, k_rank, k_rank), dtype=float)
        log_G_B_all = np.zeros((n_participants, n_sessions, n_grid, k_rank, k_rank), dtype=float)

        for i in range(n_participants):
            u_i = 0.08 * rng.standard_normal((n_grid, k_rank, k_rank))
            u_i = 0.5 * (u_i + np.swapaxes(u_i, -1, -2))
            for s in range(n_sessions):
                v_is = 0.04 * rng.standard_normal((n_grid, k_rank, k_rank))
                v_is = 0.5 * (v_is + np.swapaxes(v_is, -1, -2))
                e_is_B = 0.02 * rng.standard_normal((n_grid, k_rank, k_rank))
                e_is_B = 0.5 * (e_is_B + np.swapaxes(e_is_B, -1, -2))
                for g in range(n_grid):
                    cohort_drift = 0.0 if i < n_disc else drift_G_mat[g]
                    L_A = log_G0[g] + cohort_drift + u_i[g] + v_is[g]
                    L_B = L_A + cond_def_mat[g] + e_is_B[g]
                    log_G_A_all[i, s, g] = L_A
                    log_G_B_all[i, s, g] = L_B
                    G_A_all[i, s, g] = spd_expm(L_A)

        G_disc_grid = np.array([spd_expm(np.mean(log_G_A_all[:n_disc, :, g], axis=(0, 1))) for g in range(n_grid)])
        G_rep_ps = G_A_all[n_disc:]

        mean_cond_def = float(np.mean(np.linalg.norm(log_G_B_all[n_disc:] - log_G_A_all[n_disc:], ord="fro", axis=(-2, -1))))
        cond_def_norms.append(mean_cond_def)

        # 2. Réajustement psychométrique non biaisé des JND sur le jeu test
        jnd_pred_list, jnd_obs_list, jnd_clusters = [], [], []
        rep_failed = False
        for idx_t, i_part in enumerate(range(n_disc, n_participants)):
            for g in range(n_grid):
                true_jnd = 1.0 / math.sqrt(max(float(G_A_all[i_part, 0, g, 0, 0]), 1e-8))
                pred_jnd = 1.0 / math.sqrt(max(float(G_disc_grid[g, 0, 0]), 1e-8))

                x_trials = np.repeat(stim_levels, trials_per_level)
                p_corr = 0.5 + 0.5 * norm.cdf((x_trials - true_jnd) / 0.14)
                y_trials = rng.binomial(1, p_corr)
                obs_hat, ok = _fit_psychometric_jnd(x_trials, y_trials)
                if not ok:
                    rep_failed = True
                    break
                jnd_pred_list.append(pred_jnd)
                jnd_obs_list.append(obs_hat)
                jnd_clusters.append(idx_t)
            if rep_failed:
                break

        if rep_failed:
            nonconv += 1
            continue

        # 3. Volumes conditionnels intégrés (V_A sous G_A et V_B sous G_B)
        def build_conditional_volume_dataset(p_start, p_end):
            X0_rows, V_rows, y_rows, part_rows, pair_rows = [], [], [], [], []
            pair_counter = 0
            for local_p, i_part in enumerate(range(p_start, p_end)):
                for s in range(n_sessions):
                    base_center = 0.25 * rng.standard_normal(k_rank)
                    A_cov = np.diag([0.55, 0.40]) + 0.05 * rng.standard_normal((k_rank, k_rank))
                    cov_A = A_cov @ A_cov.T + 0.15 * np.eye(k_rank)

                    exp_factor = math.exp(float(true_ellipsoid_expansion))
                    cov_B = exp_factor * cov_A
                    center_B = base_center + np.array([
                        0.45 * float(true_ellipsoid_expansion),
                        -0.25 * float(true_ellipsoid_expansion),
                    ])

                    vol_A = integrate_ellipsoid_volume_log_euclidean(
                        base_center, cov_A, unit_ball_mc, grid_coords, log_G_A_all[i_part, s], bandwidth=1.2
                    )
                    vol_B = integrate_ellipsoid_volume_log_euclidean(
                        center_B, cov_B, unit_ball_mc, grid_coords, log_G_B_all[i_part, s], bandwidth=1.2
                    )

                    cov_sens = float(rng.standard_normal())
                    cov_rt = float(rng.standard_normal())
                    for vol_c in [vol_A, vol_B]:
                        log_v = math.log(max(vol_c, 1e-8))
                        y_val = (
                            0.75 * cov_sens
                            - 0.45 * cov_rt
                            + float(true_beta_V) * log_v
                            + 0.50 * rng.standard_normal()
                        )
                        X0_rows.append([1.0, cov_sens, cov_rt])
                        V_rows.append(log_v)
                        y_rows.append(y_val)
                        part_rows.append(local_p)
                        pair_rows.append(pair_counter)
                    pair_counter += 1

            return (
                np.asarray(X0_rows),
                np.asarray(V_rows),
                np.asarray(y_rows),
                np.asarray(part_rows),
                np.asarray(pair_rows),
            )

        X0_tr, V_tr, y_tr, _, _ = build_conditional_volume_dataset(0, n_disc)
        X0_te, V_te, y_te, p_te, pair_te = build_conditional_volume_dataset(n_disc, n_participants)

        beta_M0, _, _, _ = np.linalg.lstsq(X0_tr, y_tr, rcond=None)
        beta_M1, _, _, _ = np.linalg.lstsq(np.column_stack([X0_tr, V_tr]), y_tr, rcond=None)

        res_j = evaluate_jnd_equivalence(
            jnd_pred_list, jnd_obs_list, jnd_clusters, delta_jnd=delta_jnd_margin, n_boot=120, seed=seed + rep
        )
        res_g = evaluate_field_stability(
            G_disc_grid,
            G_rep_ps,
            delta_G_margin=delta_G_margin,
            rho_G=rho_G,
            icc_criterion=icc_criterion,
            n_boot=120,
            seed=seed + rep,
        )
        res_v = evaluate_volume_oos_gain(
            y_te, X0_te, V_te, beta_M0, beta_M1, p_te, pair_te, delta_V=delta_V, n_boot=120, n_perm=99, seed=seed + rep
        )
        conj = evaluate_conjunctive_criterion(res_j, res_g, res_v)

        pass_j += res_j["I_J"]
        pass_g += res_g["I_G"]
        pass_v += res_v["I_V"]
        pass_joint += conj["I_conjoint"]

        if res_j["ci95_bias"][0] <= 0.0 <= res_j["ci95_bias"][1]:
            cov_j_hits += 1
        true_q95 = float(
            np.quantile([log_euclidean_distance(spd_expm(log_G0[g] + drift_G_mat[g]), G_disc_grid[g]) for g in range(n_grid)], 0.95)
        )
        if res_g["ci95_q95_dG"][0] <= true_q95 <= res_g["ci95_q95_dG"][1]:
            cov_g_hits += 1

    n_req = max(int(n_replications), 1)
    n_val = int(n_replications) - nonconv

    p_j = pass_j / n_req
    p_g = pass_g / n_req
    p_v = pass_v / n_req
    p_jt = pass_joint / n_req

    powers = {"JND": p_j, "G": p_g, "volume": p_v}
    limiting = min(powers, key=powers.get)

    return {
        "scenario": scenario_label,
        "parameters": {
            "n_participants": int(n_participants),
            "n_sessions": int(n_sessions),
            "n_trials_per_jnd": int(n_trials_per_jnd),
            "true_cohort_drift_G": float(true_cohort_drift_G),
            "true_cond_deformation_G": float(true_cond_deformation_G),
            "true_ellipsoid_expansion": float(true_ellipsoid_expansion),
            "true_beta_V": float(true_beta_V),
            "delta_jnd_margin": float(delta_jnd_margin),
            "delta_G_margin": float(delta_G_margin),
            "rho_G": float(rho_G),
            "icc_criterion": icc_criterion,
        },
        "n_requested": int(n_replications),
        "n_valid": int(n_val),
        "n_nonconvergent": int(nonconv),
        "mean_cond_deformation_norm_B_vs_A": round(float(np.mean(cond_def_norms)), 4),
        "power_JND": round(p_j, 4),
        "power_JND_wilson95": wilson_ci_95(pass_j, n_req),
        "power_G": round(p_g, 4),
        "power_G_wilson95": wilson_ci_95(pass_g, n_req),
        "power_volume": round(p_v, 4),
        "power_volume_wilson95": wilson_ci_95(pass_v, n_req),
        "power_joint": round(p_jt, 4),
        "power_joint_wilson95": wilson_ci_95(pass_joint, n_req),
        "limiting_component": limiting,
        "coverage_JND": round(cov_j_hits / n_req, 4),
        "coverage_G": round(cov_g_hits / n_req, 4),
        "nonconvergence_rate": round(nonconv / n_req, 4),
    }


def run_three_reference_benchmarks(n_participants=60, n_replications=80, seed=2026):
    """
    Exécute les trois configurations de référence :
      1. Stabilité pure de G_A + déformation conditionnelle G_B vs G_A et gain volumique
      2. Contrôle négatif avec dérive intercohorte (true_cohort_drift_G > delta_G_margin)
      3. Scénario nul sans déformation conditionnelle ni lien volumique
    """
    bench_1 = run_joint_power_simulation(
        scenario_label="1_stabilite_pure_et_deformation",
        n_participants=n_participants,
        n_replications=n_replications,
        true_cohort_drift_G=0.0,
        true_cond_deformation_G=0.25,
        true_ellipsoid_expansion=0.45,
        true_beta_V=0.50,
        delta_jnd_margin=0.06,
        delta_G_margin=0.25,
        rho_G=0.65,
        seed=seed,
    )
    bench_2 = run_joint_power_simulation(
        scenario_label="2_derive_intercohorte_controle_negatif",
        n_participants=n_participants,
        n_replications=n_replications,
        true_cohort_drift_G=0.35,
        true_cond_deformation_G=0.25,
        true_ellipsoid_expansion=0.45,
        true_beta_V=0.50,
        delta_jnd_margin=0.06,
        delta_G_margin=0.25,
        rho_G=0.65,
        seed=seed + 10000,
    )
    bench_3 = run_joint_power_simulation(
        scenario_label="3_scenario_nul_sans_deformation",
        n_participants=n_participants,
        n_replications=n_replications,
        true_cohort_drift_G=0.0,
        true_cond_deformation_G=0.0,
        true_ellipsoid_expansion=0.0,
        true_beta_V=0.0,
        delta_jnd_margin=0.06,
        delta_G_margin=0.25,
        rho_G=0.65,
        seed=seed + 20000,
    )

    bench_1["joint_null_pass_rate"] = bench_3["power_joint"]
    return {
        "1_stabilite_pure_et_deformation": bench_1,
        "2_derive_intercohorte_controle_negatif": bench_2,
        "3_scenario_nul_sans_deformation": bench_3,
    }


if __name__ == "__main__":
    report = run_three_reference_benchmarks(n_participants=60, n_replications=80, seed=2026)
    print(json.dumps(report, indent=2, ensure_ascii=False))
