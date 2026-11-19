"""
test_synthetic_alignment.py

Test de récupération synthétique pour le pipeline Procrustes figé.

Ce test N'EST PAS une validation psychophysique. Il vérifie uniquement
que, sur une géométrie contrôlée :

    X_obs = s * X_ref @ R.T + t + bruit

la transformation estimée sur les ponts d'entraînement, puis appliquée
figée par apply_procrustes_transform, permet de récupérer la géométrie
de référence sur des ponts tenus à l'écart.

Deux modes sont comparés :
    centering_mode="common"        (référence principale)
    centering_mode="per_condition" (analyse de sensibilité)

Trois erreurs rapportées séparément :
    1. erreur point-à-point sur les ponts tenus à l'écart ;
    2. erreur des distances inter-points ;
    3. erreur relative du volume estimé par condition.

Contrat attendu (adapter si la signature réelle diffère) :
    apply_procrustes_transform(X, transform, *, centering_mode, condition_ids)
    transform : dict ou objet avec R, t, s
    local_volume_in_region(region_points) -> float
"""

from __future__ import annotations
import math
import numpy as np
from math import gamma

from local_jnd_bridge_calibration import (
    apply_procrustes_transform,
    local_volume_in_region,
)


# ---------- générateurs ----------

def random_rotation(rng, d):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def make_reference_geometry(rng, n_conditions, n_anchors, n_unique, d):
    centers = rng.uniform(-4.0, 4.0, size=(n_conditions, d))
    anchors = rng.standard_normal((n_anchors, d)) * 0.4
    X, cond, pid = [], [], []
    for c in range(n_conditions):
        X.append(anchors + centers[c])
        cond.extend([c] * n_anchors)
        pid.extend(range(n_anchors))
        uniq = rng.standard_normal((n_unique, d)) * 0.4
        X.append(uniq + centers[c])
        cond.extend([c] * n_unique)
        pid.extend(range(n_anchors, n_anchors + n_unique))
    return np.vstack(X), np.array(cond), np.array(pid)


# ---------- métriques ----------

def ellipsoid_volume(points):
    cov = np.cov(points.T)
    d = points.shape[1]
    unit_ball = math.pi ** (d / 2) / gamma(d / 2 + 1)
    det = max(float(np.linalg.det(cov)), 0.0)
    return unit_ball * math.sqrt(det)


def interpoint_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def distance_error(X_ref, X_est):
    D_ref = interpoint_distances(X_ref)
    D_est = interpoint_distances(X_est)
    mask = ~np.eye(len(X_ref), dtype=bool)
    rel = np.abs(D_est - D_ref)[mask] / (D_ref[mask] + 1e-12)
    return float(np.median(rel)), float(np.percentile(rel, 95))


def pointwise_error(X_ref, X_est):
    diff = np.linalg.norm(X_est - X_ref, axis=1)
    return float(np.median(diff)), float(np.percentile(diff, 95))


def volume_error_by_condition(X_ref, X_est, cond):
    errs = []
    for c in np.unique(cond):
        m = cond == c
        if m.sum() < X_ref.shape[1] + 1:
            continue
        V_ref = ellipsoid_volume(X_ref[m])
        V_est = ellipsoid_volume(X_est[m])
        if V_ref <= 1e-12:
            continue
        errs.append(abs(V_est - V_ref) / V_ref)
    return float(np.median(errs)), float(np.max(errs))


# ---------- Procrustes (contrôle du test) ----------

def estimate_similarity_transform(X_src, X_dst):
    mu_s = X_src.mean(axis=0)
    mu_d = X_dst.mean(axis=0)
    Xc = X_src - mu_s
    Yc = X_dst - mu_d
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.eye(X_src.shape[1]); D[-1, -1] = d
    R = Vt.T @ D @ U.T
    var_s = (Xc ** 2).sum()
    s = (S * np.diag(D)).sum() / (var_s + 1e-12)
    t = mu_d - s * (mu_s @ R.T)
    return {"R": R, "t": t, "s": s}


def apply_frozen(transform, X, condition_ids, mode):
    try:
        return apply_procrustes_transform(
            X, transform, centering_mode=mode, condition_ids=condition_ids,
        )
    except TypeError:
        return apply_procrustes_transform(
            X, transform=transform, centering_mode=mode,
            condition_ids=condition_ids,
        )


# ---------- cas principal ----------

def run_case(mode, seed=0):
    rng = np.random.default_rng(seed)
    d, n_cond, n_anchors, n_unique = 3, 3, 5, 6

    X_ref, cond, pid = make_reference_geometry(rng, n_cond, n_anchors, n_unique, d)

    R_true = random_rotation(rng, d)
    s_true = 1.7
    t_true = rng.uniform(-2.0, 2.0, size=d)
    X_obs = s_true * (X_ref @ R_true.T) + t_true
    X_obs = X_obs + rng.standard_normal(X_obs.shape) * 0.02

    train_mask = (pid < n_anchors) | ((pid - n_anchors) % 2 == 0)
    held_mask = ~train_mask

    transform = estimate_similarity_transform(X_obs[train_mask], X_ref[train_mask])
    X_est = apply_frozen(transform, X_obs, condition_ids=cond, mode=mode)

    pm, p95 = pointwise_error(X_ref[held_mask], X_est[held_mask])
    dm, d95 = distance_error(X_ref[held_mask], X_est[held_mask])
    vm, vmax = volume_error_by_condition(
        X_ref[held_mask], X_est[held_mask], cond[held_mask],
    )
    return {
        "mode": mode,
        "point_median": pm, "point_p95": p95,
        "dist_median": dm,  "dist_p95": d95,
        "vol_median": vm,   "vol_max": vmax,
    }


def test_common_vs_per_condition():
    r_common = run_case("common", seed=0)
    r_pc     = run_case("per_condition", seed=0)

    for r in (r_common, r_pc):
        print(
            f"[{r['mode']:14s}] "
            f"point_med={r['point_median']:.4f}  "
            f"dist_med={r['dist_median']:.4f}  "
            f"vol_med={r['vol_median']:.4f}"
        )

    # Le scénario est à référence globale : common doit être au moins
    # comparable. Tolérance 1.5x pour absorber la stochasticité.
    assert r_common["dist_median"] <= 1.5 * r_pc["dist_median"]
    assert r_common["vol_median"]  <= 1.5 * r_pc["vol_median"]
    return r_common, r_pc


def test_per_condition_absent_raises():
    rng = np.random.default_rng(1)
    X_ref, cond, _ = make_reference_geometry(rng, 3, 4, 4, 3)
    R = random_rotation(rng, 3)
    X_obs = 1.3 * (X_ref @ R.T) + rng.uniform(-1, 1, 3)
    transform = estimate_similarity_transform(X_obs, X_ref)

    cond_new = cond.copy()
    cond_new[:5] = 99  # condition jamais vue à l'entraînement
    try:
        apply_frozen(transform, X_obs, condition_ids=cond_new, mode="per_condition")
    except Exception:
        return
    raise AssertionError(
        "per_condition aurait dû refuser une condition absente de l'entraînement"
    )


if __name__ == "__main__":
    test_common_vs_per_condition()
    test_per_condition_absent_raises()
    print("OK")
