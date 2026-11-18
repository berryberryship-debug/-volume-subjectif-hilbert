#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Couche théorique exploratoire : génère des variables candidates q(t) à partir
d'une équation de Schrödinger non-linéaire discrète à jauge covariante :

    i * d_t psi = -0.5 * Delta_A psi + V_eff(|psi|^2) * psi - i * gamma * psi

Convention de jauge et d'holonomie :
  - Lien covariant : U_j = exp(-i * A_j * dx) avec A_j = flux_theta / L.
  - Produit de boucle : prod_j U_j = exp(-i * flux_theta).
  - Holonomie de configuration : Phi_hol = (-arg(prod_j U_j)) mod 2*pi = flux_theta mod 2*pi.

Statut épistémique : simulation numérique exploratoire générant des variables
candidates q(t). Ne modifie ni G(x) ni V_psi, et ne constitue ni une énergie
biologique, ni une courbure relativiste, ni une mesure de la conscience.
"""

import json
import math
import numpy as np
from scipy.linalg import lu_factor, lu_solve


def _periodic_smooth(arr, passes=2):
    out = np.asarray(arr, dtype=float).copy()
    for _ in range(int(passes)):
        out = 0.25 * np.roll(out, 1) + 0.50 * out + 0.25 * np.roll(out, -1)
    return out


def angular_distance(a, b):
    """Distance angulaire circulaire minimale sur le cercle unité (évite le saut 0 / 2*pi)."""
    return float(abs(np.angle(np.exp(1j * (float(a) - float(b))))))


class ModeleHolotelegraphiqueExploratoire:
    def __init__(
        self,
        n_x=128,
        length=10.0,
        dt=0.005,
        g_kerr=0.5,
        flux_theta=0.4,
        gamma_dissip=0.0,
        retroaction=0.0,
        coeff_courbure=0.05,
        renormalize_l2=False,
        bandwidth_init=1.0,
        seed=42,
    ):
        self.n_x = int(n_x)
        self.length = float(length)
        self.dt = float(dt)
        self.g_kerr = float(g_kerr)
        self.flux_theta = float(flux_theta)
        self.gamma_dissip = float(gamma_dissip)
        self.retroaction = float(retroaction)
        self.coeff_courbure = float(coeff_courbure)
        self.renormalize_l2 = bool(renormalize_l2)
        self.bandwidth_init = float(bandwidth_init)

        if self.n_x < 4:
            raise ValueError("n_x doit être au moins égal à 4.")
        if self.length <= 0.0:
            raise ValueError("length doit être strictement positif.")
        if self.dt <= 0.0:
            raise ValueError("dt doit être strictement positif.")
        if self.bandwidth_init <= 0.0:
            raise ValueError("bandwidth_init doit être strictement positif.")
        if self.gamma_dissip < 0.0:
            raise ValueError("gamma_dissip doit être non négatif.")

        self.dx = self.length / self.n_x
        self.rng = np.random.default_rng(seed)

        self.x = np.linspace(0.0, self.length, self.n_x, endpoint=False)
        self.a_field = np.full(self.n_x, self.flux_theta / self.length, dtype=float)
        self.u_link = np.exp(-1j * self.a_field * self.dx)

        # Pré-construction et factorisation LU unique du pas cinétique Crank-Nicolson :
        # Pour i * d_t psi = -0.5 * Delta_A psi :
        #   (I - i * (dt / 4) * Delta_A) psi_{n+1} = (I + i * (dt / 4) * Delta_A) psi_n
        lap_mat = np.zeros((self.n_x, self.n_x), dtype=complex)
        inv_dx2 = 1.0 / (self.dx ** 2)
        for j in range(self.n_x):
            jp = (j + 1) % self.n_x
            jm = (j - 1) % self.n_x
            lap_mat[j, j] = -2.0 * inv_dx2
            lap_mat[j, jp] = self.u_link[j] * inv_dx2
            lap_mat[j, jm] = np.conj(self.u_link[jm]) * inv_dx2

        eye = np.eye(self.n_x, dtype=complex)
        alpha = 0.25j * self.dt
        self._cn_rhs_mat = eye + alpha * lap_mat
        self._cn_lhs_lu = lu_factor(eye - alpha * lap_mat)

        # Paquet d'onde initial centré
        x_rel = (self.x - 0.5 * self.length) / self.bandwidth_init
        psi0 = np.exp(-0.5 * x_rel ** 2) * np.exp(1j * 0.5 * x_rel)
        norm0 = math.sqrt(float(np.sum(np.abs(psi0) ** 2)) * self.dx)
        self.psi = psi0 / max(norm0, 1e-12)
        self.pi_ref = self._compute_flux_observable(self.psi, self.u_link)

    def _compute_flux_observable(self, psi, u_link):
        d_cov = (u_link * np.roll(psi, -1) - psi) / self.dx
        return float(np.sum(np.imag(np.conj(psi) * d_cov)) * self.dx)

    def courbure_densite_lissee(self, rho):
        """Dérivée seconde spatiale lissée de la densité (non assimilable à une courbure relativiste)."""
        rho_s = _periodic_smooth(rho, passes=2)
        return (np.roll(rho_s, -1) - 2.0 * rho_s + np.roll(rho_s, 1)) / (self.dx ** 2)

    def diagnostic_energie_modele(self, psi=None, u_link=None):
        """
        Fonctionnelle diagnostique adimensionnelle du modèle numérique.
        Ne représente ni une énergie physique ni une énergie biologique/consciente.
        """
        p = self.psi if psi is None else psi
        u = self.u_link if u_link is None else u_link
        rho = np.abs(p) ** 2
        d_cov = (u * np.roll(p, -1) - p) / self.dx
        e_kin = 0.5 * float(np.sum(np.abs(d_cov) ** 2) * self.dx)
        e_int = 0.5 * self.g_kerr * float(np.sum(rho ** 2) * self.dx)
        curv = self.courbure_densite_lissee(rho)
        e_curv = 0.5 * self.coeff_courbure * float(np.sum(curv * rho) * self.dx)
        return e_kin + e_int + e_curv

    def get_configuration_metadata(self, u_link=None):
        """
        Paramètres invariants de la configuration (constants au cours du temps).
        Avec U_j = exp(-i * A_j * dx), prod_j U_j = exp(-i * flux_theta), d'où :
            Phi_hol = (-arg(prod_j U_j)) mod 2*pi = flux_theta mod 2*pi.
        """
        u = self.u_link if u_link is None else u_link
        hol_angle = (-float(np.angle(np.prod(u)))) % (2.0 * math.pi)
        return {
            "n_x": self.n_x,
            "length": self.length,
            "dt": self.dt,
            "flux_theta": self.flux_theta,
            "holonomie_config_mod_2pi": float(hol_angle),
            "retroaction": self.retroaction,
            "gamma_dissip": self.gamma_dissip,
        }

    def extract_candidate_features(self, psi=None, u_link=None):
        """
        Extrait le vecteur dynamique candidat q(t) invariant de jauge destiné à M2 :
          - ipr_sans_dimension : L * int(rho^2 dx) / (int(rho dx))^2
          - coherence_hilbertienne : |sum psi_j^* U_j psi_{j+1}| / sum |psi_j|^2
          - courbure_integrale : norme L2 spatiale de la dérivée seconde lissée de rho
          - energie_modele_diag : fonctionnelle diagnostique numérique du modèle
        """
        p = self.psi if psi is None else psi
        u = self.u_link if u_link is None else u_link
        rho = np.abs(p) ** 2
        norm_sq = max(float(np.sum(rho) * self.dx), 1e-12)

        ipr_raw = float(np.sum(rho ** 2) * self.dx) / (norm_sq ** 2)
        ipr_dimless = self.length * ipr_raw

        num_coh = abs(complex(np.sum(np.conj(p) * u * np.roll(p, -1))))
        den_coh = max(float(np.sum(rho)), 1e-12)
        coh_hilbert = num_coh / den_coh

        curv = self.courbure_densite_lissee(rho)
        curv_int = math.sqrt(float(np.sum(curv ** 2) * self.dx))
        e_diag = self.diagnostic_energie_modele(p, u)

        return {
            "ipr_sans_dimension": float(ipr_dimless),
            "coherence_hilbertienne": float(coh_hilbert),
            "courbure_integrale": float(curv_int),
            "energie_modele_diag": float(e_diag),
        }

    def step(self):
        """
        Intégration d'un pas temporel par splitting de Strang symétrique :
          1. Demi-pas potentiel : exp(-i * (dt / 2) * V_eff)
          2. Pas cinétique covariant unitaire : Crank-Nicolson pré-factorisé
          3. Second demi-pas potentiel : exp(-i * (dt / 2) * V_eff)
        """
        rho = np.abs(self.psi) ** 2
        if self.retroaction != 0.0:
            pi_curr = self._compute_flux_observable(self.psi, self.u_link)
            ratio = pi_curr / self.pi_ref if abs(self.pi_ref) > 1e-12 else 1.0
            g_eff = self.g_kerr * (1.0 + self.retroaction * (ratio - 1.0))
        else:
            g_eff = self.g_kerr

        v_eff = g_eff * rho + self.coeff_courbure * self.courbure_densite_lissee(rho)
        psi_half = self.psi * np.exp(-0.5j * v_eff * self.dt)

        rhs = self._cn_rhs_mat @ psi_half
        psi_kin = lu_solve(self._cn_lhs_lu, rhs)

        rho_kin = np.abs(psi_kin) ** 2
        v_eff_2 = g_eff * rho_kin + self.coeff_courbure * self.courbure_densite_lissee(rho_kin)
        psi_next = psi_kin * np.exp(-0.5j * v_eff_2 * self.dt)

        if self.gamma_dissip > 0.0:
            psi_next = psi_next * math.exp(-self.gamma_dissip * self.dt)

        norm_before = math.sqrt(float(np.sum(np.abs(psi_next) ** 2)) * self.dx)
        if self.gamma_dissip > 0.0 and self.renormalize_l2:
            psi_next = psi_next / max(norm_before, 1e-12)
        norm_after = math.sqrt(float(np.sum(np.abs(psi_next) ** 2)) * self.dx)

        self.psi = psi_next
        feats = self.extract_candidate_features()
        feats["norme_L2_avant_renorm"] = float(norm_before)
        feats["norme_L2_finale"] = float(norm_after)
        if self.gamma_dissip > 0.0 and self.renormalize_l2:
            feats["norme_L2_apres_renorm"] = float(norm_after)
        feats["regime"] = "dissipatif" if self.gamma_dissip > 0.0 else "unitaire"
        return feats

    def run(self, n_steps=100):
        return [self.step() for _ in range(int(n_steps))]


def verify_analytical_eigenmode(n_x=128, length=10.0, dt=0.002, n_steps=100, mode_m=2, flux_theta=0.4):
    """
    Contrôle analytique du signe et de l'ordre du schéma de Crank-Nicolson :
    Lorsque g_kerr = 0 et coeff_courbure = 0, un mode propre covariant
        psi_j(0) = exp(i * (2*pi*m/L + A) * x_j) / sqrt(L)
    satisfait Delta_A psi = lambda_m psi avec
        lambda_m = (2 * cos((2*pi*m + flux_theta) / n_x) - 2) / dx^2
    et évolue exactement selon psi(t) = psi(0) * exp(-i * (-0.5 * lambda_m) * t).
    """
    model = ModeleHolotelegraphiqueExploratoire(
        n_x=n_x,
        length=length,
        dt=dt,
        g_kerr=0.0,
        flux_theta=flux_theta,
        gamma_dissip=0.0,
        retroaction=0.0,
        coeff_courbure=0.0,
    )
    k_eff = (2.0 * math.pi * mode_m + flux_theta) / length
    psi_init = np.exp(1j * k_eff * model.x) / math.sqrt(length)
    model.psi = psi_init.copy()

    model.run(n_steps=n_steps)
    t_final = n_steps * dt

    lambda_disc = (2.0 * math.cos(k_eff * model.dx) - 2.0) / (model.dx ** 2)
    omega_exact = -0.5 * lambda_disc
    psi_exact = psi_init * np.exp(-1j * omega_exact * t_final)

    l2_err = math.sqrt(float(np.sum(np.abs(model.psi - psi_exact) ** 2)) * model.dx)
    return {
        "analytical_eigenmode_pass": bool(l2_err < 1e-6),
        "l2_state_error_vs_exact": float(l2_err),
        "omega_exact": round(omega_exact, 6),
    }


def verify_gauge_invariance(model, seed=123):
    """Vérifie l'invariance exacte de q(t) et Phi_hol sous transformation de jauge locale."""
    rng = np.random.default_rng(seed)
    chi = rng.uniform(-math.pi, math.pi, size=model.n_x)
    phase_rot = np.exp(1j * chi)

    psi_prime = phase_rot * model.psi
    u_prime = phase_rot * model.u_link * np.conj(np.roll(phase_rot, -1))

    q_orig = model.extract_candidate_features(model.psi, model.u_link)
    q_trans = model.extract_candidate_features(psi_prime, u_prime)
    hol_orig = model.get_configuration_metadata(model.u_link)["holonomie_config_mod_2pi"]
    hol_trans = model.get_configuration_metadata(u_prime)["holonomie_config_mod_2pi"]

    max_diff = max(abs(q_orig[k] - q_trans[k]) for k in q_orig)
    hol_drift = angular_distance(hol_orig, hol_trans)
    raw_phase_drift = angular_distance(
        float(np.angle(np.mean(model.psi))),
        float(np.angle(np.mean(psi_prime))),
    )

    return {
        "gauge_invariant_pass": bool(max_diff < 1e-11 and hol_drift < 1e-11),
        "max_feature_drift": float(max_diff),
        "holonomy_circular_drift": float(hol_drift),
        "non_covariant_raw_phase_drift": round(raw_phase_drift, 4),
    }


def verify_numerical_convergence(
    resolutions=(64, 128),
    dts=(0.005, 0.0025),
    bandwidths=(0.7, 1.0, 1.4),
    total_time=0.25,
    n_eval_points=25,
):
    """
    Évalue la convergence numérique sur toute la trajectoire temporelle :
      - Dérive maximale de norme L2 par rapport à 1.0
      - Erreur quadratique moyenne RMSE_q par rapport à une référence fine (n_x=256, dt=0.00125)
        pour chaque largeur initiale `bandwidth_init`.
    """
    feature_keys = ["ipr_sans_dimension", "coherence_hilbertienne", "courbure_integrale", "energie_modele_diag"]
    eval_indices_ref = np.linspace(0, int(round(total_time / 0.00125)) - 1, n_eval_points, dtype=int)

    summary = {}
    for bw in bandwidths:
        ref_model = ModeleHolotelegraphiqueExploratoire(
            n_x=256, dt=0.00125, bandwidth_init=bw, gamma_dissip=0.0, retroaction=0.0
        )
        ref_traj = ref_model.run(n_steps=int(round(total_time / 0.00125)))
        ref_mat = np.array([[ref_traj[idx][k] for k in feature_keys] for idx in eval_indices_ref], dtype=float)

        bw_report = {}
        for nx in resolutions:
            for dt in dts:
                n_steps = max(int(round(total_time / dt)), 1)
                eval_idx = np.linspace(0, n_steps - 1, n_eval_points, dtype=int)
                m = ModeleHolotelegraphiqueExploratoire(
                    n_x=nx, dt=dt, bandwidth_init=bw, gamma_dissip=0.0, retroaction=0.0
                )
                traj = m.run(n_steps=n_steps)
                sub_mat = np.array([[traj[idx][k] for k in feature_keys] for idx in eval_idx], dtype=float)

                rmse_q = float(np.sqrt(np.mean((sub_mat - ref_mat) ** 2)))
                max_norm_drift = float(np.max([abs(step["norme_L2_finale"] - 1.0) for step in traj]))

                bw_report[f"nx{nx}_dt{dt}"] = {
                    "rmse_q_vs_ref": round(rmse_q, 6),
                    "max_norm_L2_drift": float(max_norm_drift),
                }
        summary[f"bandwidth_{bw}"] = bw_report
    return summary


if __name__ == "__main__":
    mod = ModeleHolotelegraphiqueExploratoire(n_x=128, dt=0.005, gamma_dissip=0.0, retroaction=0.0)
    mod.run(n_steps=40)
    audit = {
        "statut_epistemique": "simulation_exploratoire_non_empirique",
        "metadata_configuration": mod.get_configuration_metadata(),
        "controle_mode_analytique": verify_analytical_eigenmode(),
        "controle_invariance_jauge": verify_gauge_invariance(mod),
        "controle_convergence": verify_numerical_convergence(),
        "dernier_vecteur_q_dynamique": {k: round(v, 6) for k, v in mod.extract_candidate_features().items()},
    }
    print(json.dumps(audit, indent=2, ensure_ascii=False))
