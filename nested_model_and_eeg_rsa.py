#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Couche théorique exploratoire : génère des variables candidates q(t) à partir
d'un champ complexe discret sur réseau à jauge covariante.

Ce module ne constitue ni une mesure de la conscience, ni une énergie biologique,
ni une courbure de l'espace-temps.
"""

import json
import math

import numpy as np


def _periodic_smooth(arr, passes=2):
    out = np.asarray(
        arr,
        dtype=float,
    ).copy()

    for _ in range(int(passes)):
        out = (
            0.25 * np.roll(out, 1)
            + 0.50 * out
            + 0.25 * np.roll(out, -1)
        )

    return out


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
        self.dx = self.length / self.n_x
        self.dt = float(dt)
        self.g_kerr = float(g_kerr)
        self.flux_theta = float(flux_theta)
        self.gamma_dissip = float(gamma_dissip)
        self.retroaction = float(retroaction)
        self.coeff_courbure = float(coeff_courbure)
        self.renormalize_l2 = bool(
            renormalize_l2
        )
        self.bandwidth_init = float(
            bandwidth_init
        )
        self.rng = np.random.default_rng(
            seed
        )

        self.x = np.linspace(
            0.0,
            self.length,
            self.n_x,
            endpoint=False,
        )

        self.a_field = np.full(
            self.n_x,
            self.flux_theta / self.length,
            dtype=float,
        )

        self.u_link = np.exp(
            -1j
            * self.a_field
            * self.dx
        )

        x_rel = (
            self.x
            - 0.5 * self.length
        ) / max(
            self.bandwidth_init,
            1e-6,
        )

        psi0 = (
            np.exp(
                -0.5 * x_rel ** 2
            )
            * np.exp(
                1j * 0.5 * x_rel
            )
        )

        norm0 = math.sqrt(
            float(
                np.sum(
                    np.abs(psi0) ** 2
                )
            )
            * self.dx
        )

        self.psi = psi0 / max(
            norm0,
            1e-12,
        )

        self.pi_ref = (
            self._compute_flux_observable(
                self.psi,
                self.u_link,
            )
        )

    def _compute_flux_observable(
        self,
        psi,
        u_link,
    ):
        d_cov = (
            u_link
            * np.roll(psi, -1)
            - psi
        ) / self.dx

        return float(
            np.sum(
                np.imag(
                    np.conj(psi)
                    * d_cov
                )
            )
            * self.dx
        )

    def courbure_densite_lissee(
        self,
        rho,
    ):
        """
        Dérivée seconde spatiale lissée de la densité.

        Cette quantité n'est pas assimilable à une courbure relativiste.
        """
        rho_s = _periodic_smooth(
            rho,
            passes=2,
        )

        return (
            np.roll(rho_s, -1)
            - 2.0 * rho_s
            + np.roll(rho_s, 1)
        ) / (
            self.dx ** 2
        )

    def diagnostic_energie_modele(
        self,
        psi=None,
        u_link=None,
    ):
        """
        Fonctionnelle diagnostique adimensionnelle du modèle numérique.

        Ne représente ni une énergie physique, ni une énergie biologique,
        ni une énergie de la conscience.
        """
        p = (
            self.psi
            if psi is None
            else psi
        )

        u = (
            self.u_link
            if u_link is None
            else u_link
        )

        rho = np.abs(p) ** 2

        d_cov = (
            u
            * np.roll(p, -1)
            - p
        ) / self.dx

        e_kin = 0.5 * float(
            np.sum(
                np.abs(d_cov) ** 2
            )
            * self.dx
        )

        e_int = 0.5 * self.g_kerr * float(
            np.sum(
                rho ** 2
            )
            * self.dx
        )

        curv = self.courbure_densite_lissee(
            rho
        )

        e_curv = 0.5 * self.coeff_courbure * float(
            np.sum(
                curv * rho
            )
            * self.dx
        )

        return e_kin + e_int + e_curv

    def extract_candidate_features(
        self,
        psi=None,
        u_link=None,
    ):
        """
        Extrait le vecteur candidat q(t), invariant de jauge :

        - ipr_sans_dimension
        - coherence_hilbertienne
        - holonomie_mod_2pi
        - courbure_integrale
        - energie_modele_diag
        """
        p = (
            self.psi
            if psi is None
            else psi
        )

        u = (
            self.u_link
            if u_link is None
            else u_link
        )

        rho = np.abs(p) ** 2

        norm_sq = max(
            float(
                np.sum(rho)
                * self.dx
            ),
            1e-12,
        )

        ipr_raw = (
            float(
                np.sum(
                    rho ** 2
                )
                * self.dx
            )
            / (
                norm_sq ** 2
            )
        )

        ipr_dimless = (
            self.length
            * ipr_raw
        )

        num_coh = abs(
            complex(
                np.sum(
                    np.conj(p)
                    * u
                    * np.roll(p, -1)
                )
            )
        )

        den_coh = max(
            float(
                np.sum(rho)
            ),
            1e-12,
        )

        coh_hilbert = (
            num_coh
            / den_coh
        )

        hol_angle = (
            float(
                np.angle(
                    np.prod(u)
                )
            )
            % (
                2.0 * math.pi
            )
        )

        curv = self.courbure_densite_lissee(
            rho
        )

        curv_int = math.sqrt(
            float(
                np.sum(
                    curv ** 2
                )
                * self.dx
            )
        )

        e_diag = self.diagnostic_energie_modele(
            p,
            u,
        )

        return {
            "ipr_sans_dimension": float(
                ipr_dimless
            ),
            "coherence_hilbertienne": float(
                coh_hilbert
            ),
            "holonomie_mod_2pi": float(
                hol_angle
            ),
            "courbure_integrale": float(
                curv_int
            ),
            "energie_modele_diag": float(
                e_diag
            ),
        }

    def step(self):
        """
        Intégration d'un pas temporel.

        La partie conservative utilise un splitting de Strang avec un pas
        cinétique traité par Crank-Nicolson covariant.

        En régime dissipatif, la norme avant et après une éventuelle
        renormalisation sont rapportées séparément.
        """
        rho = np.abs(
            self.psi
        ) ** 2

        if self.retroaction != 0.0:
            pi_curr = (
                self._compute_flux_observable(
                    self.psi,
                    self.u_link,
                )
            )

            ratio = (
                pi_curr
                / self.pi_ref
                if abs(self.pi_ref) > 1e-12
                else 1.0
            )

            g_eff = self.g_kerr * (
                1.0
                + self.retroaction
                * (ratio - 1.0)
            )
        else:
            g_eff = self.g_kerr

        curv = self.courbure_densite_lissee(
            rho
        )

        v_eff = (
            g_eff * rho
            + self.coeff_courbure * curv
        )

        psi_half = (
            self.psi
            * np.exp(
                -0.5j
                * v_eff
                * self.dt
            )
        )

        eye = np.eye(
            self.n_x,
            dtype=complex,
        )

        lap_mat = np.zeros(
            (
                self.n_x,
                self.n_x,
            ),
            dtype=complex,
        )

        for j in range(
            self.n_x
        ):
            jp = (
                j + 1
            ) % self.n_x

            jm = (
                j - 1
            ) % self.n_x

            lap_mat[
                j,
                j,
            ] = (
                -2.0
                / (
                    self.dx ** 2
                )
            )

            lap_mat[
                j,
                jp,
            ] = (
                self.u_link[j]
                / (
                    self.dx ** 2
                )
            )

            lap_mat[
                j,
                jm,
            ] = (
                np.conj(
                    self.u_link[jm]
                )
                / (
                    self.dx ** 2
                )
            )

        alpha = (
            0.25j
            * self.dt
        )

        rhs = (
            eye
            - alpha * lap_mat
        ) @ psi_half

        psi_kin = np.linalg.solve(
            eye
            + alpha * lap_mat,
            rhs,
        )

        rho_kin = (
            np.abs(psi_kin)
            ** 2
        )

        v_eff_2 = (
            g_eff * rho_kin
            + self.coeff_courbure
            * self.courbure_densite_lissee(
                rho_kin
            )
        )

        psi_next = (
            psi_kin
            * np.exp(
                -0.5j
                * v_eff_2
                * self.dt
            )
        )

        if self.gamma_dissip > 0.0:
            psi_next = (
                psi_next
                * math.exp(
                    -self.gamma_dissip
                    * self.dt
                )
            )

        norm_before = math.sqrt(
            float(
                np.sum(
                    np.abs(psi_next)
                    ** 2
                )
            )
            * self.dx
        )

        if (
            self.gamma_dissip > 0.0
            and self.renormalize_l2
        ):
            psi_next = (
                psi_next
                / max(
                    norm_before,
                    1e-12,
                )
            )

        norm_after = math.sqrt(
            float(
                np.sum(
                    np.abs(psi_next)
                    ** 2
                )
            )
            * self.dx
        )

        self.psi = psi_next

        feats = (
            self.extract_candidate_features()
        )

        feats[
            "norme_L2_avant_renorm"
        ] = float(
            norm_before
        )

        feats[
            "norme_L2_apres_renorm"
        ] = float(
            norm_after
        )

        feats[
            "regime"
        ] = (
            "dissipatif"
            if self.gamma_dissip > 0.0
            else "unitaire"
        )

        return feats

    def run(
        self,
        n_steps=100,
    ):
        return [
            self.step()
            for _ in range(
                int(n_steps)
            )
        ]


def verify_gauge_invariance(
    model,
    seed=123,
):
    """
    Vérifie l'invariance de q(t) sous transformation de jauge locale :

        psi'_j = exp(i chi_j) psi_j
        U'_j = exp(i chi_j) U_j exp(-i chi_{j+1})
    """
    rng = np.random.default_rng(
        seed
    )

    chi = rng.uniform(
        -math.pi,
        math.pi,
        size=model.n_x,
    )

    phase_rot = np.exp(
        1j * chi
    )

    psi_prime = (
        phase_rot
        * model.psi
    )

    u_prime = (
        phase_rot
        * model.u_link
        * np.conj(
            np.roll(
                phase_rot,
                -1,
            )
        )
    )

    q_orig = (
        model.extract_candidate_features(
            model.psi,
            model.u_link,
        )
    )

    q_trans = (
        model.extract_candidate_features(
            psi_prime,
            u_prime,
        )
    )

    max_diff = max(
        abs(
            q_orig[key]
            - q_trans[key]
        )
        for key in q_orig
    )

    raw_phase_shift = abs(
        float(
            np.mean(
                np.angle(
                    model.psi
                )
            )
        )
        - float(
            np.mean(
                np.angle(
                    psi_prime
                )
            )
        )
    )

    return {
        "gauge_invariant_pass": bool(
            max_diff < 1e-11
        ),
        "max_feature_drift": float(
            max_diff
        ),
        "non_covariant_raw_phase_drift": round(
            raw_phase_shift,
            4,
        ),
    }


def verify_numerical_convergence(
    resolutions=(64, 128),
    dts=(0.005, 0.0025),
    total_time=0.25,
):
    """
    Compare la dérive de la norme L2, de l'énergie diagnostique et de q(t)
    sous raffinement de grille et de pas temporel.
    """
    runs = {}

    for nx in resolutions:
        for dt in dts:
            n_steps = max(
                int(
                    round(
                        total_time
                        / dt
                    )
                ),
                1,
            )

            model = (
                ModeleHolotelegraphiqueExploratoire(
                    n_x=nx,
                    dt=dt,
                    gamma_dissip=0.0,
                    retroaction=0.0,
                )
            )

            trajectory = model.run(
                n_steps=n_steps
            )

            last = trajectory[-1]

            runs[
                f"nx{nx}_dt{dt}"
            ] = {
                "norme_L2_finale": round(
                    last[
                        "norme_L2_apres_renorm"
                    ],
                    10,
                ),
                "ipr_sans_dimension": round(
                    last[
                        "ipr_sans_dimension"
                    ],
                    6,
                ),
                "coherence_hilbertienne": round(
                    last[
                        "coherence_hilbertienne"
                    ],
                    6,
                ),
                "energie_modele_diag": round(
                    last[
                        "energie_modele_diag"
                    ],
                    6,
                ),
            }

    return runs


if __name__ == "__main__":
    model = (
        ModeleHolotelegraphiqueExploratoire(
            n_x=128,
            dt=0.005,
            gamma_dissip=0.0,
            retroaction=0.0,
        )
    )

    model.run(
        n_steps=40
    )

    audit = {
        "statut_epistemique": (
            "simulation_exploratoire_non_empirique"
        ),
        "controle_invariance_jauge": (
            verify_gauge_invariance(
                model
            )
        ),
        "controle_convergence": (
            verify_numerical_convergence()
        ),
        "dernier_vecteur_q": {
            key: round(
                value,
                6,
            )
            for key, value in (
                model.extract_candidate_features()
            ).items()
        },
    }

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )
