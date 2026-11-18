#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Couche théorique exploratoire : génération de variables candidates q(t)
à partir d'un champ complexe discret sur un réseau à jauge covariante.

Ce module ne constitue :

- ni une mesure de la conscience ;
- ni une mesure d'énergie biologique ;
- ni une courbure de l'espace-temps ;
- ni une mesure directe de V_psi ;
- ni une preuve physique concernant l'expérience de Young ou l'observateur.

La holonomie est traitée comme un paramètre de configuration du modèle.
Elle n'est pas incluse dans le vecteur de prédicteurs individuels q(t).
"""

import json
import math

import numpy as np


def _periodic_smooth(
    arr,
    passes=2,
):
    """
    Lissage périodique d'un tableau réel.
    """
    values = np.asarray(
        arr,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "arr doit être un tableau unidimensionnel."
        )

    if int(passes) < 0:
        raise ValueError(
            "passes doit être non négatif."
        )

    out = values.copy()

    for _ in range(
        int(passes)
    ):
        out = (
            0.25 * np.roll(
                out,
                1,
            )
            + 0.50 * out
            + 0.25 * np.roll(
                out,
                -1,
            )
        )

    return out


def _angular_distance(
    angle_a,
    angle_b,
):
    """
    Distance circulaire entre deux angles.
    """
    return abs(
        float(
            np.angle(
                np.exp(
                    1j
                    * (
                        float(angle_a)
                        - float(angle_b)
                    )
                )
            )
        )
    )


class ModeleHolotelegraphiqueExploratoire:
    """
    Modèle exploratoire d'un champ complexe discret sur un réseau périodique.

    Les caractéristiques dynamiques candidates q(t) sont :

        - ipr_sans_dimension
        - coherence_hilbertienne
        - courbure_integrale
        - energie_modele_diag

    La holonomie est une propriété de configuration et est accessible
    séparément par get_configuration_metadata().
    """

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
        self.n_x = int(
            n_x
        )
        self.length = float(
            length
        )
        self.dt = float(
            dt
        )
        self.g_kerr = float(
            g_kerr
        )
        self.flux_theta = float(
            flux_theta
        )
        self.gamma_dissip = float(
            gamma_dissip
        )
        self.retroaction = float(
            retroaction
        )
        self.coeff_courbure = float(
            coeff_courbure
        )
        self.renormalize_l2 = bool(
            renormalize_l2
        )
        self.bandwidth_init = float(
            bandwidth_init
        )
        self.seed = int(
            seed
        )

        if self.n_x < 4:
            raise ValueError(
                "n_x doit être au moins égal à 4."
            )

        if self.length <= 0.0:
            raise ValueError(
                "length doit être strictement positif."
            )

        if self.dt <= 0.0:
            raise ValueError(
                "dt doit être strictement positif."
            )

        if self.bandwidth_init <= 0.0:
            raise ValueError(
                "bandwidth_init doit être strictement positif."
            )

        if self.gamma_dissip < 0.0:
            raise ValueError(
                "gamma_dissip doit être non négatif."
            )

        scalar_parameters = {
            "g_kerr": self.g_kerr,
            "flux_theta": self.flux_theta,
            "retroaction": self.retroaction,
            "coeff_courbure": self.coeff_courbure,
        }

        for name, value in scalar_parameters.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} doit être fini."
                )

        self.dx = (
            self.length
            / self.n_x
        )

        self.rng = np.random.default_rng(
            self.seed
        )

        self.x = np.linspace(
            0.0,
            self.length,
            self.n_x,
            endpoint=False,
        )

        # Convention de jauge :
        #
        #     A_j = flux_theta / length
        #     U_j = exp(-i A_j dx)
        #
        # Par conséquent :
        #
        #     prod_j U_j = exp(-i flux_theta)
        #
        # et la holonomie vaut :
        #
        #     -flux_theta modulo 2*pi.
        self.a_field = np.full(
            self.n_x,
            self.flux_theta
            / self.length,
            dtype=float,
        )

        self.u_link = np.exp(
            -1j
            * self.a_field
            * self.dx
        )

        self.eye = np.eye(
            self.n_x,
            dtype=complex,
        )

        self.lap_mat = (
            self._build_covariant_laplacian_matrix(
                self.u_link
            )
        )

        self._initialize_wavepacket()

        self.pi_ref = (
            self._compute_flux_observable(
                self.psi,
                self.u_link,
            )
        )

    def _initialize_wavepacket(self):
        """
        Initialise puis normalise le paquet d'onde complexe.
        """
        x_rel = (
            self.x
            - 0.5 * self.length
        ) / self.bandwidth_init

        psi0 = (
            np.exp(
                -0.5
                * x_rel ** 2
            )
            * np.exp(
                1j
                * 0.5
                * x_rel
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

        self.psi = (
            psi0
            / max(
                norm0,
                1e-12,
            )
        )

    def _build_covariant_laplacian_matrix(
        self,
        u_link,
    ):
        """
        Construit le Laplacien covariant discret :

            (Delta_A psi)_j =
                (U_j psi_{j+1}
                 - 2 psi_j
                 + conj(U_{j-1}) psi_{j-1})
                / dx^2
        """
        links = np.asarray(
            u_link,
            dtype=complex,
        )

        if links.shape != (
            self.n_x,
        ):
            raise ValueError(
                "u_link doit avoir la forme (n_x,)."
            )

        if not np.all(
            np.isfinite(
                links.real
            )
        ) or not np.all(
            np.isfinite(
                links.imag
            )
        ):
            raise ValueError(
                "u_link contient des valeurs non finies."
            )

        lap_mat = np.zeros(
            (
                self.n_x,
                self.n_x,
            ),
            dtype=complex,
        )

        denominator = (
            self.dx ** 2
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
                / denominator
            )

            lap_mat[
                j,
                jp,
            ] = (
                links[j]
                / denominator
            )

            lap_mat[
                j,
                jm,
            ] = (
                np.conj(
                    links[jm]
                )
                / denominator
            )

        return lap_mat

    def _validate_field(
        self,
        psi,
        name="psi",
    ):
        field = np.asarray(
            psi,
            dtype=complex,
        )

        if field.shape != (
            self.n_x,
        ):
            raise ValueError(
                f"{name} doit avoir la forme (n_x,)."
            )

        if not np.all(
            np.isfinite(
                field.real
            )
        ) or not np.all(
            np.isfinite(
                field.imag
            )
        ):
            raise ValueError(
                f"{name} contient des valeurs non finies."
            )

        return field

    def _validate_links(
        self,
        u_link,
    ):
        links = np.asarray(
            u_link,
            dtype=complex,
        )

        if links.shape != (
            self.n_x,
        ):
            raise ValueError(
                "u_link doit avoir la forme (n_x,)."
            )

        if not np.all(
            np.isfinite(
                links.real
            )
        ) or not np.all(
            np.isfinite(
                links.imag
            )
        ):
            raise ValueError(
                "u_link contient des valeurs non finies."
            )

        return links

    def _compute_flux_observable(
        self,
        psi,
        u_link,
    ):
        """
        Observable de flux discrète basée sur la dérivée covariante avant.
        """
        field = self._validate_field(
            psi,
            name="psi",
        )

        links = self._validate_links(
            u_link
        )

        d_cov = (
            links
            * np.roll(
                field,
                -1,
            )
            - field
        ) / self.dx

        value = float(
            np.sum(
                np.imag(
                    np.conj(field)
                    * d_cov
                )
            )
            * self.dx
        )

        if not np.isfinite(value):
            raise FloatingPointError(
                "L'observable de flux est non finie."
            )

        return value

    def courbure_densite_lissee(
        self,
        rho,
    ):
        """
        Dérivée seconde spatiale lissée de la densité.

        Cette quantité est une dérivée seconde discrète de densité.
        Elle n'est pas une courbure relativiste.
        """
        density = np.asarray(
            rho,
            dtype=float,
        )

        if density.shape != (
            self.n_x,
        ):
            raise ValueError(
                "rho doit avoir la forme (n_x,)."
            )

        if not np.all(
            np.isfinite(
                density
            )
        ):
            raise ValueError(
                "rho contient des valeurs non finies."
            )

        rho_s = _periodic_smooth(
            density,
            passes=2,
        )

        curvature = (
            np.roll(
                rho_s,
                -1,
            )
            - 2.0 * rho_s
            + np.roll(
                rho_s,
                1,
            )
        ) / (
            self.dx ** 2
        )

        if not np.all(
            np.isfinite(
                curvature
            )
        ):
            raise FloatingPointError(
                "La dérivée seconde de densité est non finie."
            )

        return curvature

    def diagnostic_energie_modele(
        self,
        psi=None,
        u_link=None,
    ):
        """
        Fonctionnelle diagnostique adimensionnelle.

        Cette quantité ne représente ni une énergie physique,
        ni une énergie biologique,
        ni une énergie de la conscience.
        """
        field = (
            self.psi
            if psi is None
            else self._validate_field(
                psi,
                name="psi",
            )
        )

        links = (
            self.u_link
            if u_link is None
            else self._validate_links(
                u_link
            )
        )

        rho = (
            np.abs(
                field
            )
            ** 2
        )

        d_cov = (
            links
            * np.roll(
                field,
                -1,
            )
            - field
        ) / self.dx

        e_kin = 0.5 * float(
            np.sum(
                np.abs(
                    d_cov
                )
                ** 2
            )
            * self.dx
        )

        e_int = 0.5 * self.g_kerr * float(
            np.sum(
                rho
                ** 2
            )
            * self.dx
        )

        curvature = (
            self.courbure_densite_lissee(
                rho
            )
        )

        e_curv = 0.5 * self.coeff_courbure * float(
            np.sum(
                curvature
                * rho
            )
            * self.dx
        )

        energy = (
            e_kin
            + e_int
            + e_curv
        )

        if not np.isfinite(
            energy
        ):
            raise FloatingPointError(
                "L'énergie diagnostique est non finie."
            )

        return float(
            energy
        )

    def holonomie_mod_2pi(
        self,
        u_link=None,
    ):
        """
        Calcule la holonomie de configuration modulo 2*pi.

        Avec U_j = exp(-i A_j dx), la valeur attendue pour le champ
        uniforme est -flux_theta modulo 2*pi.
        """
        links = (
            self.u_link
            if u_link is None
            else self._validate_links(
                u_link
            )
        )

        holonomy = (
            float(
                np.angle(
                    np.prod(
                        links
                    )
                )
            )
            % (
                2.0
                * math.pi
            )
        )

        if not np.isfinite(
            holonomy
        ):
            raise FloatingPointError(
                "La holonomie est non finie."
            )

        return holonomy

    def get_configuration_metadata(
        self,
    ):
        """
        Retourne les paramètres de configuration du modèle.

        La holonomie est volontairement exclue du vecteur q(t).
        """
        return {
            "n_x": int(
                self.n_x
            ),
            "length": float(
                self.length
            ),
            "dx": float(
                self.dx
            ),
            "dt": float(
                self.dt
            ),
            "g_kerr": float(
                self.g_kerr
            ),
            "flux_theta": float(
                self.flux_theta
            ),
            "gamma_dissip": float(
                self.gamma_dissip
            ),
            "retroaction": float(
                self.retroaction
            ),
            "coeff_courbure": float(
                self.coeff_courbure
            ),
            "renormalize_l2": bool(
                self.renormalize_l2
            ),
            "bandwidth_init": float(
                self.bandwidth_init
            ),
            "seed": int(
                self.seed
            ),
            "pi_ref": float(
                self.pi_ref
            ),
            "holonomie_mod_2pi": float(
                self.holonomie_mod_2pi()
            ),
            "holonomie_role": (
                "parametre_de_configuration"
            ),
        }

    def extract_candidate_features(
        self,
        psi=None,
        u_link=None,
    ):
        """
        Extrait uniquement les variables dynamiques candidates q(t).

        La holonomie n'est pas incluse ici, car elle dépend de la
        configuration de jauge et non d'une observation individuelle.

        Retourne :

        - ipr_sans_dimension
        - coherence_hilbertienne
        - courbure_integrale
        - energie_modele_diag
        """
        field = (
            self.psi
            if psi is None
            else self._validate_field(
                psi,
                name="psi",
            )
        )

        links = (
            self.u_link
            if u_link is None
            else self._validate_links(
                u_link
            )
        )

        rho = (
            np.abs(
                field
            )
            ** 2
        )

        norm_sq = max(
            float(
                np.sum(
                    rho
                )
                * self.dx
            ),
            1e-12,
        )

        ipr_raw = (
            float(
                np.sum(
                    rho
                    ** 2
                )
                * self.dx
            )
            / (
                norm_sq
                ** 2
            )
        )

        ipr_dimless = (
            self.length
            * ipr_raw
        )

        numerator_coherence = abs(
            complex(
                np.sum(
                    np.conj(
                        field
                    )
                    * links
                    * np.roll(
                        field,
                        -1,
                    )
                )
            )
        )

        denominator_coherence = max(
            float(
                np.sum(
                    rho
                )
            ),
            1e-12,
        )

        coherence = (
            numerator_coherence
            / denominator_coherence
        )

        curvature = (
            self.courbure_densite_lissee(
                rho
            )
        )

        curvature_integral = math.sqrt(
            float(
                np.sum(
                    curvature
                    ** 2
                )
                * self.dx
            )
        )

        energy_diag = (
            self.diagnostic_energie_modele(
                field,
                links,
            )
        )

        features = {
            "ipr_sans_dimension": float(
                ipr_dimless
            ),
            "coherence_hilbertienne": float(
                coherence
            ),
            "courbure_integrale": float(
                curvature_integral
            ),
            "energie_modele_diag": float(
                energy_diag
            ),
        }

        if not all(
            np.isfinite(
                value
            )
            for value in features.values()
        ):
            raise FloatingPointError(
                "Une caractéristique q(t) est non finie."
            )

        return features

    def step(self):
        """
        Intègre un pas temporel.

        La partie potentielle est traitée par deux demi-pas.
        La partie cinétique est traitée par Crank-Nicolson sur le
        Laplacien covariant préassemblé.

        La convention de signe du terme cinétique est celle induite
        par la matrice lap_mat et le schéma ci-dessous.
        """
        rho = (
            np.abs(
                self.psi
            )
            ** 2
        )

        if self.retroaction != 0.0:
            pi_current = (
                self._compute_flux_observable(
                    self.psi,
                    self.u_link,
                )
            )

            if abs(
                self.pi_ref
            ) > 1e-12:
                ratio = (
                    pi_current
                    / self.pi_ref
                )
            else:
                ratio = 1.0

            g_eff = (
                self.g_kerr
                * (
                    1.0
                    + self.retroaction
                    * (
                        ratio
                        - 1.0
                    )
                )
            )
        else:
            g_eff = self.g_kerr

        curvature = (
            self.courbure_densite_lissee(
                rho
            )
        )

        v_eff = (
            g_eff
            * rho
            + self.coeff_courbure
            * curvature
        )

        psi_half = (
            self.psi
            * np.exp(
                -0.5j
                * v_eff
                * self.dt
            )
        )

        alpha = (
            0.25j
            * self.dt
        )

        rhs = (
            self.eye
            - alpha
            * self.lap_mat
        ) @ psi_half

        psi_kin = np.linalg.solve(
            self.eye
            + alpha
            * self.lap_mat,
            rhs,
        )

        rho_kin = (
            np.abs(
                psi_kin
            )
            ** 2
        )

        curvature_kin = (
            self.courbure_densite_lissee(
                rho_kin
            )
        )

        v_eff_2 = (
            g_eff
            * rho_kin
            + self.coeff_courbure
            * curvature_kin
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
                    np.abs(
                        psi_next
                    )
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
                    np.abs(
                        psi_next
                    )
                    ** 2
                )
            )
            * self.dx
        )

        if not np.isfinite(
            norm_before
        ) or not np.isfinite(
            norm_after
        ):
            raise FloatingPointError(
                "La norme L2 devient non finie."
            )

        self.psi = psi_next

        features = (
            self.extract_candidate_features()
        )

        features[
            "norme_L2_avant_renorm"
        ] = float(
            norm_before
        )

        features[
            "norme_L2_finale"
        ] = float(
            norm_after
        )

        # Alias conservé pour compatibilité avec l'ancienne interface.
        features[
            "norme_L2_apres_renorm"
        ] = float(
            norm_after
        )

        features[
            "regime"
        ] = (
            "dissipatif"
            if self.gamma_dissip > 0.0
            else "unitaire"
        )

        return features

    def run(
        self,
        n_steps=100,
    ):
        n_steps = int(
            n_steps
        )

        if n_steps < 0:
            raise ValueError(
                "n_steps doit être non négatif."
            )

        return [
            self.step()
            for _ in range(
                n_steps
            )
        ]


def verify_gauge_invariance(
    model,
    seed=123,
):
    """
    Vérifie l'invariance des variables dynamiques sous :

        psi'_j = exp(i chi_j) psi_j
        U'_j = exp(i chi_j) U_j exp(-i chi_{j+1})

    La holonomie est contrôlée séparément comme paramètre de configuration.
    """
    rng = np.random.default_rng(
        seed
    )

    chi = rng.uniform(
        -math.pi,
        math.pi,
        size=model.n_x,
    )

    phase_rotation = np.exp(
        1j
        * chi
    )

    psi_prime = (
        phase_rotation
        * model.psi
    )

    u_prime = (
        phase_rotation
        * model.u_link
        * np.conj(
            np.roll(
                phase_rotation,
                -1,
            )
        )
    )

    q_original = (
        model.extract_candidate_features(
            model.psi,
            model.u_link,
        )
    )

    q_transformed = (
        model.extract_candidate_features(
            psi_prime,
            u_prime,
        )
    )

    feature_drift = {
        key: abs(
            q_original[key]
            - q_transformed[key]
        )
        for key in q_original
    }

    max_feature_drift = max(
        feature_drift.values()
    )

    holonomy_original = (
        model.holonomie_mod_2pi(
            model.u_link
        )
    )

    holonomy_transformed = (
        model.holonomie_mod_2pi(
            u_prime
        )
    )

    holonomy_drift = (
        _angular_distance(
            holonomy_original,
            holonomy_transformed,
        )
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
            max_feature_drift
            < 1e-11
            and holonomy_drift
            < 1e-11
        ),
        "max_feature_drift": float(
            max_feature_drift
        ),
        "holonomy_drift": float(
            holonomy_drift
        ),
        "non_covariant_raw_phase_drift": round(
            raw_phase_shift,
            4,
        ),
        "holonomie_originale": float(
            holonomy_original
        ),
        "holonomie_transformee": float(
            holonomy_transformed
        ),
    }


def verify_numerical_convergence(
    resolutions=(64, 128),
    dts=(0.005, 0.0025),
    total_time=0.25,
    bandwidths=(1.0,),
):
    """
    Compare les sorties finales pour plusieurs résolutions spatiales,
    pas temporels et largeurs initiales.

    Ce contrôle fournit une analyse de sensibilité numérique.
    Il ne remplace pas une étude d'erreur contre une solution analytique.
    """
    runs = {}

    for nx in resolutions:
        for dt in dts:
            for bandwidth in bandwidths:
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
                        bandwidth_init=bandwidth,
                        gamma_dissip=0.0,
                        retroaction=0.0,
                    )
                )

                trajectory = model.run(
                    n_steps=n_steps
                )

                last = trajectory[-1]

                key = (
                    f"nx{nx}_"
                    f"dt{dt}_"
                    f"bw{bandwidth}"
                )

                runs[key] = {
                    "norme_L2_finale": round(
                        last[
                            "norme_L2_finale"
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
                    "courbure_integrale": round(
                        last[
                            "courbure_integrale"
                        ],
                        6,
                    ),
                    "energie_modele_diag": round(
                        last[
                            "energie_modele_diag"
                        ],
                        6,
                    ),
                    "holonomie_mod_2pi": round(
                        model.holonomie_mod_2pi(),
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
        "configuration": (
            model.get_configuration_metadata()
        ),
        "controle_invariance_jauge": (
            verify_gauge_invariance(
                model
            )
        ),
        "controle_convergence": (
            verify_numerical_convergence()
        ),
        "dernier_vecteur_q": (
            model.extract_candidate_features()
        ),
    }

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )
