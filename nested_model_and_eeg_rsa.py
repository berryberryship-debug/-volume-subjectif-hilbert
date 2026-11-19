#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Couche théorique exploratoire : génération de variables candidates q(t)
à partir d'un champ complexe discret sur un réseau à jauge covariante.

La holonomie est un paramètre de configuration. Elle n'est pas incluse
dans le vecteur de prédicteurs individuels q(t).

Le module ne constitue :

- ni une mesure de la conscience ;
- ni une mesure d'énergie biologique ;
- ni une mesure EEG/MEG ;
- ni une courbure relativiste ;
- ni une preuve physique concernant l'expérience vécue.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional

import numpy as np


__all__ = [
    "ModeleHolotelegraphiqueExploratoire",
    "verify_gauge_invariance",
    "verify_numerical_convergence",
]


Q_FEATURE_NAMES = (
    "ipr_sans_dimension",
    "coherence_hilbertienne",
    "courbure_integrale",
    "energie_modele_diag",
)


def _periodic_smooth(
    arr: np.ndarray,
    passes: int = 2,
) -> np.ndarray:
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

    passes = int(
        passes
    )

    if passes < 0:
        raise ValueError(
            "passes doit être non négatif."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "arr contient des valeurs non finies."
        )

    result = values.copy()

    for _ in range(
        passes
    ):
        result = (
            0.25
            * np.roll(
                result,
                1,
            )
            + 0.50
            * result
            + 0.25
            * np.roll(
                result,
                -1,
            )
        )

    return result


def _angular_distance(
    angle_a: float,
    angle_b: float,
) -> float:
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

    Le vecteur dynamique q(t) contient uniquement :

        - ipr_sans_dimension
        - coherence_hilbertienne
        - courbure_integrale
        - energie_modele_diag

    La holonomie est conservée séparément dans les métadonnées de
    configuration et ne constitue pas un prédicteur individuel.
    """

    def __init__(
        self,
        n_x: int = 128,
        length: float = 10.0,
        dt: float = 0.005,
        g_kerr: float = 0.5,
        flux_theta: float = 0.4,
        gamma_dissip: float = 0.0,
        retroaction: float = 0.0,
        coeff_courbure: float = 0.05,
        renormalize_l2: bool = False,
        bandwidth_init: float = 1.0,
        seed: int = 42,
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

        self._validate_parameters()

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

        if not np.allclose(
            self.lap_mat,
            self.lap_mat.conj().T,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError(
                "Le Laplacien covariant discret n'est pas hermitien."
            )

        self.cn_alpha = (
            0.25j
            * self.dt
        )

        self.cn_left = (
            self.eye
            + self.cn_alpha
            * self.lap_mat
        )

        self.cn_right = (
            self.eye
            - self.cn_alpha
            * self.lap_mat
        )

        self._initialize_wavepacket()

        self.pi_ref = (
            self._compute_flux_observable(
                self.psi,
                self.u_link,
            )
        )

    def _validate_parameters(
        self,
    ) -> None:
        """
        Vérifie les paramètres numériques de configuration.
        """
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

        if self.gamma_dissip < 0.0:
            raise ValueError(
                "gamma_dissip doit être non négatif."
            )

        if self.bandwidth_init <= 0.0:
            raise ValueError(
                "bandwidth_init doit être strictement positif."
            )

        parameters = {
            "g_kerr": self.g_kerr,
            "flux_theta": self.flux_theta,
            "retroaction": self.retroaction,
            "coeff_courbure": self.coeff_courbure,
        }

        for name, value in parameters.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} doit être fini."
                )

    def _initialize_wavepacket(
        self,
    ) -> None:
        """
        Initialise puis normalise le paquet d'onde complexe.
        """
        x_relative = (
            self.x
            - 0.5
            * self.length
        ) / self.bandwidth_init

        psi_initial = (
            np.exp(
                -0.5
                * x_relative
                ** 2
            )
            * np.exp(
                0.5j
                * x_relative
            )
        )

        norm_initial = math.sqrt(
            float(
                np.sum(
                    np.abs(
                        psi_initial
                    )
                    ** 2
                )
            )
            * self.dx
        )

        if norm_initial <= 1e-15:
            raise FloatingPointError(
                "La norme initiale du champ est nulle."
            )

        self.psi = (
            psi_initial
            / norm_initial
        )

    def _build_covariant_laplacian_matrix(
        self,
        u_link: np.ndarray,
    ) -> np.ndarray:
        """
        Construit le Laplacien covariant discret :

            (Delta_A psi)_j =
                (U_j psi_{j+1}
                 - 2 psi_j
                 + conj(U_{j-1}) psi_{j-1})
                / dx²
        """
        links = self._validate_links(
            u_link
        )

        laplacian = np.zeros(
            (
                self.n_x,
                self.n_x,
            ),
            dtype=complex,
        )

        denominator = (
            self.dx
            ** 2
        )

        for j in range(
            self.n_x
        ):
            j_plus = (
                j + 1
            ) % self.n_x

            j_minus = (
                j - 1
            ) % self.n_x

            laplacian[
                j,
                j,
            ] = (
                -2.0
                / denominator
            )

            laplacian[
                j,
                j_plus,
            ] = (
                links[j]
                / denominator
            )

            laplacian[
                j,
                j_minus,
            ] = (
                np.conj(
                    links[j_minus]
                )
                / denominator
            )

        return laplacian

    def _validate_field(
        self,
        psi: np.ndarray,
        name: str = "psi",
    ) -> np.ndarray:
        """
        Vérifie un champ complexe de forme (n_x,).
        """
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
        u_link: np.ndarray,
    ) -> np.ndarray:
        """
        Vérifie un tableau de liens complexes de forme (n_x,).
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

        return links

    def _compute_flux_observable(
        self,
        psi: np.ndarray,
        u_link: np.ndarray,
    ) -> float:
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

        covariant_derivative = (
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
                    np.conj(
                        field
                    )
                    * covariant_derivative
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
        rho: np.ndarray,
    ) -> np.ndarray:
        """
        Calcule une dérivée seconde spatiale lissée de la densité.

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

        smoothed_density = _periodic_smooth(
            density,
            passes=2,
        )

        curvature = (
            np.roll(
                smoothed_density,
                -1,
            )
            - 2.0
            * smoothed_density
            + np.roll(
                smoothed_density,
                1,
            )
        ) / (
            self.dx
            ** 2
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
        psi: Optional[np.ndarray] = None,
        u_link: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calcule une fonctionnelle diagnostique adimensionnelle.

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

        density = (
            np.abs(
                field
            )
            ** 2
        )

        covariant_derivative = (
            links
            * np.roll(
                field,
                -1,
            )
            - field
        ) / self.dx

        kinetic_term = 0.5 * float(
            np.sum(
                np.abs(
                    covariant_derivative
                )
                ** 2
            )
            * self.dx
        )

        interaction_term = (
            0.5
            * self.g_kerr
            * float(
                np.sum(
                    density
                    ** 2
                )
                * self.dx
            )
        )

        curvature = (
            self.courbure_densite_lissee(
                density
            )
        )

        curvature_term = (
            0.5
            * self.coeff_courbure
            * float(
                np.sum(
                    curvature
                    * density
                )
                * self.dx
            )
        )

        energy = (
            kinetic_term
            + interaction_term
            + curvature_term
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
        u_link: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calcule la holonomie de configuration modulo 2*pi.

        Avec :

            U_j = exp(-i A_j dx)

        la valeur attendue pour le champ uniforme est :

            -flux_theta modulo 2*pi.
        """
        links = (
            self.u_link
            if u_link is None
            else self._validate_links(
                u_link
            )
        )

        holonomy = float(
            np.angle(
                np.prod(
                    links
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
    ) -> Dict[str, object]:
        """
        Retourne les paramètres de configuration.

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
            "q_feature_names": list(
                Q_FEATURE_NAMES
            ),
            "holonomie_mod_2pi": float(
                self.holonomie_mod_2pi()
            ),
            "holonomie_role": (
                "parametre_de_configuration"
            ),
            "model_status": (
                "simulation_exploratoire_non_empirique"
            ),
        }

    def extract_candidate_features(
        self,
        psi: Optional[np.ndarray] = None,
        u_link: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Extrait les variables dynamiques candidates q(t).

        La holonomie n'est pas incluse dans ce dictionnaire.

        Retourne :

        - ipr_sans_dimension ;
        - coherence_hilbertienne ;
        - courbure_integrale ;
        - energie_modele_diag.
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

        density = (
            np.abs(
                field
            )
            ** 2
        )

        norm_squared = max(
            float(
                np.sum(
                    density
                )
                * self.dx
            ),
            1e-12,
        )

        ipr_raw = (
            float(
                np.sum(
                    density
                    ** 2
                )
                * self.dx
            )
            / (
                norm_squared
                ** 2
            )
        )

        ipr_dimensionless = (
            self.length
            * ipr_raw
        )

        coherence_numerator = abs(
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

        coherence_denominator = max(
            float(
                np.sum(
                    density
                )
            ),
            1e-12,
        )

        coherence = (
            coherence_numerator
            / coherence_denominator
        )

        curvature = (
            self.courbure_densite_lissee(
                density
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

        diagnostic_energy = (
            self.diagnostic_energie_modele(
                field,
                links,
            )
        )

        features = {
            "ipr_sans_dimension": float(
                ipr_dimensionless
            ),
            "coherence_hilbertienne": float(
                coherence
            ),
            "courbure_integrale": float(
                curvature_integral
            ),
            "energie_modele_diag": float(
                diagnostic_energy
            ),
        }

        if set(
            features.keys()
        ) != set(
            Q_FEATURE_NAMES
        ):
            raise RuntimeError(
                "Les caractéristiques q(t) ne correspondent pas à "
                "Q_FEATURE_NAMES."
            )

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

    def step(
        self,
    ) -> Dict[str, object]:
        """
        Intègre un pas temporel.

        La partie potentielle est traitée par deux demi-pas.
        La partie cinétique est traitée par Crank-Nicolson sur le
        Laplacien covariant préassemblé.
        """
        density = (
            np.abs(
                self.psi
            )
            ** 2
        )

        if self.retroaction != 0.0:
            current_flux = (
                self._compute_flux_observable(
                    self.psi,
                    self.u_link,
                )
            )

            if abs(
                self.pi_ref
            ) > 1e-12:
                flux_ratio = (
                    current_flux
                    / self.pi_ref
                )
            else:
                flux_ratio = 1.0

            effective_g = (
                self.g_kerr
                * (
                    1.0
                    + self.retroaction
                    * (
                        flux_ratio
                        - 1.0
                    )
                )
            )
        else:
            effective_g = self.g_kerr

        curvature = (
            self.courbure_densite_lissee(
                density
            )
        )

        effective_potential = (
            effective_g
            * density
            + self.coeff_courbure
            * curvature
        )

        psi_half = (
            self.psi
            * np.exp(
                -0.5j
                * effective_potential
                * self.dt
            )
        )

        rhs = (
            self.cn_right
            @ psi_half
        )

        psi_kinetic = np.linalg.solve(
            self.cn_left,
            rhs,
        )

        density_kinetic = (
            np.abs(
                psi_kinetic
            )
            ** 2
        )

        curvature_kinetic = (
            self.courbure_densite_lissee(
                density_kinetic
            )
        )

        second_potential = (
            effective_g
            * density_kinetic
            + self.coeff_courbure
            * curvature_kinetic
        )

        psi_next = (
            psi_kinetic
            * np.exp(
                -0.5j
                * second_potential
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

        norm_before_renormalization = math.sqrt(
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
                    norm_before_renormalization,
                    1e-12,
                )
            )

        norm_after_renormalization = math.sqrt(
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
            not np.isfinite(
                norm_before_renormalization
            )
            or not np.isfinite(
                norm_after_renormalization
            )
        ):
            raise FloatingPointError(
                "La norme L2 devient non finie."
            )

        self.psi = psi_next

        features = (
            self.extract_candidate_features()
        )

        result: Dict[str, object] = {
            **features,
            "norme_L2_avant_renorm": float(
                norm_before_renormalization
            ),
            "norme_L2_finale": float(
                norm_after_renormalization
            ),
            "norme_L2_apres_renorm": float(
                norm_after_renormalization
            ),
            "regime": (
                "dissipatif"
                if self.gamma_dissip > 0.0
                else "unitaire"
            ),
        }

        return result

    def run(
        self,
        n_steps: int = 100,
    ):
        """
        Exécute plusieurs pas temporels.
        """
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
    model: ModeleHolotelegraphiqueExploratoire,
    seed: int = 123,
) -> Dict[str, object]:
    """
    Vérifie l'invariance des variables dynamiques sous :

        psi'_j = exp(i chi_j) psi_j
        U'_j = exp(i chi_j) U_j exp(-i chi_{j+1})

    La holonomie est contrôlée séparément.
    """
    rng = np.random.default_rng(
        int(seed)
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
        name: abs(
            q_original[name]
            - q_transformed[name]
        )
        for name in Q_FEATURE_NAMES
    }

    maximum_feature_drift = max(
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

    raw_phase_drift = abs(
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
            maximum_feature_drift
            < 1e-11
            and holonomy_drift
            < 1e-11
        ),
        "feature_drifts": feature_drift,
        "max_feature_drift": float(
            maximum_feature_drift
        ),
        "holonomy_drift": float(
            holonomy_drift
        ),
        "non_covariant_raw_phase_drift": float(
            raw_phase_drift
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
    total_time: float = 0.25,
    bandwidths=(1.0,),
) -> Dict[str, object]:
    """
    Compare les sorties finales pour plusieurs configurations numériques.

    Ce contrôle fournit une analyse de sensibilité numérique. Il ne remplace
    pas une étude d'erreur contre une solution analytique.
    """
    if total_time <= 0.0:
        raise ValueError(
            "total_time doit être strictement positif."
        )

    runs: Dict[str, object] = {}

    for resolution in resolutions:
        for dt in dts:
            for bandwidth in bandwidths:
                resolution = int(
                    resolution
                )

                dt = float(
                    dt
                )

                bandwidth = float(
                    bandwidth
                )

                if resolution < 4:
                    raise ValueError(
                        "Chaque résolution doit être au moins égale à 4."
                    )

                if dt <= 0.0:
                    raise ValueError(
                        "Chaque pas temporel doit être strictement positif."
                    )

                if bandwidth <= 0.0:
                    raise ValueError(
                        "Chaque largeur initiale doit être strictement positive."
                    )

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
                        n_x=resolution,
                        dt=dt,
                        bandwidth_init=bandwidth,
                        gamma_dissip=0.0,
                        retroaction=0.0,
                    )
                )

                trajectory = model.run(
                    n_steps=n_steps
                )

                final_state = trajectory[-1]

                key = (
                    f"nx{resolution}_"
                    f"dt{dt}_"
                    f"bw{bandwidth}"
                )

                runs[key] = {
                    "configuration": {
                        "n_x": resolution,
                        "dt": dt,
                        "bandwidth_init": bandwidth,
                        "total_time": float(
                            total_time
                        ),
                        "n_steps": n_steps,
                    },
                    "norme_L2_finale": round(
                        float(
                            final_state[
                                "norme_L2_finale"
                            ]
                        ),
                        10,
                    ),
                    "ipr_sans_dimension": round(
                        float(
                            final_state[
                                "ipr_sans_dimension"
                            ]
                        ),
                        6,
                    ),
                    "coherence_hilbertienne": round(
                        float(
                            final_state[
                                "coherence_hilbertienne"
                            ]
                        ),
                        6,
                    ),
                    "courbure_integrale": round(
                        float(
                            final_state[
                                "courbure_integrale"
                            ]
                        ),
                        6,
                    ),
                    "energie_modele_diag": round(
                        float(
                            final_state[
                                "energie_modele_diag"
                            ]
                        ),
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
