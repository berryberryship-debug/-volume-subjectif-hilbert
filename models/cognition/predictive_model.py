# -*- coding: utf-8 -*-
"""
PREDICTIVE MODEL
===============

Prédiction des conséquences d'actions sur un état simulé.

Les coefficients sont des paramètres expérimentaux.
Ils ne représentent pas des constantes biologiques.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math


ACTIONS = (
    "fuir",
    "chercher",
    "maintenir",
)

STATE_KEYS = (
    "energy",
    "danger",
    "fatigue",
    "integrity",
    "injury",
)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class ModelState:
    energy: float = 1.0
    danger: float = 0.0
    fatigue: float = 0.0
    integrity: float = 1.0
    injury: float = 0.0

    def __post_init__(self) -> None:
        for key in STATE_KEYS:
            value = float(getattr(self, key))

            if not math.isfinite(value):
                raise ValueError(f"{key} doit être une valeur finie.")

            setattr(self, key, clamp(value))

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "ModelState":
        return cls(**{
            key: values.get(key, 0.0)
            for key in STATE_KEYS
        })

    def as_dict(self) -> Dict[str, float]:
        return {
            key: getattr(self, key)
            for key in STATE_KEYS
        }


@dataclass
class Prediction:
    action: str
    before: ModelState
    predicted_after: ModelState
    predicted_error: float = 0.0


@dataclass
class PredictiveModel:
    """
    Modèle de transition déterministe.

    Les coefficients sont partagés avec l'environnement de simulation.
    """

    flee_energy_cost: float = 0.08
    flee_danger_factor: float = 0.88
    flee_fatigue_gain: float = 0.10

    search_energy_cost: float = 0.12
    search_danger_factor: float = 1.03
    search_fatigue_gain: float = 0.06

    maintain_energy_cost: float = 0.02
    maintain_danger_factor: float = 1.00
    maintain_fatigue_gain: float = 0.01

    recovery_rate: float = 0.03

    def __post_init__(self) -> None:
        coefficients = (
            self.flee_energy_cost,
            self.flee_danger_factor,
            self.flee_fatigue_gain,
            self.search_energy_cost,
            self.search_danger_factor,
            self.search_fatigue_gain,
            self.maintain_energy_cost,
            self.maintain_danger_factor,
            self.maintain_fatigue_gain,
            self.recovery_rate,
        )

        if not all(math.isfinite(float(x)) for x in coefficients):
            raise ValueError("Tous les coefficients doivent être finis.")

        for cost in (
            self.flee_energy_cost,
            self.search_energy_cost,
            self.maintain_energy_cost,
            self.flee_fatigue_gain,
            self.search_fatigue_gain,
            self.maintain_fatigue_gain,
            self.recovery_rate,
        ):
            if cost < 0.0:
                raise ValueError("Les coûts et taux doivent être positifs.")

        for factor in (
            self.flee_danger_factor,
            self.search_danger_factor,
            self.maintain_danger_factor,
        ):
            if factor < 0.0:
                raise ValueError("Les facteurs de danger doivent être positifs.")

    def transition(
        self,
        state: ModelState,
        action: str,
    ) -> ModelState:
        """
        Applique une transition d'action sans événement externe.

        L'objet state d'entrée n'est jamais modifié.
        """

        if action not in ACTIONS:
            raise ValueError(
                f"Action inconnue : {action}. "
                f"Actions disponibles : {ACTIONS}"
            )

        energy = state.energy
        danger = state.danger
        fatigue = state.fatigue
        injury = state.injury

        if action == "fuir":
            energy -= self.flee_energy_cost
            danger *= self.flee_danger_factor
            fatigue += self.flee_fatigue_gain

        elif action == "chercher":
            energy -= self.search_energy_cost
            danger *= self.search_danger_factor
            fatigue += self.search_fatigue_gain

        elif action == "maintenir":
            energy -= self.maintain_energy_cost
            danger *= self.maintain_danger_factor
            fatigue += self.maintain_fatigue_gain

        # Récupération limitée de l'énergie.
        energy += self.recovery_rate * (1.0 - fatigue)

        # L'intégrité est liée à l'intensité de blessure.
        integrity = 1.0 - injury

        return ModelState(
            energy=clamp(energy),
            danger=clamp(danger),
            fatigue=clamp(fatigue),
            integrity=clamp(integrity),
            injury=clamp(injury),
        )

    def predict(
        self,
        state: ModelState,
        action: str,
    ) -> Prediction:
        predicted_after = self.transition(state, action)

        return Prediction(
            action=action,
            before=ModelState.from_dict(state.as_dict()),
            predicted_after=predicted_after,
        )

    @staticmethod
    def prediction_error(
        prediction: Prediction,
        observed_after: ModelState,
    ) -> float:
        """
        Erreur moyenne absolue sur les cinq dimensions.

        La comparaison porte sur l'état après action, avant tout
        événement externe ultérieur.
        """

        predicted = prediction.predicted_after.as_dict()
        observed = observed_after.as_dict()

        error = sum(
            abs(predicted[key] - observed[key])
            for key in STATE_KEYS
        ) / len(STATE_KEYS)

        prediction.predicted_error = error

        return error

    def coefficients(self) -> Dict[str, float]:
        return {
            "flee_energy_cost": self.flee_energy_cost,
            "flee_danger_factor": self.flee_danger_factor,
            "flee_fatigue_gain": self.flee_fatigue_gain,
            "search_energy_cost": self.search_energy_cost,
            "search_danger_factor": self.search_danger_factor,
            "search_fatigue_gain": self.search_fatigue_gain,
            "maintain_energy_cost": self.maintain_energy_cost,
            "maintain_danger_factor": self.maintain_danger_factor,
            "maintain_fatigue_gain": self.maintain_fatigue_gain,
            "recovery_rate": self.recovery_rate,
        }
