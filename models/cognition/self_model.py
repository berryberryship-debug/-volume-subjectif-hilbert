# -*- coding: utf-8 -*-
"""
SELF MODEL
==========

Représentation fonctionnelle d'un état interne simulé.

Les variables sont des grandeurs abstraites normalisées.
Elles ne correspondent pas directement à des mesures physiologiques.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class SelfState:
    energy: float = 1.0
    danger: float = 0.0
    fatigue: float = 0.0
    integrity: float = 1.0
    injury: float = 0.0

    def __post_init__(self) -> None:
        self.energy = _clamp(self.energy)
        self.danger = _clamp(self.danger)
        self.fatigue = _clamp(self.fatigue)
        self.integrity = _clamp(self.integrity)
        self.injury = _clamp(self.injury)


@dataclass
class SelfModel:
    """
    Maintient et actualise un état interne.

    injury représente une intensité de blessure normalisée.
    integrity représente l'intégrité fonctionnelle restante.
    """

    state: SelfState

    def update(
        self,
        energy: float | None = None,
        danger: float | None = None,
        fatigue: float | None = None,
        injury: float | None = None,
        energy_delta: float = 0.0,
        danger_delta: float = 0.0,
        fatigue_delta: float = 0.0,
        injury_delta: float = 0.0,
    ) -> SelfState:
        """
        Met à jour l'état.

        Les valeurs absolues remplacent les variables correspondantes.
        Les deltas sont ensuite appliqués.

        Une blessure accrue diminue l'intégrité.
        """

        if energy is not None:
            self.state.energy = _clamp(energy)

        if danger is not None:
            self.state.danger = _clamp(danger)

        if fatigue is not None:
            self.state.fatigue = _clamp(fatigue)

        if injury is not None:
            self.state.injury = _clamp(injury)

        self.state.energy = _clamp(
            self.state.energy + energy_delta
        )

        self.state.danger = _clamp(
            self.state.danger + danger_delta
        )

        self.state.fatigue = _clamp(
            self.state.fatigue + fatigue_delta
        )

        self.state.injury = _clamp(
            self.state.injury + injury_delta
        )

        # L'intégrité est calculée à partir de la blessure.
        self.state.integrity = _clamp(
            1.0 - self.state.injury
        )

        # Une blessure peut augmenter la fatigue simulée.
        self.state.fatigue = _clamp(
            self.state.fatigue + 0.05 * self.state.injury
        )

        return self.state

    def snapshot(self) -> SelfState:
        """Retourne une copie de l'état courant."""

        return SelfState(
            energy=self.state.energy,
            danger=self.state.danger,
            fatigue=self.state.fatigue,
            integrity=self.state.integrity,
            injury=self.state.injury,
        )

    def as_dict(self) -> dict:
        return {
            "energy": self.state.energy,
            "danger": self.state.danger,
            "fatigue": self.state.fatigue,
            "integrity": self.state.integrity,
            "injury": self.state.injury,
        }
