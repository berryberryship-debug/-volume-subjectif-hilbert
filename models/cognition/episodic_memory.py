# -*- coding: utf-8 -*-
"""
EPISODIC MEMORY
===============

Mémoire d'épisodes simulés.

Chaque épisode conserve :
- l'état avant l'action ;
- l'action ;
- l'état après l'action ;
- l'erreur de prédiction ;
- les étiquettes descriptives.

La récupération par proximité est une opération numérique,
pas une reproduction de la mémoire autobiographique humaine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


STATE_KEYS = (
    "energy",
    "danger",
    "fatigue",
    "integrity",
    "injury",
)


@dataclass
class Episode:
    cycle: int
    action: str
    state_before: Dict[str, float]
    state_after: Dict[str, float]
    prediction_error: float
    tags: List[str] = field(default_factory=list)
    outcome: str = ""

    def __post_init__(self) -> None:
        self.prediction_error = float(self.prediction_error)

        if self.prediction_error < 0.0:
            raise ValueError("prediction_error doit être positif ou nul.")

        self.state_before = self._normalize_state(self.state_before)
        self.state_after = self._normalize_state(self.state_after)
        self.tags = list(self.tags)

    @staticmethod
    def _normalize_state(state: Dict[str, float]) -> Dict[str, float]:
        normalized = {}

        for key in STATE_KEYS:
            value = float(state.get(key, 0.0))

            if not math.isfinite(value):
                raise ValueError(f"État non fini pour {key}.")

            normalized[key] = max(0.0, min(1.0, value))

        return normalized


@dataclass
class EpisodicMemory:
    capacity: int = 1000
    episodes: List[Episode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity doit être supérieur à zéro.")

    def store(self, episode: Episode) -> None:
        """Stocke un épisode et applique une capacité maximale."""

        self.episodes.append(episode)

        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def retrieve_by_action(self, action: str) -> List[Episode]:
        return [
            episode
            for episode in self.episodes
            if episode.action == action
        ]

    def retrieve_by_tag(self, tag: str) -> List[Episode]:
        return [
            episode
            for episode in self.episodes
            if tag in episode.tags
        ]

    def retrieve_similar_state(
        self,
        state: Dict[str, float],
        limit: int = 5,
    ) -> List[Episode]:
        """
        Recherche les épisodes dont l'état initial est le plus proche.

        Distance euclidienne sur les cinq variables normalisées.
        """

        if limit < 1:
            return []

        query = Episode._normalize_state(state)

        ranked = []

        for episode in self.episodes:
            squared_distance = sum(
                (
                    query[key]
                    - episode.state_before[key]
                ) ** 2
                for key in STATE_KEYS
            )

            distance = math.sqrt(squared_distance)

            ranked.append((distance, episode))

        ranked.sort(key=lambda item: (item[0], item[1].cycle))

        return [
            episode
            for _, episode in ranked[:limit]
        ]

    def recent(self, limit: int = 10) -> List[Episode]:
        if limit < 1:
            return []

        return self.episodes[-limit:]

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)
