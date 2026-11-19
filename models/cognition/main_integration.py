# -*- coding: utf-8 -*-
"""
MAIN INTEGRATION
================

Simulation intégrée du sous-système cognitif expérimental.

Cycle :
1. événement environnemental ;
2. observation de l'état ;
3. génération de représentations candidates ;
4. compétition dans le Global Workspace ;
5. sélection d'une action ;
6. prédiction de la transition ;
7. exécution de l'action ;
8. comparaison prédiction-observation ;
9. révision métacognitive ;
10. stockage épisodique.

Exécution depuis la racine du projet :

    python -m models.cognition.main_integration

Ce programme est autonome et ne modifie aucun des cinq modèles
principaux du dossier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse
import json
import math

from .metacognition import Metacognition
from .workspace import GlobalWorkspace, Idee
from .self_model import SelfModel, SelfState
from .episodic_memory import Episode, EpisodicMemory
from .predictive_model import (
    ACTIONS,
    ModelState,
    PredictiveModel,
    Prediction,
)


@dataclass
class SimulationConfig:
    cycles: int = 100
    initial_energy: float = 1.0
    initial_danger: float = 0.10
    initial_fatigue: float = 0.0
    initial_injury: float = 0.0

    event_danger_cycle: int = 35
    event_danger_increase: float = 0.45

    event_recovery_cycle: int = 65
    event_danger_decrease: float = 0.30

    print_every: int = 10

    def __post_init__(self) -> None:
        if self.cycles < 1:
            raise ValueError("cycles doit être supérieur à zéro.")

        for key in (
            "initial_energy",
            "initial_danger",
            "initial_fatigue",
            "initial_injury",
        ):
            value = float(getattr(self, key))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} doit être dans [0, 1].")

        if self.event_danger_cycle < 0:
            raise ValueError("event_danger_cycle doit être positif ou nul.")

        if self.event_recovery_cycle < 0:
            raise ValueError("event_recovery_cycle doit être positif ou nul.")

        if self.print_every < 1:
            raise ValueError("print_every doit être supérieur à zéro.")


class CognitiveSimulation:
    """Orchestrateur de la simulation cognitive."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        predictive_model: Optional[PredictiveModel] = None,
    ) -> None:

        self.config = config or SimulationConfig()

        initial_state = SelfState(
            energy=self.config.initial_energy,
            danger=self.config.initial_danger,
            fatigue=self.config.initial_fatigue,
            integrity=1.0 - self.config.initial_injury,
            injury=self.config.initial_injury,
        )

        self.self_model = SelfModel(initial_state)
        self.workspace = GlobalWorkspace()
        self.memory = EpisodicMemory(capacity=1000)
        self.metacognition = Metacognition(
            learning_rate=0.20,
            surprise_threshold=0.20,
        )
        self.predictive_model = predictive_model or PredictiveModel()

        self.cycle_history: List[Dict] = []

        self.metacognition.add_belief(
            "energy",
            initial_state.energy,
            confidence=0.50,
        )
        self.metacognition.add_belief(
            "danger",
            initial_state.danger,
            confidence=0.50,
        )
        self.metacognition.add_belief(
            "fatigue",
            initial_state.fatigue,
            confidence=0.50,
        )

    @staticmethod
    def _to_model_state(state: SelfState) -> ModelState:
        return ModelState(
            energy=state.energy,
            danger=state.danger,
            fatigue=state.fatigue,
            integrity=state.integrity,
            injury=state.injury,
        )

    @staticmethod
    def _state_dict(state: SelfState) -> Dict[str, float]:
        return {
            "energy": state.energy,
            "danger": state.danger,
            "fatigue": state.fatigue,
            "integrity": state.integrity,
            "injury": state.injury,
        }

    def apply_external_events(self, cycle: int) -> List[str]:
        """
        Applique les perturbations extérieures avant la perception.

        Les événements sont explicites et consignés.
        """

        events = []

        if cycle == self.config.event_danger_cycle:
            self.self_model.update(
                danger_delta=self.config.event_danger_increase
            )
            events.append("danger_increase")

        if cycle == self.config.event_recovery_cycle:
            self.self_model.update(
                danger_delta=-self.config.event_danger_decrease
            )
            events.append("danger_decrease")

        return events

    def generate_candidates(self, cycle: int) -> None:
        """Transforme les variables internes en représentations candidates."""

        state = self.self_model.snapshot()

        danger_salience = state.danger
        energy_salience = 1.0 - state.energy
        fatigue_salience = state.fatigue

        self.workspace.ajouter(
            Idee(
                name="danger",
                content="Le niveau de danger demande une réponse.",
                salience=danger_salience,
                priority=2,
                source="self_model",
                cycle_created=cycle,
            )
        )

        self.workspace.ajouter(
            Idee(
                name="low_energy",
                content="Le niveau d'énergie est faible.",
                salience=energy_salience,
                priority=1,
                source="self_model",
                cycle_created=cycle,
            )
        )

        self.workspace.ajouter(
            Idee(
                name="fatigue",
                content="La fatigue augmente le coût des actions.",
                salience=fatigue_salience,
                priority=1,
                source="self_model",
                cycle_created=cycle,
            )
        )

    @staticmethod
    def select_action(
        selected_idea: Optional[Idee],
        state: SelfState,
    ) -> str:
        """
        Associe la représentation dominante à une action.

        Cette politique est volontairement simple et déterministe.
        """

        if selected_idea is None:
            return "maintenir"

        if selected_idea.name == "danger":
            if state.danger >= 0.25:
                return "fuir"
            return "maintenir"

        if selected_idea.name == "low_energy":
            return "maintenir"

        if selected_idea.name == "fatigue":
            return "maintenir"

        return "maintenir"

    def execute_action(
        self,
        action: str,
        prediction: Prediction,
    ) -> tuple[SelfState, float]:
        """
        Exécute exactement la transition définie par le modèle prédictif.

        La comparaison est donc effectuée avant les événements externes
        du cycle suivant.
        """

        predicted_state = prediction.predicted_after

        observed_state = self.self_model.update(
            energy=predicted_state.energy,
            danger=predicted_state.danger,
            fatigue=predicted_state.fatigue,
            injury=predicted_state.injury,
        )

        observed_copy = self.self_model.snapshot()

        error = self.predictive_model.prediction_error(
            prediction,
            self._to_model_state(observed_copy),
        )

        return observed_copy, error

    def update_metacognition(
        self,
        prediction: Prediction,
        observed_state: SelfState,
        cycle: int,
    ) -> None:
        predicted = prediction.predicted_after

        for variable in ("energy", "danger", "fatigue"):
            self.metacognition.revise(
                name=variable,
                observed_value=getattr(observed_state, variable),
                predicted_value=getattr(predicted, variable),
                cycle=cycle,
            )

    def store_episode(
        self,
        cycle: int,
        action: str,
        before: SelfState,
        after: SelfState,
        error: float,
        events: List[str],
    ) -> None:

        tags = list(events)

        if after.danger >= 0.50:
            tags.append("high_danger")

        if after.energy <= 0.30:
            tags.append("low_energy")

        if error > 0.05:
            tags.append("prediction_error")

        self.memory.store(
            Episode(
                cycle=cycle,
                action=action,
                state_before=self._state_dict(before),
                state_after=self._state_dict(after),
                prediction_error=error,
                tags=tags,
                outcome="transition_completed",
            )
        )

    def step(self, cycle: int) -> Dict:
        """Exécute un cycle complet."""

        events = self.apply_external_events(cycle)

        state_before = self.self_model.snapshot()

        self.generate_candidates(cycle)

        selected_idea = self.workspace.diffuser(cycle)

        action = self.select_action(
            selected_idea,
            state_before,
        )

        model_state = self._to_model_state(state_before)

        prediction = self.predictive_model.predict(
            model_state,
            action,
        )

        state_after, error = self.execute_action(
            action,
            prediction,
        )

        self.update_metacognition(
            prediction,
            state_after,
            cycle,
        )

        self.store_episode(
            cycle=cycle,
            action=action,
            before=state_before,
            after=state_after,
            error=error,
            events=events,
        )

        result = {
            "cycle": cycle,
            "events": events,
            "selected_idea": (
                selected_idea.name
                if selected_idea is not None
                else None
            ),
            "action": action,
            "prediction_error": error,
            "state_before": self._state_dict(state_before),
            "state_after": self._state_dict(state_after),
        }

        self.cycle_history.append(result)

        return result

    def run(self, verbose: bool = True) -> Dict:
        """Lance l'ensemble des cycles."""

        for cycle in range(self.config.cycles):
            result = self.step(cycle)

            if verbose and (
                cycle % self.config.print_every == 0
                or cycle == self.config.cycles - 1
            ):
                state = result["state_after"]

                print(
                    f"[{cycle:03d}] "
                    f"idée={result['selected_idea']!s:12s} "
                    f"action={result['action']:8s} "
                    f"E={state['energy']:.3f} "
                    f"D={state['danger']:.3f} "
                    f"F={state['fatigue']:.3f} "
                    f"erreur={result['prediction_error']:.6f}"
                )

        return self.summary()

    def summary(self) -> Dict:
        """Produit un bilan exploitable par un autre module."""

        errors = [
            item["prediction_error"]
            for item in self.cycle_history
        ]

        mean_error = (
            sum(errors) / len(errors)
            if errors
            else 0.0
        )

        final_state = self.self_model.as_dict()

        action_counts = {
            action: sum(
                1
                for item in self.cycle_history
                if item["action"] == action
            )
            for action in ACTIONS
        }

        return {
            "cycles": len(self.cycle_history),
            "mean_prediction_error": mean_error,
            "final_state": final_state,
            "action_counts": action_counts,
            "memory_size": len(self.memory),
            "metacognition": self.metacognition.summary(),
        }


def run_tests() -> None:
    """
    Tests minimaux de cohérence.

    Ils vérifient les invariants fonctionnels, pas la validité
    empirique de l'architecture.
    """

    # Test 1 : une blessure diminue l'intégrité.
    self_model = SelfModel(
        SelfState(
            energy=1.0,
            danger=0.0,
            fatigue=0.0,
            integrity=1.0,
            injury=0.0,
        )
    )

    self_model.update(injury=0.40)

    assert abs(self_model.state.integrity - 0.60) < 1e-9
    assert self_model.state.fatigue > 0.0

    # Test 2 : prédiction et transition doivent correspondre.
    predictor = PredictiveModel()

    initial = ModelState(
        energy=0.80,
        danger=0.70,
        fatigue=0.20,
        integrity=1.0,
        injury=0.0,
    )

    prediction = predictor.predict(initial, "fuir")
    observed = predictor.transition(initial, "fuir")

    error = predictor.prediction_error(prediction, observed)

    assert error < 1e-12

    # Test 3 : une mémoire doit pouvoir récupérer un épisode.
    memory = EpisodicMemory(capacity=10)

    memory.store(
        Episode(
            cycle=0,
            action="fuir",
            state_before=initial.as_dict(),
            state_after=observed.as_dict(),
            prediction_error=error,
            tags=["test"],
        )
    )

    assert len(memory) == 1
    assert len(memory.retrieve_by_action("fuir")) == 1
    assert len(memory.retrieve_by_tag("test")) == 1

    # Test 4 : la métacognition doit enregistrer une révision.
    meta = Metacognition()
    meta.add_belief("energy", 0.8, confidence=0.5)

    record = meta.revise(
        name="energy",
        observed_value=0.7,
        predicted_value=0.8,
        cycle=1,
    )

    assert record.prediction_error > 0.0
    assert len(meta.history) == 1

    # Test 5 : l'intégration doit produire tous les cycles.
    simulation = CognitiveSimulation(
        SimulationConfig(cycles=5)
    )

    result = simulation.run(verbose=False)

    assert result["cycles"] == 5
    assert result["memory_size"] == 5

    print("Tous les tests minimaux sont réussis.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation cognitive expérimentale."
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Nombre de cycles de simulation.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Désactive l'affichage périodique.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Exécute les tests minimaux.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Affiche le bilan final au format JSON.",
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    config = SimulationConfig(cycles=args.cycles)
    simulation = CognitiveSimulation(config)

    result = simulation.run(verbose=not args.quiet)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif not args.quiet:
        print("\nBILAN FINAL")
        print("-----------")
        print(f"Cycles : {result['cycles']}")
        print(
            "Erreur prédictive moyenne : "
            f"{result['mean_prediction_error']:.6f}"
        )
        print(f"Actions : {result['action_counts']}")
        print(f"Mémoire : {result['memory_size']} épisodes")
        print(f"État final : {result['final_state']}")


if __name__ == "__main__":
    main()
