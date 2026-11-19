#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation exploratoire de stéréomorphogenèse sur graphe — V1.1 (Refactorisé).

Pipeline :
1. Génération de coordonnées 3D.
2. Construction d'un graphe géométrique kNN symétrique.
3. Compression d'arêtes (pondérée, aléatoire ou témoin complet).
4. Relaxation géométrique mécanique (forces élastiques + répulsion).
5. Reconstruction du graphe kNN sur la nouvelle configuration spatiale.
6. Intégration de la dynamique double courant F-I-B.
7. Évaluation des distances topologiques et dynamiques.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .compression_temoins import compresser_aleatoire, compresser_ponderee
    from .dynamique_double_courant import laplacien_graphe, spearman, step
except ImportError:
    from compression_temoins import compresser_aleatoire, compresser_ponderee
    from dynamique_double_courant import laplacien_graphe, spearman, step


# ============================================================================
# CONFIGURATION ET HYPERPARAMÈTRES
# ============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    version: str = "V1.1"

    # Graphe et échantillonnage
    k_knn: int = 10
    n_noeuds: int = 128
    n_graines: int = 100

    # Pas de temps
    t_initial: int = 100
    t_compresse: int = 100
    t_final: int = 200

    # Paramètres de contraction et seuils
    lambda_contraction: float = 0.6
    seuil_echec: float = 2.0

    # Relaxation mécanique
    relax_iterations: int = 80
    relax_eta: float = 0.05
    relax_kappa: float = 1.0
    relax_krep: float = 0.5
    relax_tolerance_deplacement: float = 1e-6
    relax_tolerance_force: float = 1e-6
    repulsion_fraction: float = 0.5

    # Gestion des singularités géométriques
    collision_tolerance: float = 1e-12
    collision_separation: float = 1e-8

    # Paramètres de la dynamique F-I-B
    alpha: float = 0.25
    beta: float = 0.05
    g: float = 0.6
    d_f: float = 0.1
    d_i: float = 8.0
    gamma: float = 1.0
    seuil_b: float = 0.0

    conditions: Tuple[str, ...] = (
        "ponderee",
        "aleatoire",
        "ressorts_complets",
    )


CONFIG = SimulationConfig()


# ============================================================================
# VALIDATION DES ENTRÉES
# ============================================================================

def valider_adjacence(adjacency: np.ndarray) -> np.ndarray:
    """Valide, rend symétrique et annule la diagonale d'une matrice d'adjacence."""
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("L'adjacence doit être une matrice carrée 2D.")
    if not np.all(np.isfinite(A)):
        raise ValueError("L'adjacence contient des valeurs non finies.")
    if np.any(A < 0.0):
        raise ValueError("L'adjacence ne peut contenir de poids négatifs.")

    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


def valider_points(points: np.ndarray) -> np.ndarray:
    """Valide les coordonnées spatiales des nœuds."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] < 1:
        raise ValueError("La matrice des points doit être de dimensions (N >= 2, D >= 1).")
    if not np.all(np.isfinite(P)):
        raise ValueError("Les coordonnées contiennent des valeurs non finies.")
    return P


# ============================================================================
# GÉOMÉTRIE ET TOPOLOGIE
# ============================================================================

def knn_graph(points: np.ndarray, k: int = CONFIG.k_knn) -> np.ndarray:
    """
    Construit un graphe kNN symétrisé par union via partitionnement partiel.
    """
    P = valider_points(points)
    n_nodes = len(P)
    k = int(k)

    if not 1 <= k < n_nodes:
        raise ValueError("k doit satisfaire 1 <= k < nombre de nœuds.")

    # Calcul des distances euclidiennes par paires
    distances = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)

    # Sélection efficace des k plus proches voisins (O(N) par nœud)
    voisins = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    lignes = np.arange(n_nodes)[:, None]
    adjacency[lignes, voisins] = 1.0

    # Symétrisation par union
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def statistiques_aretes(points: np.ndarray, adjacency: np.ndarray) -> Dict[str, float]:
    """Extrait les propriétés métriques des arêtes du réseau."""
    P = valider_points(points)
    A = valider_adjacence(adjacency)

    edges = np.argwhere(np.triu(A, 1) > 0.0)
    if len(edges) == 0:
        return {
            "n_aretes": 0.0,
            "longueur_moyenne": 0.0,
            "longueur_minimale": 0.0,
            "longueur_maximale": 0.0,
        }

    lengths = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
    return {
        "n_aretes": float(len(edges)),
        "longueur_moyenne": float(np.mean(lengths)),
        "longueur_minimale": float(np.min(lengths)),
        "longueur_maximale": float(np.max(lengths)),
    }


def longueur_moyenne(points: np.ndarray, adjacency: np.ndarray) -> float:
    """Renvoie la distance euclidienne moyenne entre nœuds adjacents."""
    stats = statistiques_aretes(points, adjacency)
    if stats["n_aretes"] == 0:
        raise ValueError("Le graphe ne contient aucune arête.")
    return stats["longueur_moyenne"]


def separer_collisions(
    points: np.ndarray,
    tolerance: float = CONFIG.collision_tolerance,
    separation: float = CONFIG.collision_separation,
) -> Tuple[np.ndarray, int]:
    """Régularisation numérique déterministe prévenant les divisions par zéro."""
    P = valider_points(points).copy()
    n_nodes, dim = P.shape

    dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    paires = np.argwhere(np.triu(dist <= tolerance, 1))
    count = len(paires)

    for i, j in paires:
        direction = np.zeros(dim, dtype=float)
        if dim >= 2:
            angle = 2.0 * math.pi * (i + j + 1) / (n_nodes + 1)
            direction[0] = math.cos(angle)
            direction[1] = math.sin(angle)
        else:
            direction[0] = 1.0

        P[i] += 0.5 * separation * direction
        P[j] -= 0.5 * separation * direction

    return P, count


# ============================================================================
# RELAXATION MÉCANIQUE
# ============================================================================

def relaxer(
    points: np.ndarray,
    adjacency: np.ndarray,
    longueur_reference: float,
    cfg: SimulationConfig = CONFIG,
) -> Dict[str, Any]:
    """
    Minimise l'énergie élastique sous contrainte de répulsion locale.
    """
    P = valider_points(points).copy()
    A = valider_adjacence(adjacency)

    if longueur_reference <= 0.0:
        raise ValueError("longueur_reference doit être strictement positive.")

    longueur_repos = cfg.lambda_contraction * longueur_reference
    longueur_repulsion = cfg.repulsion_fraction * longueur_reference

    P, collision_count = separer_collisions(P, cfg.collision_tolerance, cfg.collision_separation)

    n_nodes = len(P)
    identity = np.eye(n_nodes)
    converged = False
    max_displacement = 0.0
    force_norm = 0.0
    iterations_used = 0

    for it in range(cfg.relax_iterations):
        diff = P[:, None, :] - P[None, :, :]  # Vecteurs r_ij = P_i - P_j
        dist = np.linalg.norm(diff, axis=2)
        dist_safe = np.maximum(dist, 1e-12)
        directions = diff / dist_safe[:, :, None]

        # Force de rappel élastique le long des arêtes
        f_ressorts = -cfg.relax_kappa * (A * (dist - longueur_repos))[:, :, None] * directions

        # Répulsion locale pour éviter l'effondrement
        masque_rep = (dist < longueur_repulsion) * (1.0 - identity)
        f_repulsion = cfg.relax_krep * (masque_rep * (longueur_repulsion - dist))[:, :, None] * directions

        force_totale = np.sum(f_ressorts + f_repulsion, axis=1)
        deplacement = cfg.relax_eta * force_totale
        P += deplacement

        max_displacement = float(np.max(np.linalg.norm(deplacement, axis=1)))
        force_norm = float(np.linalg.norm(force_totale))
        iterations_used = it + 1

        if not np.all(np.isfinite(P)):
            raise FloatingPointError("Divergence numérique lors de la relaxation géométrique.")

        if (
            max_displacement <= cfg.relax_tolerance_deplacement
            and force_norm <= cfg.relax_tolerance_force
        ):
            converged = True
            break

    edge_stats = statistiques_aretes(P, A)
    return {
        "points": P,
        "converged": converged,
        "iterations_used": iterations_used,
        "max_displacement": max_displacement,
        "force_norm": force_norm,
        "collision_count": collision_count,
        "mean_edge_length": edge_stats["longueur_moyenne"],
        "max_edge_length": edge_stats["longueur_maximale"],
    }


# ============================================================================
# MESURES TOPOLOGIQUES
# ============================================================================

def _recuperer_ensembles_aretes(A0: np.ndarray, A1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    adj0 = valider_adjacence(A0)
    adj1 = valider_adjacence(A1)
    return np.triu(adj0, 1) > 0.0, np.triu(adj1, 1) > 0.0


def delta_relationnelle(A_init: np.ndarray, A_final: np.ndarray) -> float:
    """Proportion des arêtes initiales absentes du graphe reconstruit (perte)."""
    e0, e1 = _recuperer_ensembles_aretes(A_init, A_final)
    n_init = np.sum(e0)
    return float(1.0 - np.sum(e0 & e1) / n_init) if n_init > 0 else 0.0


def delta_gain(A_init: np.ndarray, A_final: np.ndarray) -> float:
    """Proportion d'arêtes nouvellement créées dans le graphe reconstruit (gain)."""
    e0, e1 = _recuperer_ensembles_aretes(A_init, A_final)
    n_final = np.sum(e1)
    return float(1.0 - np.sum(e0 & e1) / n_final) if n_final > 0 else 0.0


def distance_jaccard(A_init: np.ndarray, A_final: np.ndarray) -> float:
    """Distance de Jaccard entre les topologies de connectivité."""
    e0, e1 = _recuperer_ensembles_aretes(A_init, A_final)
    inter = np.sum(e0 & e1)
    union = np.sum(e0 | e1)
    return float(1.0 - inter / union) if union > 0 else 0.0


# ============================================================================
# INTÉGRATION DU SYSTÈME DYNAMIQUE
# ============================================================================

def initialiser_champs(n_nodes: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialise les champs continus et le champ de décision binaire."""
    rng = np.random.default_rng(seed)
    F = rng.uniform(-0.5, 0.5, size=n_nodes)
    I = rng.uniform(-0.5, 0.5, size=n_nodes)
    B = (F + 0.5 * I > CONFIG.seuil_b).astype(float)
    return F, I, B


def evoluer(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    adjacency: np.ndarray,
    n_steps: int,
    cfg: SimulationConfig = CONFIG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagateur de la dynamique réaction-diffusion couplée sur le graphe."""
    A = valider_adjacence(adjacency)
    field_f = np.asarray(F, dtype=float).copy()
    field_i = np.asarray(I, dtype=float).copy()
    field_b = np.asarray(B, dtype=float).copy()

    laplacian = laplacien_graphe(A)

    for _ in range(n_steps):
        field_f, field_i, field_b = step(
            field_f,
            field_i,
            field_b,
            laplacian,
            alpha=cfg.alpha,
            beta=cfg.beta,
            g=cfg.g,
            d_f=cfg.d_f,
            d_i=cfg.d_i,
            gamma=cfg.gamma,
            seuil=cfg.seuil_b,
        )

    return field_f, field_i, field_b


def distance_FI(F1: np.ndarray, I1: np.ndarray, F2: np.ndarray, I2: np.ndarray) -> float:
    """Distance euclidienne dans l'espace des phases continu (F, I)."""
    return float(np.sqrt(np.sum((F1 - F2) ** 2) + np.sum((I1 - I2) ** 2)))


# ============================================================================
# SIMULATION PRINCIPALE
# ============================================================================

def une_simulation(seed: int, cfg: SimulationConfig = CONFIG) -> Dict[str, Dict[str, Any]]:
    """Exécute une réplication complète pour l'ensemble des conditions comparées."""
    rng = np.random.default_rng(seed)

    # 1. Configuration géométrique initiale
    points_init = rng.uniform(0.0, 1.0, size=(cfg.n_noeuds, 3))
    adj_init = knn_graph(points_init, cfg.k_knn)
    l_ref = longueur_moyenne(points_init, adj_init)

    # 2. Régime dynamique initial
    F, I, B = initialiser_champs(cfg.n_noeuds, seed + 100_000)
    F_ref, I_ref, B_ref = evoluer(F, I, B, adj_init, cfg.t_initial, cfg)
    F_one, I_one, _ = evoluer(F_ref, I_ref, B_ref, adj_init, 1, cfg)

    # 3. Réduction de graphe (50 % des arêtes)
    n_edges = int(np.sum(np.triu(adj_init, 1)))
    n_edges_target = n_edges // 2

    conditions = {
        "ponderee": compresser_ponderee(adj_init, F_ref, I_ref, n_edges_target, rng),
        "aleatoire": compresser_aleatoire(adj_init, n_edges_target, rng),
        "ressorts_complets": adj_init.copy(),
    }

    resultats: Dict[str, Dict[str, Any]] = {}

    for nom, adj_intermediaire in conditions.items():
        adj_inter = valider_adjacence(adj_intermediaire)

        # 4. Relaxation morphologique
        relax_res = relaxer(points_init, adj_inter, l_ref, cfg)
        points_finaux = relax_res["points"]

        # 5. Reconstruction kNN
        adj_finale = knn_graph(points_finaux, cfg.k_knn)

        # 6. Évolution dynamique sur les états transitoires et reconstruits
        f, i, b = evoluer(F_ref, I_ref, B_ref, adj_inter, cfg.t_compresse, cfg)
        f, i, b = evoluer(f, i, b, adj_finale, cfg.t_final, cfg)

        # 7. Évaluation des distances
        dist_retour_fi = min(
            distance_FI(f, i, F_ref, I_ref),
            distance_FI(f, i, F_one, I_one),
        )

        deplacement = float(np.sqrt(np.mean(np.sum((points_finaux - points_init) ** 2, axis=1))))
        stats_aretes_fin = statistiques_aretes(points_finaux, adj_finale)

        resultats[nom] = {
            "delta_relationnelle": delta_relationnelle(adj_init, adj_finale),
            "delta_gain": delta_gain(adj_init, adj_finale),
            "distance_jaccard": distance_jaccard(adj_init, adj_finale),
            "distance_retour_FI": dist_retour_fi,
            "distance_retour_B": float(np.mean(b != B_ref)),
            "deplacement_moyen": deplacement,
            "relaxation_converged": relax_res["converged"],
            "relaxation_iterations": relax_res["iterations_used"],
            "relaxation_max_displacement": relax_res["max_displacement"],
            "relaxation_force_norm": relax_res["force_norm"],
            "final_mean_edge_length": stats_aretes_fin["longueur_moyenne"],
        }

    return resultats


# ============================================================================
# ANALYSE STATISTIQUE ET BOOTSTRAP
# ============================================================================

def intervalle_bootstrap_moyenne(
    valeurs: np.ndarray,
    seed: int = 7,
    n_bootstrap: int = 2000,
) -> Tuple[float, float]:
    """Intervalle de confiance bootstrap à 95 % (méthode des percentiles)."""
    values = np.asarray(valeurs, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    echantillons = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    moyennes = np.mean(echantillons, axis=1)
    return float(np.percentile(moyennes, 2.5)), float(np.percentile(moyennes, 97.5))


def afficher_resultats(resultats: List[Dict[str, Dict[str, Any]]], cfg: SimulationConfig = CONFIG) -> None:
    """Mise en forme des résultats empiriques."""
    print(f"=== Simulation exploratoire de stéréomorphogenèse — {cfg.version} ===")
    print(f"Réplications : {len(resultats)} | Nœuds : {cfg.n_noeuds} | kNN : {cfg.k_knn}\n")

    titres = f"{'Condition':22s}{'Perte':>10s}{'Gain':>10s}{'Jaccard':>10s}{'Retour FI':>12s}{'Retour B':>10s}{'Échecs':>12s}"
    print(titres)
    print("-" * len(titres))

    for condition in cfg.conditions:
        perte = np.array([r[condition]["delta_relationnelle"] for r in resultats])
        gain = np.array([r[condition]["delta_gain"] for r in resultats])
        jaccard = np.array([r[condition]["distance_jaccard"] for r in resultats])
        fi = np.array([r[condition]["distance_retour_FI"] for r in resultats])
        b = np.array([r[condition]["distance_retour_B"] for r in resultats])
        echecs = int(np.sum(fi >= cfg.seuil_echec))

        print(
            f"{condition:22s}"
            f"{np.mean(perte):10.3f}"
            f"{np.mean(gain):10.3f}"
            f"{np.mean(jaccard):10.3f}"
            f"{np.mean(fi):12.3f}"
            f"{np.mean(b):10.3f}"
            f"{echecs:6d}/{len(resultats):<3d}"
        )

        rho = spearman(perte, fi)
        print(f"  └─ Spearman (perte vs retour FI) : {rho:.4f}")

    diff_perte = np.array([r["ponderee"]["delta_relationnelle"] - r["aleatoire"]["delta_relationnelle"] for r in resultats])
    diff_fi = np.array([r["ponderee"]["distance_retour_FI"] - r["aleatoire"]["distance_retour_FI"] for r in resultats])

    ci_perte = intervalle_bootstrap_moyenne(diff_perte)
    ci_fi = intervalle_bootstrap_moyenne(diff_fi)

    print("\n=== Comparaisons appariées : Pondérée - Aléatoire ===")
    print(f"Delta perte     : {np.mean(diff_perte):+.4f} [{ci_perte[0]:+.4f}, {ci_perte[1]:+.4f}]")
    print(f"Delta retour FI : {np.mean(diff_fi):+.4f} [{ci_fi[0]:+.4f}, {ci_fi[1]:+.4f}]")


# ============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================================

def main() -> None:
    graines = list(range(CONFIG.n_graines))
    resultats = [une_simulation(seed) for seed in graines]

    afficher_resultats(resultats)

    archive = {
        "metadata": asdict(CONFIG),
        "results": resultats,
    }

    nom_fichier = "resultats_stereomorphogenese.json"
    with open(nom_fichier, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

    print(f"\nDonnées archivées avec succès dans '{nom_fichier}'.")


if __name__ == "__main__":
    main()
