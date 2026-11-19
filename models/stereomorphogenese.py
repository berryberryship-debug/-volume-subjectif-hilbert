#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation exploratoire de stéréomorphogenèse sur graphe — V1.3.

Chaîne topologique :
    A_0  (graphe initial kNN)
     │
     ▼  compression
    A^c  (graphe comprimé, avant relaxation)
     │
     ▼  relaxation géométrique + reconstruction kNN
    A^f  (graphe final)

Chaîne dynamique :
    X_0  (état initial après t_initial sur A_0 ; X_1 = step(X_0, A_0))
     │
     ▼  t_compresse sur A^c
    X_c  (état après compression dynamique)
     │
     ▼  t_final sur A^f
    X_f  (état après transformation morphologique)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from .compression_temoins import compresser_aleatoire, compresser_ponderee
    from .dynamique_double_courant import laplacien_graphe, spearman, step
except ImportError:
    from compression_temoins import compresser_aleatoire, compresser_ponderee
    from dynamique_double_courant import laplacien_graphe, spearman, step


# ============================================================================
# CONFIGURATION ET HYPERPARAMÈTRES (V1.3)
# ============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    version: str = "V1.3"

    # Graphe et échantillonnage
    k_knn: int = 10
    n_noeuds: int = 128
    n_graines: int = 100

    # Pas de temps
    t_initial: int = 100
    t_compresse: int = 100
    t_final: int = 200

    # Paramètres de contraction et seuils normalisés (seuil opérationnel à calibrer)
    lambda_contraction: float = 0.6
    seuil_echec_rms: float = 0.25

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
        "temoin_intact",
        "ponderee",
        "aleatoire",
        "ressorts_complets",
    )

    def __post_init__(self) -> None:
        if self.n_noeuds < 2:
            raise ValueError("n_noeuds doit être >= 2.")
        if not (1 <= self.k_knn < self.n_noeuds):
            raise ValueError("k_knn doit satisfaire 1 <= k_knn < n_noeuds.")
        if self.n_graines < 1:
            raise ValueError("n_graines doit être >= 1.")
        if self.t_initial < 0 or self.t_compresse < 0 or self.t_final < 0:
            raise ValueError("Les durées d'évolution doivent être >= 0.")
        if self.lambda_contraction <= 0.0 or not math.isfinite(self.lambda_contraction):
            raise ValueError("lambda_contraction doit être strictement positif et fini.")
        if self.seuil_echec_rms <= 0.0 or not math.isfinite(self.seuil_echec_rms):
            raise ValueError("seuil_echec_rms doit être strictement positif et fini.")
        if self.relax_iterations < 1:
            raise ValueError("relax_iterations doit être >= 1.")
        if self.relax_eta <= 0.0 or self.relax_kappa < 0.0 or self.relax_krep < 0.0:
            raise ValueError("Les paramètres de relaxation doivent être cohérents et positifs.")
        if self.collision_tolerance < 0.0 or self.collision_separation <= 0.0:
            raise ValueError("Les paramètres de collision doivent être valides.")
        if not self.conditions:
            raise ValueError("conditions ne peut pas être vide.")


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

def knn_graph(points: np.ndarray, k: int) -> np.ndarray:
    """
    Construit un graphe kNN symétrisé par union avec départage déterministe
    des distances égales via l'indice de colonne.
    """
    P = valider_points(points)
    n_nodes = len(P)
    k = int(k)

    if not 1 <= k < n_nodes:
        raise ValueError("k doit satisfaire 1 <= k < nombre de nœuds.")

    distances = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    col_indices = np.broadcast_to(np.arange(n_nodes)[None, :], distances.shape)

    # Départage déterministe : clé primaire = indice voisin, clé secondaire = distance
    voisins = np.empty((n_nodes, k), dtype=int)
    for i in range(n_nodes):
        order = np.lexsort((col_indices[i], distances[i]))
        voisins[i] = order[:k]

    lignes = np.arange(n_nodes)[:, None]
    adjacency[lignes, voisins] = 1.0

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def statistiques_aretes(points: np.ndarray, adjacency: np.ndarray) -> Dict[str, float]:
    """Extrait les propriétés métriques des arêtes actives du réseau."""
    P = valider_points(points)
    A = valider_adjacence(adjacency)

    edges = np.argwhere(np.triu(A > 0.0, 1))
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
    tolerance: float,
    separation: float,
) -> Tuple[np.ndarray, int, int]:
    """
    Détecte les collisions sous tolérance, applique un déplacement déterministe
    et audite les collisions résiduelles.
    """
    P = valider_points(points).copy()
    n_nodes, dim = P.shape

    dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    paires = np.argwhere(np.triu(dist <= tolerance, 1))
    initial_collisions = int(len(paires))

    for idx_paire, (i, j) in enumerate(paires):
        direction = np.zeros(dim, dtype=float)
        if dim >= 2:
            angle = 2.0 * math.pi * (int(i) + int(j) + idx_paire + 1) / (n_nodes + 1)
            direction[0] = math.cos(angle)
            direction[1] = math.sin(angle)
        else:
            direction[0] = 1.0

        P[i] += 0.5 * separation * direction
        P[j] -= 0.5 * separation * direction

    dist_after = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(dist_after, np.inf)
    residual_collisions = int(np.sum(np.triu(dist_after <= tolerance, 1)))

    return P, initial_collisions, residual_collisions


# ============================================================================
# RELAXATION MÉCANIQUE
# ============================================================================

def _energie_elastique_et_repulsion(
    P: np.ndarray,
    A: np.ndarray,
    longueur_repos: float,
    longueur_repulsion: float,
    kappa: float,
    krep: float,
) -> float:
    """Calcule l'énergie potentielle discrète associée aux ressorts et à la répulsion."""
    dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    upper = np.triu(np.ones_like(dist, dtype=bool), 1)

    e_ressorts = 0.5 * kappa * float(np.sum((A[upper] > 0.0) * ((dist[upper] - longueur_repos) ** 2)))
    delta_rep = np.maximum(longueur_repulsion - dist[upper], 0.0)
    e_repulsion = 0.5 * krep * float(np.sum(delta_rep ** 2))
    return e_ressorts + e_repulsion


def relaxer(
    points: np.ndarray,
    adjacency: np.ndarray,
    longueur_reference: float,
    cfg: SimulationConfig,
) -> Dict[str, Any]:
    """
    Minimise l'énergie élastique sous contrainte de répulsion locale, avec suivi
    de l'énergie mécanique et détection de divergence ou d'oscillation.
    """
    P = valider_points(points).copy()
    A = valider_adjacence(adjacency)

    if longueur_reference <= 0.0:
        raise ValueError("longueur_reference doit être strictement positive.")

    longueur_repos = cfg.lambda_contraction * longueur_reference
    longueur_repulsion = cfg.repulsion_fraction * longueur_reference

    P, collision_count, residual_collisions = separer_collisions(
        P, cfg.collision_tolerance, cfg.collision_separation
    )

    n_nodes = len(P)
    identity = np.eye(n_nodes)
    converged = False
    energy_diverged = False
    max_displacement = 0.0
    force_norm = 0.0
    iterations_used = 0

    energie_init = _energie_elastique_et_repulsion(
        P, A, longueur_repos, longueur_repulsion, cfg.relax_kappa, cfg.relax_krep
    )
    energie_courante = energie_init
    energie_max_observee = energie_init

    for it in range(cfg.relax_iterations):
        diff = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist_safe = np.maximum(dist, 1e-12)
        directions = diff / dist_safe[:, :, None]

        f_ressorts = -cfg.relax_kappa * ((A > 0.0)[:, :, None] * (dist - longueur_repos)[:, :, None]) * directions
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

        energie_courante = _energie_elastique_et_repulsion(
            P, A, longueur_repos, longueur_repulsion, cfg.relax_kappa, cfg.relax_krep
        )
        if not math.isfinite(energie_courante) or energie_courante > max(100.0 * (energie_init + 1.0), 1e6):
            energy_diverged = True
            break
        energie_max_observee = max(energie_max_observee, energie_courante)

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
        "energy_diverged": energy_diverged,
        "iterations_used": iterations_used,
        "max_displacement": max_displacement,
        "force_norm": force_norm,
        "energy_init": energie_init,
        "energy_final": energie_courante,
        "energy_max": energie_max_observee,
        "collision_count": collision_count,
        "residual_collisions": residual_collisions,
        "mean_edge_length": edge_stats["longueur_moyenne"],
        "max_edge_length": edge_stats["longueur_maximale"],
    }


# ============================================================================
# MESURES TOPOLOGIQUES (A_0 -> A^c -> A^f, PAIRES NON ORDONNÉES STRICTES)
# ============================================================================

def _recuperer_ensembles_aretes(A_avant: np.ndarray, A_apres: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    adj_avant = valider_adjacence(A_avant)
    adj_apres = valider_adjacence(A_apres)
    if adj_avant.shape != adj_apres.shape:
        raise ValueError("Les matrices d'adjacence doivent avoir la même dimension.")
    return np.triu(adj_avant > 0.0, 1), np.triu(adj_apres > 0.0, 1)


def perte_aretes(A_avant: np.ndarray, A_apres: np.ndarray) -> float:
    """Proportion des arêtes présentes dans A_avant qui sont absentes de A_apres."""
    e_av, e_ap = _recuperer_ensembles_aretes(A_avant, A_apres)
    n_av = int(np.sum(e_av))
    return float(1.0 - np.sum(e_av & e_ap) / n_av) if n_av > 0 else 0.0


def gain_aretes(A_avant: np.ndarray, A_apres: np.ndarray) -> float:
    """Proportion des paires non ordonnées absentes avant qui apparaissent après."""
    e_av, e_ap = _recuperer_ensembles_aretes(A_avant, A_apres)
    masque_non_ordonne = np.triu(np.ones(e_av.shape, dtype=bool), k=1)

    absentes_avant = masque_non_ordonne & (~e_av)
    n_absentes = int(np.sum(absentes_avant))
    n_creees = int(np.sum(absentes_avant & e_ap))
    return float(n_creees / n_absentes) if n_absentes > 0 else 0.0


def distance_jaccard(A_avant: np.ndarray, A_apres: np.ndarray) -> float:
    """Distance de Jaccard entre deux topologies sur les paires non ordonnées."""
    e_av, e_ap = _recuperer_ensembles_aretes(A_avant, A_apres)
    inter = int(np.sum(e_av & e_ap))
    union = int(np.sum(e_av | e_ap))
    return float(1.0 - inter / union) if union > 0 else 0.0


def analyser_transitions_graphes(
    A0: np.ndarray,
    Ac: np.ndarray,
    Af: np.ndarray,
) -> Dict[str, float]:
    """Calcule pertes et gains pour A0 -> Ac, Ac -> Af et A0 -> Af."""
    return {
        "perte_compression_A0_Ac": perte_aretes(A0, Ac),
        "gain_compression_A0_Ac": gain_aretes(A0, Ac),
        "transformation_morpho_perte_Ac_Af": perte_aretes(Ac, Af),
        "transformation_morpho_gain_Ac_Af": gain_aretes(Ac, Af),
        "ecart_total_perte_A0_Af": perte_aretes(A0, Af),
        "ecart_total_gain_A0_Af": gain_aretes(A0, Af),
        "jaccard_A0_Af": distance_jaccard(A0, Af),
    }


# ============================================================================
# INTÉGRATION DU SYSTÈME DYNAMIQUE ET DISTANCES D'ÉTAPE
# ============================================================================

def initialiser_champs(
    n_nodes: int,
    seed: int,
    seuil_b: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialise les champs continus et le champ binaire avec seuil explicite."""
    rng = np.random.default_rng(seed)
    F = rng.uniform(-0.5, 0.5, size=n_nodes)
    I = rng.uniform(-0.5, 0.5, size=n_nodes)
    B = (F + 0.5 * I > float(seuil_b)).astype(float)
    return F, I, B


def evoluer(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    adjacency: np.ndarray,
    n_steps: int,
    cfg: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagateur de la dynamique réaction-diffusion couplée sur le graphe."""
    A = valider_adjacence(adjacency)
    field_f = np.asarray(F, dtype=float).copy()
    field_i = np.asarray(I, dtype=float).copy()
    field_b = np.asarray(B, dtype=float).copy()

    laplacian = laplacien_graphe(A)

    for _ in range(int(n_steps)):
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


def _norme_etat_FI(F_a: np.ndarray, I_a: np.ndarray, F_b: np.ndarray, I_b: np.ndarray) -> float:
    diff = np.concatenate([F_a - F_b, I_a - I_b])
    return float(np.linalg.norm(diff))


def mesures_dynamiques_etape(
    F0: np.ndarray,
    I0: np.ndarray,
    F1: np.ndarray,
    I1: np.ndarray,
    Fc: np.ndarray,
    Ic: np.ndarray,
    Ff: np.ndarray,
    If: np.ndarray,
) -> Dict[str, float]:
    """
    Calcule séparément les écarts dynamiques le long de X_0 -> X_c -> X_f :
      - delta_c : dist(X_c, X_0)
      - delta_f : dist(X_f, X_c)
      - delta_t : dist(X_f, X_0)
      - d1_ref_step : dist(X_f, X_1) avec X_1 = step(X_0, A_0)
      - d_rms_FI : delta_t / sqrt(2 * N)
    """
    n_nodes = len(F0)
    delta_c = _norme_etat_FI(Fc, Ic, F0, I0)
    delta_f = _norme_etat_FI(Ff, If, Fc, Ic)
    delta_t = _norme_etat_FI(Ff, If, F0, I0)
    d1_ref_step = _norme_etat_FI(Ff, If, F1, I1)
    d_rms = float(delta_t / math.sqrt(2.0 * n_nodes)) if n_nodes > 0 else 0.0

    return {
        "delta_c_FI": delta_c,
        "delta_f_FI": delta_f,
        "delta_t_FI": delta_t,
        "d1_ref_step_FI": d1_ref_step,
        "d_rms_FI": d_rms,
    }


# ============================================================================
# SIMULATION PRINCIPALE (V1.3)
# ============================================================================

def une_simulation(seed: int, cfg: SimulationConfig = CONFIG) -> Dict[str, Dict[str, Any]]:
    """Exécute une réplication complète avec états initiaux et durées partagés."""
    rng = np.random.default_rng(seed)

    # 1. Configuration géométrique initiale A_0
    points_init = rng.uniform(0.0, 1.0, size=(cfg.n_noeuds, 3))
    A0 = knn_graph(points_init, cfg.k_knn)
    l_ref = longueur_moyenne(points_init, A0)

    # 2. État dynamique initial commun X_0 et état décalé d'un pas X_1 = step(X_0, A_0)
    F, I, B = initialiser_champs(cfg.n_noeuds, seed + 100_000, seuil_b=cfg.seuil_b)
    F0, I0, B0 = evoluer(F, I, B, A0, cfg.t_initial, cfg)
    F1, I1, _ = evoluer(F0, I0, B0, A0, 1, cfg)

    # 3. Nombre d'arêtes actives non orientées dans A_0
    n_edges_actifs = int(np.count_nonzero(np.triu(A0 > 0.0, 1)))
    n_edges_cible = n_edges_actifs // 2

    A_pond = valider_adjacence(compresser_ponderee(A0, F0, I0, n_edges_cible, rng))
    A_alea = valider_adjacence(compresser_aleatoire(A0, n_edges_cible, rng))

    resultats: Dict[str, Dict[str, Any]] = {}

    for condition in cfg.conditions:
        if condition == "temoin_intact":
            # Sans compression, sans relaxation, sans reconstruction kNN
            Ac = A0.copy()
            points_finaux = points_init.copy()
            Af = A0.copy()
            relax_res = {
                "converged": True,
                "energy_diverged": False,
                "iterations_used": 0,
                "max_displacement": 0.0,
                "force_norm": 0.0,
                "energy_init": 0.0,
                "energy_final": 0.0,
                "energy_max": 0.0,
                "collision_count": 0,
                "residual_collisions": 0,
            }
        elif condition == "ponderee":
            Ac = A_pond
            relax_res = relaxer(points_init, Ac, l_ref, cfg)
            points_finaux = relax_res["points"]
            Af = knn_graph(points_finaux, cfg.k_knn)
        elif condition == "aleatoire":
            Ac = A_alea
            relax_res = relaxer(points_init, Ac, l_ref, cfg)
            points_finaux = relax_res["points"]
            Af = knn_graph(points_finaux, cfg.k_knn)
        elif condition == "ressorts_complets":
            Ac = A0.copy()
            relax_res = relaxer(points_init, Ac, l_ref, cfg)
            points_finaux = relax_res["points"]
            Af = knn_graph(points_finaux, cfg.k_knn)
        else:
            raise ValueError(f"Condition non prise en charge : {condition!r}")

        # Évolution dynamique : X_0 -> X_c (sur A^c) -> X_f (sur A^f)
        Fc, Ic, Bc = evoluer(F0, I0, B0, Ac, cfg.t_compresse, cfg)
        Ff, If, Bf = evoluer(Fc, Ic, Bc, Af, cfg.t_final, cfg)

        topo = analyser_transitions_graphes(A0, Ac, Af)
        dyn = mesures_dynamiques_etape(F0, I0, F1, I1, Fc, Ic, Ff, If)
        deplacement = float(np.sqrt(np.mean(np.sum((points_finaux - points_init) ** 2, axis=1))))
        stats_aretes_fin = statistiques_aretes(points_finaux, Af)

        resultats[condition] = {
            **topo,
            **dyn,
            "echec_rms": bool(dyn["d_rms_FI"] >= cfg.seuil_echec_rms),
            "distance_retour_B": float(np.mean(Bf != B0)),
            "deplacement_moyen": deplacement,
            "relaxation_converged": relax_res["converged"],
            "relaxation_energy_diverged": relax_res["energy_diverged"],
            "relaxation_iterations": relax_res["iterations_used"],
            "relaxation_max_displacement": relax_res["max_displacement"],
            "relaxation_force_norm": relax_res["force_norm"],
            "relaxation_energy_init": relax_res["energy_init"],
            "relaxation_energy_final": relax_res["energy_final"],
            "relaxation_energy_max": relax_res["energy_max"],
            "collision_count": relax_res["collision_count"],
            "residual_collisions": relax_res["residual_collisions"],
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
    """Intervalle de confiance bootstrap à 95 % avec validation des entrées."""
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap doit être >= 1.")
    values = np.asarray(valeurs, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Le tableau des valeurs pour le bootstrap est vide.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Le tableau des valeurs contient des éléments non finis.")

    rng = np.random.default_rng(seed)
    echantillons = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True)
    moyennes = np.mean(echantillons, axis=1)
    return float(np.percentile(moyennes, 2.5)), float(np.percentile(moyennes, 97.5))


def afficher_resultats(resultats: List[Dict[str, Dict[str, Any]]], cfg: SimulationConfig = CONFIG) -> None:
    """Mise en forme des résultats V1.3 avec vérification de cohérence."""
    if not resultats:
        raise ValueError("Aucun résultat à afficher.")
    for idx_rep, rep in enumerate(resultats):
        for cond in cfg.conditions:
            if cond not in rep:
                raise KeyError(f"Condition {cond!r} absente de la réplication {idx_rep}.")

    print(f"=== Simulation exploratoire de stéréomorphogenèse — {cfg.version} ===")
    print(f"Réplications : {len(resultats)} | Nœuds : {cfg.n_noeuds} | kNN : {cfg.k_knn}\n")

    titres = (
        f"{'Condition':18s}"
        f"{'Perte A0->Ac':>13s}"
        f"{'Perte Ac->Af':>13s}"
        f"{'Écart A0->Af':>13s}"
        f"{'Δc FI':>9s}"
        f"{'Δf FI':>9s}"
        f"{'Δt FI':>9s}"
        f"{'d_rms':>9s}"
        f"{'Échecs':>10s}"
    )
    print(titres)
    print("-" * len(titres))

    for condition in cfg.conditions:
        p_ac = np.array([r[condition]["perte_compression_A0_Ac"] for r in resultats])
        p_af = np.array([r[condition]["transformation_morpho_perte_Ac_Af"] for r in resultats])
        e_tot = np.array([r[condition]["ecart_total_perte_A0_Af"] for r in resultats])
        dc_arr = np.array([r[condition]["delta_c_FI"] for r in resultats])
        df_arr = np.array([r[condition]["delta_f_FI"] for r in resultats])
        dt_arr = np.array([r[condition]["delta_t_FI"] for r in resultats])
        rms_arr = np.array([r[condition]["d_rms_FI"] for r in resultats])
        echecs = int(np.sum(rms_arr >= cfg.seuil_echec_rms))

        print(
            f"{condition:18s}"
            f"{np.mean(p_ac):13.3f}"
            f"{np.mean(p_af):13.3f}"
            f"{np.mean(e_tot):13.3f}"
            f"{np.mean(dc_arr):9.3f}"
            f"{np.mean(df_arr):9.3f}"
            f"{np.mean(dt_arr):9.3f}"
            f"{np.mean(rms_arr):9.4f}"
            f"{echecs:5d}/{len(resultats):<4d}"
        )

        rho = spearman(e_tot, rms_arr)
        print(f"  └─ Spearman (écart total A0->Af vs d_rms FI) : {rho:.4f}")

    if "ponderee" in resultats[0] and "aleatoire" in resultats[0]:
        diff_etot = np.array(
            [r["ponderee"]["ecart_total_perte_A0_Af"] - r["aleatoire"]["ecart_total_perte_A0_Af"] for r in resultats]
        )
        diff_rms = np.array([r["ponderee"]["d_rms_FI"] - r["aleatoire"]["d_rms_FI"] for r in resultats])

        ci_etot = intervalle_bootstrap_moyenne(diff_etot)
        ci_rms = intervalle_bootstrap_moyenne(diff_rms)

        print("\n=== Comparaisons appariées : Pondérée - Aléatoire ===")
        print(f"Delta écart A0->Af : {np.mean(diff_etot):+.4f} [{ci_etot[0]:+.4f}, {ci_etot[1]:+.4f}]")
        print(f"Delta d_rms FI     : {np.mean(diff_rms):+.4f} [{ci_rms[0]:+.4f}, {ci_rms[1]:+.4f}]")


# ============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================================

def main() -> None:
    graines = list(range(CONFIG.n_graines))
    resultats = [une_simulation(seed, CONFIG) for seed in graines]

    afficher_resultats(resultats, CONFIG)

    archive = {
        "metadata": asdict(CONFIG),
        "results": resultats,
    }

    nom_fichier = "resultats_stereomorphogenese_v1_3.json"
    with open(nom_fichier, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

    print(f"\nDonnées archivées avec succès dans '{nom_fichier}'.")


if __name__ == "__main__":
    main()
