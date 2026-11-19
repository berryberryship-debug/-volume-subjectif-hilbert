# Interface phénoménale, continuité incarnée et géométrie mesurable

**Cadre théorique et protocole opérationnel — version finale**

---

## 1. Thèse directrice (portée limitée)

Nous traitons l’espace phénoménal comme une **interface spatiale fonctionnelle** : un format perceptif qui sélectionne, organise et stabilise des relations pertinentes pour l’action et la viabilité de l’organisme, sans prétendre en déduire la structure ontologique du réel.

**Ce que l’étude affirme**

- Les seuils de discrimination peuvent être résumés localement par une métrique \(G(x)\).
- Des volumes relatifs \(V_\psi\) dans cette géométrie sont estimables et peuvent être prédictifs hors échantillon.
- La stabilité de \(G\) et le gain prédictif de \(V_\psi\) sont testables par des critères préenregistrés.

**Ce que l’étude n’affirme pas**

- Que l’espace perçu soit « faux » ou qu’il masque une réalité quantique.
- Que \(G(x)\) ou \(V_\psi\) soient une métrique d’espace-temps physique.
- Qu’une lecture idéaliste ou anti-réaliste suive logiquement des mesures psychophysiques.

Le masquage perceptif délimite le format dans lequel l’expérience apparaît ; il n’est pas interprété comme accès à une ontologie cachée.

---

## 2. Repères philosophiques et limites de portée

| Approche | Usage dans le cadre | Limite méthodologique |
| :--- | :--- | :--- |
| **Kant** | Rappel que l’espace d’apparition a des conditions formelles. | Ne constitue pas une théorie neurobiologique testable dans ce protocole. |
| **Metzinger** | Transparence du modèle phénoménal de soi et du monde. | N’établit pas le statut ontologique de l’espace perçu. |
| **Hoffman** | Inspiration : interface orientée vers la *fitness*. | Dépend des hypothèses du modèle évolutif ; non testé directement ici. |

### Interprétation métaphysique (externe au protocole)

Les lectures idéalistes (par exemple certaines formulations de type Kastrup) relèvent d’une **interprétation métaphysique externe**. Elles ne découlent pas des mesures psychophysiques, ne sont pas des prémisses des estimateurs, et ne sont pas requises pour tester les hypothèses géométriques du dépôt.

---

## 3. Hypothèse de continuité incarnée (formulation statistique)

Plutôt qu’importer sans démonstration le formalisme d’une symétrie de jauge physique, la stabilité de l’expérience à travers des variations sensorielles et motrices est posée comme hypothèse d’**invariant relationnel de continuité incarnée** (métaphore heuristique uniquement).

**Formulation opérationnelle.** Sous des transformations sensori-motrices bornées, la structure relationnelle des discriminations — résumée par \(G\) et par des indices de similarité préenregistrés — reste stable au sens des critères \(I_J\) et \(I_G\) (équivalence des JND, stabilité log-euclidienne de \(G\), ICC interséance).

Sur le plan statistique, la frontière d’échange organisme–milieu peut être traitée via une **structure de dépendance conditionnelle** (couverture de Markov au sens strict). Cette notion n’est pas assimilée à une membrane anatomique, une frontière phénoménologique directe ou une limite fréquentielle. L’interface perceptive est un mécanisme fonctionnel contribuant à la régulation et à la viabilité ; seules ses conséquences mesurables sur \(G\) et \(V_\psi\) entrent dans les tests.

---

## 4. Pont formel vers le volume subjectif (\(G(x)\), \(V_\psi\))

La chaîne empirique s’écrit :

\[
D_{\mathrm{subj}} \longrightarrow W_\theta \longrightarrow G(x) \longrightarrow V_\psi
\]

| Symbole | Définition opérationnelle | Module |
| :--- | :--- | :--- |
| \(D_{\mathrm{subj}}\) | Dissimilarités ou seuils JND observés sur un ensemble fixe de stimuli / états. | Psychophysique |
| \(W_\theta\) | Alignement et calibration (Procrustes figé, échelle monotone ou linéaire). | `local_jnd_bridge_calibration` |
| \(G(x)\) | Métrique locale SPD estimée aux ancres, interpolée (linéaire ou log-euclidienne). | `fit_local_metric_at_anchor`, interpolation |
| \(V_\psi\) | Volume relatif dans cette géométrie, typiquement \(\propto \mathbb{E}[\sqrt{\det G(z)}]\) sur une région. | `local_volume_in_region` / intégration ellipsoïdale |

**Statut des quantités**

- \(G(x)\) mesure une géométrie locale des seuils de discrimination ; ce n’est pas une métrique \((M, g_{\mu\nu})\) d’espace-temps.
- \(V_\psi\) mesure l’étendue relative d’une région d’états dans cette géométrie calibrée ; ce n’est pas automatiquement un invariant topologique.

---

## 5. Critère confirmatoire conjonctif (préenregistré)

La confirmation empirique repose sur trois estimands indépendants, évalués en réplication avec paramètres et marges fixés avant analyse :

| Volet | Estimand | Critère (bootstrap) |
| :--- | :--- | :--- |
| \(I_J\) | Biais moyen \(\widehat{\mathrm{JND}}_{\mathrm{pred}} - \mathrm{JND}_{\mathrm{obs}}\) | \(\mathrm{CI}_{95\%}(\mathrm{biais}) \subset [-\delta_J,+\delta_J]\) |
| \(I_G\) | Stabilité log-euclidienne de \(G\) + ICC interséance | borne sup. de \(\mathrm{CI}_{95\%}(Q_{0.95}(d_G)) \le \delta_G\) **et** borne inf. de \(\mathrm{CI}_{95\%}(\mathrm{ICC}) \ge \rho_G\) |
| \(I_V\) | Gain prédictif hors échantillon de \(V_\psi\) au-delà de \(M_0\) | borne inf. de \(\mathrm{CI}_{95\%}(\Delta S_V) > \delta_V\) |

\[
I_{\mathrm{conjoint}} = I_J \times I_G \times I_V
\]

On ne confirme le modèle géométrique que si les trois volets passent. Les marges \(\delta_J\), \(\delta_G\), \(\rho_G\), \(\delta_V\), le plan de split découverte / validation / test et les procédures de bootstrap ou de permutation sont préenregistrés.

**Modules associés**

- Estimation : `local_jnd_bridge_calibration` (v1.1.0)
- Confirmation : `evaluate_jnd_equivalence`, `evaluate_field_stability`, `evaluate_volume_oos_gain`, `evaluate_conjunctive_criterion`
- Puissance : `run_joint_power_simulation`, `run_three_reference_benchmarks`

---

## 6. Opérationnalisation topologique (optionnelle, distincte)

Une déformation métrique \(\Delta G(x)\) ou \(\Delta V_\psi\) **n’implique pas** un changement topologique. Pour tester si la structure relationnelle demeure invariante ou se reconfigure (attention, états modifiés, synesthésie, etc.), un volet topologique peut être ajouté **indépendamment** du critère conjonctif :

1. **Espace d’états et dissimilarité** — matrice \(D_{\mathrm{subj}}\) calibrée sur un ensemble fixe de stimuli ou d’états.
2. **Filtration** — famille croissante de graphes de similarité ou de complexes simpliciaux construits sur \(D_{\mathrm{subj}}\).
3. **Invariants calculés** — composantes connexes (\(\beta_0\)), cycles persistants (\(\beta_1\)), diagrammes d’homologie persistante.
4. **Critère de stabilité** — persistance des caractéristiques dans la filtration préenregistrée, évaluée par bootstrap au niveau du participant et par réplication indépendante.

Le succès de \(I_{\mathrm{conjoint}}\) n’entraîne pas l’invariance topologique, et réciproquement.

---

## 7. Tests prédictifs emboîtés (\(M_0 \to M_1 \to M_2\))

Toute variable candidate \(q\) issue d’un modèle de continuité relationnelle ou d’une simulation dynamique est confrontée hors échantillon aux covariables de base \(C\) et au volume calibré \(V_\psi\) :

\[
\begin{align*}
M_0 &: \quad Y \sim C \\
M_1 &: \quad Y \sim C + V_\psi \\
M_2 &: \quad Y \sim C + V_\psi + q_{\perp C}
\end{align*}
\]

- Paramètres estimés sur l’échantillon de **découverte**, évalués sur l’échantillon de **test** (coefficients figés).
- \(q\) n’acquiert un statut empirique que si \(M_2\) améliore significativement la prédiction de \(Y\) au-delà de \(M_1\).
- \(Y\) peut être une RDM subjective indépendante ou une RDM EEG/MEG par fenêtre temporelle.
- Correction pour tests multiples préenregistrée (par exemple max-\(T\)).

---

## 8. Exigences qui distinguent ce cadre d’un manifeste

| Exigence | Mise en œuvre |
| :--- | :--- |
| Hypothèses falsifiables | \(I_J\), \(I_G\), \(I_V\) avec marges numériques |
| Préenregistrement | Marges, splits, critères, corrections |
| Séparation estimation / confirmation | Calibration en découverte ; validation conjonctive en réplication |
| Statut des quantités | \(G\), \(V_\psi\) définis par des estimateurs, non par une métaphysique |
| Limites explicites | Pas d’ontologie dérivée ; pas de \(g_{\mu\nu}\) ; topologie optionnelle et séparée |
| Contrôles négatifs | Scénario nul et dérive intercohorte dans les simulations de puissance |

---

## 9. Formulation courte (résumé / abstract)

Nous opérationnalisons l’espace phénoménal comme interface spatiale fonctionnelle via une géométrie locale des seuils \(G(x)\) et des volumes relatifs \(V_\psi\). Un critère conjonctif préenregistré teste en réplication (i) l’équivalence des JND, (ii) la stabilité log-euclidienne et l’ICC de \(G\), (iii) le gain prédictif hors échantillon de \(V_\psi\). Les lectures métaphysiques sont exclues du protocole empirique ; les quantités \(G\) et \(V_\psi\) ne sont pas interprétées comme métrique d’espace-temps ni comme invariants topologiques automatiques.

---

## 10. Livrables techniques associés

| Livrable | Fichier | Rôle |
| :--- | :--- | :--- |
| Calibration | `local_jnd_bridge_calibration.py` (v1.1.0) | Procrustes, \(G(x)\), distances, volumes, échelles |
| Validation conjonctive | module \(I_J\), \(I_G\), \(I_V\) + simulation de puissance | Confirmation préenregistrée |
| Test synthétique Procrustes | `test_synthetic_alignment.py` | Récupération géométrique contrôlée |
| Cadre théorique | le présent document | Portée, hypothèses, critères |

---

*Version finale campée — alignée sur les modules d’estimation et de validation du dépôt.*
