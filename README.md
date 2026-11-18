# Volume subjectif & géométrie des états vécus

Cadre computationnel et psychophysique destiné à reconstruire une géométrie relationnelle d'états subjectifs à partir de données comportementales, puis à évaluer sa reproductibilité hors échantillon.

Le projet distingue explicitement :

\[
\text{état subjectif}
\neq
\text{représentation calibrée}
\neq
\text{géométrie reconstruite}
\neq
\text{variable physique}.
\]

Le terme **volume subjectif** désigne ici un volume géométrique dans un espace de représentation des états subjectifs. Il ne constitue pas une mesure physique du volume de la conscience.

## Chaîne computationnelle

Le pipeline général est :

\[
\text{stimulus}
\rightarrow
\text{jugement perceptif}
\rightarrow
\text{JND}
\rightarrow
D_{\mathrm{subj}}
\xrightarrow{\mathcal C_\theta}
W_\theta
\xrightarrow{\mathcal E}
G
\xrightarrow{\mathcal V}
V_\psi.
\]

Les opérations sont :

- \(\mathcal C_\theta\) : calibration et construction de la représentation relationnelle ;
- \(\mathcal E\) : plongement ou construction géométrique ;
- \(\mathcal V\) : fonctionnelle de volume appliquée à une région reconstruite.

## Définition des objets

### \(D_{\mathrm{subj}}\)

\(D_{\mathrm{subj}}\) désigne une matrice de dissimilarités, de distances perceptives ou de réponses psychophysiques entre stimuli, états ou expériences rapportées.

Cette matrice doit être issue d'une tâche définie : jugement de différence, seuil différentiel, comparaison par paires, estimation de similarité ou mesure équivalente.

### \(W_\theta\)

\(W_\theta\) désigne une représentation relationnelle calibrée selon les paramètres \(\theta\).

Ces paramètres peuvent inclure :

- la normalisation ;
- la métrique utilisée ;
- les paramètres de calibration ;
- les ponts psychophysiques ;
- les contraintes d'alignement ;
- les paramètres de régularisation.

### \(G(x)\)

\(G(x)\) désigne une géométrie construite à partir des représentations calibrées des états subjectifs.

Il ne s'agit pas d'une géométrie ontologique des états vécus ni d'une métrique physique de l'espace-temps.

### \(V_\psi\)

\(V_\psi\) désigne le volume d'une région reconstruite dans l'espace géométrique de représentation \(\psi\).

Il s'agit d'une grandeur du modèle. Elle peut être comparée entre conditions ou sujets, mais ne constitue pas directement une mesure physique de la conscience.

## Architecture du dépôt

```text
├── volume_subjectif_formats.py
│   └── Représentations W_theta, volumes relatifs V_k,
│       dimension effective d_eff et RSA partielle
│
├── local_jnd_bridge_calibration.py
│   └── Calibration locale par JND, alignement Procrustes,
│       champ métrique construit G(x) et distances intégrées
│
├── pipeline_volume_subjectif_complet.py
│   └── Chaînage calibration–géométrie–volume,
│       intégration ellipsoïdale et export JSON/CSV
│
├── conjunctive_validation_and_power.py
│   └── Validation conjonctive I_J, I_G, I_V
│       et benchmark de puissance Monte-Carlo
│
├── nested_model_and_eeg_rsa.py
│   └── Modèles emboîtés figés M0, M1, M2,
│       prédiction multifenêtre et permutation appariée
│
├── models/
│   ├── vacuite_holotelegraphique_unifie.py
│   │   └── Générateur exploratoire de variables candidates q(t)
│   │
│   └── v95_skyrme_soliton.py
│       └── Modèle mathématique auxiliaire de soliton de Skyrme
│
├── REGISTRE_REVENDICATIONS.md
│   └── Classification des revendications :
│       mathématique, numérique, empirique ou hypothétique
│
└── LICENSE
    └── Licence MIT
```

## Rôle des modules

### `volume_subjectif_formats.py`

Ce module fournit les formats et métriques de représentation :

- construction de \(W_\theta\) ;
- calcul de distances relationnelles ;
- estimation de la dimension effective \(d_{\mathrm{eff}}\) ;
- calcul de volumes relatifs \(V_k\) ;
- analyses de type RSA ou RSA partielle.

La sortie est une représentation computationnelle. Elle ne constitue pas une mesure directe d'un état mental au sens ontologique.

### `local_jnd_bridge_calibration.py`

Ce module utilise des ponts de calibration locaux, notamment issus de seuils différentiels perceptibles, afin de construire une relation entre variations de stimulus et variations de représentation.

Le schéma conceptuel est :

\[
\text{stimulus}
\rightarrow
\text{jugement perceptif}
\rightarrow
\text{JND}
\rightarrow
D_{\mathrm{subj}}
\rightarrow
W_\theta.
\]

Les ponts utilisés pour la calibration doivent être séparés des données utilisées pour l'évaluation finale.

### `pipeline_volume_subjectif_complet.py`

Ce module réalise le chaînage principal :

1. préparation des données ;
2. calibration des représentations ;
3. construction de la géométrie locale ;
4. intégration des distances ;
5. estimation du volume ;
6. export des résultats.

Les paramètres ajustés dans les données d'entraînement doivent rester figés pendant l'évaluation hors échantillon.

### `conjunctive_validation_and_power.py`

La validation conjonctive est représentée par :

\[
I_{\mathrm{joint}}
=
I_J
\land
I_G
\land
I_V.
\]

Les critères doivent être définis explicitement dans le protocole et dans le registre des revendications.

Une notation binaire possible est :

\[
I_J =
\begin{cases}
1 & \text{si le critère psychophysique est satisfait},\\
0 & \text{sinon},
\end{cases}
\]

avec des définitions analogues pour \(I_G\) et \(I_V\).

Le module peut fournir :

- une puissance marginale ;
- une puissance conjointe ;
- des intervalles de Wilson ;
- la composante limitante ;
- des scénarios Monte-Carlo de référence.

La puissance conjointe décrit la performance du protocole. Elle ne constitue pas une probabilité de vérité du modèle.

Commande :

```bash
python3 conjunctive_validation_and_power.py
```

### `nested_model_and_eeg_rsa.py`

Le module compare des modèles emboîtés :

\[
M_0=C,
\]

\[
M_1=C+V_\psi,
\]

\[
M_2=C+V_\psi+q.
\]

Les paramètres de standardisation, de résidualisation et de régression doivent être ajustés dans l'échantillon d'entraînement uniquement.

Les contrastes prédictifs sont :

\[
\Delta S_V
=
\mathrm{MSE}(M_0)
-
\mathrm{MSE}(M_1),
\]

\[
\Delta S_q
=
\mathrm{MSE}(M_1)
-
\mathrm{MSE}(M_2).
\]

Une valeur positive indique une réduction de l'erreur hors échantillon.

Les entrées attendues sont :

```text
C_train, C_test       : covariables de contrôle
V_psi_train, V_psi_test
Q_train, Q_test       : variables candidates
Y_train, Y_test       : cibles multifenêtres
pair_ids              : unités échangeables du test
```

Pour une analyse EEG/MEG au sens strict, chaque colonne de \(Y\) doit correspondre à une RDM vectorisée à une fenêtre temporelle donnée :

```text
Y[observation, fenêtre] =
    partie_supérieure_vectorisée(RDM_EEGMEG[fenêtre])
```

Le module ne construit pas les RDM EEG/MEG. Il ne remplace pas :

- le prétraitement EEG/MEG ;
- la définition des unités d'analyse ;
- le choix des distances ;
- la construction des RDM ;
- la gestion de la dépendance entre cellules d'une même RDM.

Les cellules d'une même RDM ne doivent pas être permutées comme si elles étaient des observations indépendantes.

Commande :

```bash
python3 nested_model_and_eeg_rsa.py
```

### `models/vacuite_holotelegraphique_unifie.py`

Ce module simule un champ complexe discret avec une connexion de jauge covariante.

Il produit des variables candidates dynamiques \(q(t)\), par exemple :

\[
q(t)=
[
\mathrm{IPR},
\mathrm{cohérence},
\mathrm{courbure},
\mathrm{énergie\ diagnostique}
].
\]

La holonomie associée à `flux_theta` est un paramètre de configuration du modèle. Elle ne doit pas être utilisée comme prédicteur individuel si elle reste constante entre les observations.

Les métadonnées de configuration peuvent contenir :

```text
flux_theta
holonomie_mod_2pi
n_x
length
dt
bandwidth_init
gamma_dissip
retroaction
```

Le module ne constitue :

- ni une mesure de la conscience ;
- ni une mesure EEG/MEG ;
- ni une mesure d'énergie biologique ;
- ni une courbure relativiste ;
- ni une preuve d'un mécanisme physique de l'expérience vécue.

Commande :

```bash
python3 models/vacuite_holotelegraphique_unifie.py
```

### `models/v95_skyrme_soliton.py`

Ce module résout numériquement un modèle radial de Skyrme dans l'ansatz hérisson.

Il peut produire :

- un profil radial \(F(r)\) ;
- une charge topologique numérique ;
- des contributions énergétiques ;
- un rapport à une borne de référence ;
- des contrôles de convergence.

Il s'agit d'un modèle mathématique et numérique auxiliaire. Sa présence dans le dépôt ne constitue pas un lien causal avec \(V_\psi\), les états subjectifs ou la conscience.

## Validation et reproductibilité

Avant toute interprétation, les contrôles suivants sont recommandés :

```bash
python3 -m py_compile \
  volume_subjectif_formats.py \
  local_jnd_bridge_calibration.py \
  pipeline_volume_subjectif_complet.py \
  conjunctive_validation_and_power.py \
  nested_model_and_eeg_rsa.py
```

Pour les modules du répertoire `models/` :

```bash
python3 -m py_compile \
  models/vacuite_holotelegraphique_unifie.py \
  models/v95_skyrme_soliton.py
```

Les analyses doivent conserver :

- la version du code ;
- les paramètres numériques ;
- la graine aléatoire ;
- les données d'entrée ;
- la définition des unités échangeables ;
- les ponts utilisés pour la calibration ;
- les données d'entraînement et de test ;
- les sorties brutes ;
- les critères d'échec ;
- les transformations appliquées ;
- le statut épistémique de chaque résultat.

## Registre des revendications

Chaque revendication doit préciser :

```text
claim_id
texte de la revendication
fichier source
entrée
transformation
sortie
niveau épistémique
jeu de données
critère de validation
critère d'échec
limites
```

Les niveaux épistémiques sont :

| Niveau | Définition |
|---|---|
| Mathématique | Définition formelle ou propriété analytique |
| Numérique | Résultat d'une simulation ou d'une discrétisation |
| Empirique | Résultat obtenu sur des données observées |
| Hypothétique | Interprétation ou mécanisme proposé mais non établi |

## Limites d'interprétation

Le dépôt ne démontre pas :

- une topologie ontologique de la conscience ;
- une énergie de la conscience ;
- une courbure de l'espace-temps induite par l'observateur ;
- une dimension physique supplémentaire ;
- un accès à une structure cachée du réel ;
- une équivalence entre \(q(t)\) et un état subjectif ;
- une équivalence entre \(V_\psi\) et un volume physique de la conscience.

Une variable théorique \(q(t)\) ne peut être considérée comme empiriquement informative que si elle :

1. est définie avant l'analyse confirmatoire ;
2. est calculée sans fuite d'information ;
3. améliore une prédiction hors échantillon ;
4. résiste aux contrôles appropriés ;
5. se réplique sur des données indépendantes.

La distinction fondamentale reste :

\[
\text{numérique}
\rightarrow
\text{géométrique}
\rightarrow
\text{phénoménologique}
\rightarrow
\text{physique}.
\]

Chaque transition doit être documentée par une opération explicite et ne peut être remplacée par une analogie sémantique.

## Licence

Le dépôt est distribué sous licence MIT. Voir le fichier `LICENSE`.
