
# Volume subjectif & géométrie des états vécus

> **Terminologie** — \(V_\psi\) désigne un volume géométrique calculé dans
> un espace de représentation calibré à partir de données psychophysiques.
> Il ne constitue ni un volume physique, ni une mesure directe de la
> conscience, ni une propriété neuronale établie.
> 
Cadre computationnel et psychophysique destiné à reconstruire une géométrie relationnelle d'états subjectifs à partir de données comportementales, puis à évaluer sa reproductibilité hors échantillon.

Le projet distingue explicitement :

\[\text{état subjectif} \neq \text{représentation calibrée} \neq \text{géométrie reconstruite} \neq \text{variable physique}.\]

Le terme **volume subjectif** désigne ici un volume géométrique dans un espace de représentation des états subjectifs. Il ne constitue pas une mesure physique du volume de la conscience.

## Chaîne computationnelle

Le pipeline général est :

\[\text{stimulus} \rightarrow \text{jugement perceptif} \rightarrow \text{JND} \rightarrow D_{\mathrm{subj}} \xrightarrow{\mathcal C_\theta} W_\theta \xrightarrow{\mathcal E} G \xrightarrow{\mathcal V} V_\psi.\]

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
