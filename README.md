# Volume subjectif & géométrie des états vécus

Cadre computationnel et psychophysique pour estimer la géométrie relationnelle d'états subjectifs ($D_{\mathrm{subj}} \to W_\theta \to G(x) \to V_\psi$), contrôler sa reproductibilité hors échantillon par critère conjonctif, et confronter des variables théoriques candidates $q(t)$ à des données comportementales ou EEG/MEG.

## Architecture du dépôt

```text
├── volume_subjectif_formats.py             # Métrique W_theta, volume relatif V_k, dimension effective d_eff, RSA partielle
├── local_jnd_bridge_calibration.py         # Alignement Procrustes sur ponts, champ métrique local G(x), distances intégrées
├── pipeline_volume_subjectif_complet.py    # Intégration volumique par ellipsoïde, séparation stricte des ponts, export JSON/CSV
├── conjunctive_validation_and_power.py     # Validation conjonctive I_J * I_G * I_V et benchmark de puissance Monte-Carlo
├── nested_model_and_eeg_rsa.py             # Modèles emboîtés figés M0 -> M1 -> M2 et analyse multifenêtre compatible avec des RDM
├── models/
│   ├── vacuite_holotelegraphique_unifie.py # Couche théorique exploratoire générant des variables candidates q(t)
│   └── v95_skyrme_soliton.py               # Modèle mathématique et numérique d'un soliton de Skyrme (B = 1)
├── REGISTRE_REVENDICATIONS.md              # Classification épistémique : mathématique, numérique, empirique, hypothétique
├── CADRE_INTERFACE_PHENOMENALE.md          # Cadre conceptuel de l'interface phénoménale et continuité incarnée
└── LICENSE                                 # Licence MIT
