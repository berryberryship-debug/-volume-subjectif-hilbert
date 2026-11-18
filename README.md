# Volume subjectif & géométrie des états vécus

Protocole Python pour apprendre une métrique semi-définie positive $W_\theta = L_\theta L_\theta^\top$ à partir de dissemblances subjectives observées, estimer l'étendue (**volume subjectif** $V_k$) et la répartition des modes actifs (**ouverture** via la dimension effective $d_{\mathrm{eff}}$), et calibrer l'échelle sur des stimuli-ponts indépendants.

## Structure des scripts

* **`volume_subjectif_formats.py`** : adaptation aux formats de recueil (notes de similarité, choix en triplets, arrangements spatiaux), estimation de $L_\theta$ et $W_\theta$, calcul de $V_k$ et $d_{\mathrm{eff}}$, comparaison RSA avec contrôle des covariables et calibration d'une unité psychophysique $U_\psi$.
* **`local_jnd_bridge_calibration.py`** : alignement Procrustes des arrangements spatiaux sur stimuli-ponts, estimation de métriques locales directionnelles $G(x)$ avec shrinkage, distances intégrées le long d'un chemin et validation sur des ponts tenus à l'écart.
* **`pipeline_volume_subjectif_complet.py`** : assemblage de bout en bout avec séparation stricte par nœuds-ponts (calibration / validation / test), intégration volumique par échantillonnage uniforme dans l'ellipsoïde, comparaison avec une densité globale équivalente, test de permutation (incluant plans appariés et $T_{\max}$) et export atomique JSON/CSV.

## Principes méthodologiques

* **Partir des dissemblances avant d'étiqueter les axes :** les relations de distance et les directions d'étirement ou de contraction précèdent l'interprétation des axes (invariante sous rotation orthogonale).
* **Séparation stricte des ensembles :** les stimuli-ponts utilisés pour l'alignement Procrustes et la calibration d'échelle ne servent pas à ajuster $W_\theta$, et les conditions comparées sont évaluées hors échantillon.
* **Limites d'interprétation :** les volumes et indicateurs d'ouverture dépendent du protocole, du choix du rang $k$, de la fiabilité des mesures et de la qualité de la calibration psychophysique ; ils décrivent une géométrie de comparaison plutôt qu'un espace physique autonome.

## Exécution rapide

```bash
python3 volume_subjectif_formats.py
python3 local_jnd_bridge_calibration.py
python3 pipeline_volume_subjectif_complet.py
