# Architecture du dépôt

Séparation stricte entre **pipeline empirique principal** et **modules exploratoires**.

```
├── pipeline_principal/          # chaîne empirique officielle
│   ├── local_jnd_bridge_calibration.py
│   ├── volume_subjectif_formats.py
│   ├── pipeline_volume_subjectif_complet.py
│   ├── conjunctive_validation_and_power.py
│   ├── quality_controls.py
│   └── test_synthetic_alignment.py
│
├── exploratory/                 # prototypes (non importés par le principal)
│   ├── vacuite_holotelegraphique_unifie.py
│   ├── v95_skyrme_soliton.py
│   └── README.md
│
├── PROTOCOL.md
├── REGISTRE_REVENDICATIONS.md
└── README.md
```

---

## 1. Règle d’or

`pipeline_principal/` **n’importe jamais** `exploratory/`.

Vérification :

```bash
grep -rn "from exploratory" pipeline_principal/
grep -rn "import exploratory" pipeline_principal/
```

Les deux commandes doivent rester vides. Aucune dépendance cachée vers les modules spéculatifs.

---

## 2. `pipeline_principal/` — chaîne empirique

| Fichier | Rôle | Lien cadre |
| :--- | :--- | :--- |
| `local_jnd_bridge_calibration.py` | Procrustes figé, \(G(x)\), distances intégrées, volumes locaux, calibration linéaire / monotone, split calib–val–test | \(W_\theta\), \(G(x)\), \(V_\psi\) (estimation) |
| `volume_subjectif_formats.py` | Schémas et formats de \(V_\psi\) | Sorties volume standardisées |
| `pipeline_volume_subjectif_complet.py` | Enchaînement bout-en-bout jusqu’à \(V_\psi\) | Chaîne \(D_{\mathrm{subj}} \to W_\theta \to G \to V_\psi\) |
| `conjunctive_validation_and_power.py` | \(I_J\), \(I_G\), \(I_V\), critère conjonctif, simulation de puissance | Confirmation préenregistrée |
| `quality_controls.py` | Contrôles qualité, invariants numériques | Garde-fous |
| `test_synthetic_alignment.py` | Récupération Procrustes synthétique (held-out) | Audit géométrique, pas validation psychophysique |

**Flux typique**

```text
Données brutes
  → local_jnd_bridge_calibration   (alignement, G, distances, volumes, échelles)
  → volume_subjectif_formats       (normalisation des sorties V_ψ)
  → pipeline_volume_subjectif_complet
  → conjunctive_validation_and_power   (I_J ∧ I_G ∧ I_V + puissance)
  → quality_controls
```

Les tests synthétiques (`test_synthetic_alignment`) ne font pas partie de la décision confirmatoire ; ils valident la géométrie d’alignement.

---

## 3. `exploratory/` — hors protocole

| Contenu | Statut |
| :--- | :--- |
| `vacuite_holotelegraphique_unifie.py` | Prototype / spéculatif |
| `v95_skyrme_soliton.py` | Prototype / spéculatif |
| `README.md` | Rappel : non importé par `pipeline_principal/` |

Ces modules peuvent évoluer librement. Ils **ne portent aucune revendication empirique** du registre et n’entrent pas dans `PROTOCOL.md` comme étapes de décision.

---

## 4. Documents racine

| Fichier | Rôle |
| :--- | :--- |
| `PROTOCOL.md` | Protocole préenregistré (splits, marges \(\delta_J,\delta_G,\rho_G,\delta_V\), critères) |
| `REGISTRE_REVENDICATIONS.md` | Ce qui est affirmé / non affirmé (aligné sur le cadre final) |
| `README.md` | Vue d’ensemble du dépôt et point d’entrée |

---

## 5. Alignement avec le cadre théorique final

| Élément du cadre | Localisation dans le dépôt |
| :--- | :--- |
| Estimation \(G(x)\), \(V_\psi\) | `local_jnd_bridge_calibration.py`, pipeline volume |
| Critère \(I_J \wedge I_G \wedge I_V\) | `conjunctive_validation_and_power.py` |
| Séparation découverte / réplication | `PROTOCOL.md` + splits dans calibration / validation |
| Exclusion de la métaphysique du protocole | `REGISTRE_REVENDICATIONS.md` |
| Isolation du spéculatif | `exploratory/` + règle d’import |

---

## 6. Invariants d’architecture

1. **Frontière nette** — seul `pipeline_principal/` produit les résultats confirmatoires.
2. **Pas d’import exploratoire** — vérifié par grep (CI recommandée).
3. **Estimation ≠ confirmation** — calibration et volumes d’un côté ; \(I_J\), \(I_G\), \(I_V\) de l’autre.
4. **Préenregistrement** — marges et critères dans `PROTOCOL.md`, pas dans le code exploratoire.
5. **Registre** — toute affirmation publique doit figurer dans `REGISTRE_REVENDICATIONS.md` et être supportée par le pipeline principal.

---

*Architecture alignée sur `cadre_interface_phenomenale_FINAL.md`.*
