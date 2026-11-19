# PROTOCOL.md — Stimuli-ponts et référentiel Procrustes

## Rôle

Un *pont* est un stimulus mesuré dans au moins deux conditions. Il sert
exclusivement à (1) estimer la transformation Procrustes, (2) évaluer
la transformation figée sur des ponts tenus à l'écart. Ce n'est pas une
donnée de validation psychophysique.

## Sélection

- Pont physiquement identique entre conditions (même stimulus, même protocole).
- Sélection antérieure à l'analyse, non révisée a posteriori.
- Minimum recommandé : ≥ d+2 ponts d'entraînement, ≥ 3 tenus à l'écart,
  d = dimension effective.

## Split

- Entraînement : ponts servant à l'estimation Procrustes.
- Tenus à l'écart : ponts servant uniquement à l'évaluation de la
  transformation figée (erreur point-à-point, distances, volume).
- Consigner identifiants et seed.

## Correspondances

- Chaque pont porte le même identifiant dans toutes les conditions.
- Correspondances vérifiées avant analyse ; toute modification invalide
  le référentiel et doit être tracée.

## Transformations autorisées

- Réflexion : uniquement si la chiralité n'est pas une propriété du
  domaine. Sinon contraindre det(R) = +1.
- Échelle : autorisée ; reporter s.
- Rotation et translation : toujours estimées, jamais fixées a priori.

## Centrage

- `common` (référence) : un centre unique estimé sur les ponts
  d'entraînement, appliqué à toutes les conditions, y compris nouvelles.
- `per_condition` (sensibilité) : centres par condition. Lève une erreur
  explicite si une condition est totalement absente de l'entraînement.

## Reproductibilité

Consigner : seed, liste des ponts d'entraînement, liste des ponts tenus
à l'écart, R, t, s, `centering_mode`, et le rapport `check_D_subj`.
