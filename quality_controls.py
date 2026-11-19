"""
quality_controls.py

Contrôles de cohérence minimaux sur D_subj avant calibration.

L'inégalité triangulaire est un DIAGNOSTIC, pas une exigence :
une dissimilarité empirique peut ne pas être métrique stricte.
"""

from __future__ import annotations

import numpy as np

__all__ = ["check_D_subj"]


def check_D_subj(D, *, name="D_subj", tol_sym=1e-6, tol_tri=1e-6):
    """Vérifie la cohérence de D_subj et retourne un rapport.

    Lève ValueError sur les violations bloquantes (carré, fini, diagonale
    nulle, symétrie, positivité). Retourne un rapport incluant le
    diagnostic de l'inégalité triangulaire, qui ne bloque pas l'analyse.
    """
    D = np.asarray(D, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{name} doit être une matrice carrée ; reçu shape={D.shape}")
    n = D.shape[0]

    if not np.isfinite(D).all():
        raise ValueError(f"{name} contient des valeurs non finies")

    diag_max = float(np.max(np.abs(np.diag(D))))
    if diag_max > tol_sym:
        raise ValueError(f"{name} : diagonale non nulle (max={diag_max})")

    asym = float(np.max(np.abs(D - D.T)))
    if asym > tol_sym:
        raise ValueError(f"{name} : matrice non symétrique (max asym={asym})")

    dmin = float(D.min())
    if dmin < -tol_sym:
        raise ValueError(f"{name} : valeurs négatives (min={dmin})")

    off = ~np.eye(n, dtype=bool)
    n_zero = int(((np.abs(D) < 1e-12) & off).sum() // 2)

    scale = float(D.max()) + 1e-12
    worst = 0.0
    n_viol = 0
    for i in range(n):
        viol = (D[i, :][None, :] - (D[i, :][:, None] + D)) / scale
        m = float(viol.max())
        if m > worst:
            worst = m
        n_viol += int((viol > tol_tri).sum())

    return {
        "n": n,
        "diag_max_abs": diag_max,
        "asymmetry_max": asym,
        "min_value": dmin,
        "n_zero_offdiag_pairs": n_zero,
        "tri_violations_frac": n_viol / (n ** 3),
        "tri_worst_relative": worst,
        "tri_is_diagnostic_only": True,
    }
