"""
quality_controls.py

Contrôles de cohérence minimaux sur D_subj avant calibration.
L'inégalité triangulaire est un DIAGNOSTIC, pas une exigence :
une dissimilarité empirique peut ne pas être métrique stricte.
"""

from __future__ import annotations
import numpy as np


def check_D_subj(D, *, name="D_subj", tol_sym=1e-6, tol_tri=1e-6):
    D = np.asarray(D, dtype=float)
    report = {}

    assert D.ndim == 2 and D.shape[0] == D.shape[1], f"{name} doit être carrée"
    n = D.shape[0]
    report["n"] = n

    assert np.isfinite(D).all(), f"{name} contient des valeurs non finies"

    report["diag_max_abs"] = float(np.max(np.abs(np.diag(D))))
    assert report["diag_max_abs"] <= tol_sym, f"{name} : diagonale non nulle"

    report["asymmetry_max"] = float(np.max(np.abs(D - D.T)))
    assert report["asymmetry_max"] <= tol_sym, f"{name} : non symétrique"

    report["min_value"] = float(D.min())
    assert report["min_value"] >= -tol_sym, f"{name} : valeurs négatives"

    # doublons potentiels (points confondus), diagnostic
    zero_off = (np.abs(D) < 1e-12) & ~np.eye(n, dtype=bool)
    report["n_zero_offdiag_pairs"] = int(zero_off.sum() // 2)

    # inégalité triangulaire — diagnostic
    tri_viol, tri_worst = 0, 0.0
    for i in range(n):
        lhs = D[i, :][None, :]
        rhs = D[i, :][:, None] + D
        viol = (lhs - rhs) / (D.max() + 1e-12)
        tri_worst = max(tri_worst, float(viol.max()))
        tri_viol += int((viol > tol_tri).sum())
    report["tri_violations_frac"] = tri_viol / (n ** 3)
    report["tri_worst_relative"] = tri_worst
    if tri_viol > 0:
        report["warning"] = (
            "Inégalité triangulaire non respectée : D_subj n'est pas "
            "métrique stricte. Diagnostic seulement, ne bloque pas l'analyse."
        )
    return report
