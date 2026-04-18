"""
Rotation curve slope test: Does the outer slope depend on mass
after controlling for confounding variables?

This tests whether the mass dependence of the RAR exponent p(M)
has an independent signature in the rotation curve SHAPE,
not just in the RAR fit quality.

Confounders controlled:
  - Radial coverage (R_last / R_peak)
  - Gas fraction
  - Data quality (mean eV/V)

Usage:
    python run_slope_test.py
"""
import numpy as np
from sdhg import load_sparc, A0, G, MSUN, KPC


def main():
    print("=" * 70)
    print("Rotation Curve Slope Test")
    print("=" * 70)

    galaxies = load_sparc()

    data = []
    for gid, pts in galaxies.items():
        if len(pts) < 8:
            continue

        R_arr = np.array([p[0] for p in pts])
        V_arr = np.array([p[1] for p in pts])
        Vd_arr = np.array([p[3] for p in pts])
        Vg_arr = np.array([p[4] for p in pts])
        eV_arr = np.array([p[2] for p in pts])

        V_last = V_arr[-1] * 1e3
        R_last = R_arr[-1] * KPC
        M_est = 0.5 * V_last ** 2 * R_last / G / MSUN

        # Outer slope: d(lnV)/d(lnR) in outer 30%
        n_out = max(3, len(R_arr) // 3)
        R_out = R_arr[-n_out:]
        V_out = V_arr[-n_out:]
        if R_out.min() <= 0 or V_out.min() <= 0:
            continue
        if np.log10(R_out).std() < 0.05:
            continue
        slope = np.polyfit(np.log10(R_out), np.log10(V_out), 1)[0]

        # Confounders
        V_peak_idx = np.argmax(V_arr)
        R_d_est = R_arr[max(V_peak_idx, 1)]
        coverage = R_arr[-1] / max(R_d_est, 0.1)
        Vd_last = abs(Vd_arr[-1])
        Vg_last = abs(Vg_arr[-1])
        f_gas = Vg_last ** 2 / max(Vd_last ** 2 + Vg_last ** 2, 1)
        quality = np.mean(eV_arr / np.maximum(V_arr, 1))

        data.append((gid, M_est, slope, coverage, f_gas, quality))

    M = np.array([d[1] for d in data])
    logM = np.log10(M)
    S = np.array([d[2] for d in data])
    C = np.array([d[3] for d in data])
    F = np.array([d[4] for d in data])
    Q = np.array([d[5] for d in data])

    print(f"\nGalaxies: {len(data)}")

    # Raw correlation
    r0 = np.corrcoef(logM, S)[0, 1]
    print(f"\nRaw correlation (slope vs logM): r = {r0:+.3f}")

    # Multiple regression
    X = np.column_stack([logM, np.log10(np.maximum(C, 0.1)), F, Q, np.ones(len(data))])
    coeffs = np.linalg.lstsq(X, S, rcond=None)[0]
    S_pred = X @ coeffs
    S_res_var = np.var(S - S_pred)
    X_var = np.linalg.inv(X.T @ X)
    se = np.sqrt(S_res_var * X_var[0, 0])
    t_val = coeffs[0] / max(se, 1e-10)

    print(f"\nMultiple regression (controlling for coverage, f_gas, quality):")
    print(f"  logM coefficient: {coeffs[0]:+.4f}")
    print(f"  t-value: {t_val:.2f}")
    print(f"  Significant at p<0.01: {'YES' if abs(t_val) > 2.6 else 'NO'}")

    # Partial correlations
    tests = []
    for name, Z in [("coverage", np.log10(np.maximum(C, 0.1))),
                     ("f_gas", F),
                     ("quality", Q)]:
        r_SM = np.corrcoef(logM, S)[0, 1]
        r_SZ = np.corrcoef(S, Z)[0, 1]
        r_MZ = np.corrcoef(logM, Z)[0, 1]
        r_partial = (r_SM - r_SZ * r_MZ) / np.sqrt(max((1 - r_SZ ** 2) * (1 - r_MZ ** 2), 1e-10))
        tests.append((name, r_partial))
        print(f"  Partial r (controlling {name}): {r_partial:+.3f}")

    # High-quality subset
    good = Q < np.median(Q)
    r_good = np.corrcoef(logM[good], S[good])[0, 1]
    print(f"  High-quality only (N={good.sum()}): r = {r_good:+.3f}")

    # Verdict
    survived = sum(abs(r) > 0.3 for _, r in tests) + (abs(t_val) > 2) + (abs(r_good) > 0.3)
    print(f"\n  Tests survived: {survived}/5")

    print(f"\n{'=' * 70}")
    if survived >= 4:
        print("  The slope-mass correlation survives all confounder controls.")
        print("  This is consistent with mass-dependent effective dimension,")
        print("  but does NOT prove it — other physical mechanisms")
        print("  (e.g., mass-dependent halo profiles in LCDM) could also explain it.")
    else:
        print("  The correlation is partially explained by confounders.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
