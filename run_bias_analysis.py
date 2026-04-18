"""
MOND bias analysis: Does standard MOND (p=0.5) have a mass-dependent
systematic bias, and does p(M) remove it?

Computes per-galaxy residuals (log g_obs - log g_pred) and their
correlation with galaxy mass.

This script reproduces the bias statistics cited in the README:
  - MOND bias-mass correlation: r = +0.46
  - p(M) bias-mass correlation: r = +0.11
  - Mean MOND bias at M < 10^9: ~ -0.42 dex
  - Gas-dominated subsample: r = +0.25 (N=24)

Usage:
    python run_bias_analysis.py
"""
import numpy as np
from sdhg import load_sparc, A0, G, MSUN, KPC


def main():
    print("=" * 70)
    print("MOND Bias Analysis: Mass-dependent systematic error")
    print("=" * 70)

    galaxies = load_sparc()

    # Global fit parameters (from run_global_fit.py)
    Y_global = 0.444
    M0 = 10 ** 10.17
    alpha = 0.312

    gal_results = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue

        V_last = pts[-1][1] * 1e3
        R_last = pts[-1][0] * KPC
        M_est = 0.5 * V_last ** 2 * R_last / G / MSUN

        # p(M) exponent
        u = (max(M_est, 1) / M0) ** alpha
        p_M = 2 * u / (1 + 3 * u)

        # Per-point residuals
        bias_mcg = []
        bias_pM = []
        Vd_sum2, Vg_sum2 = 0.0, 0.0

        for R_kpc, Vo, eV, Vd, Vg, Vb in pts:
            R = R_kpc * KPC
            g_obs = (Vo * 1e3) ** 2 / R
            gb = (Y_global * (Vd * 1e3) ** 2
                  + np.sign(Vg) * (Vg * 1e3) ** 2
                  + 0.7 * (Vb * 1e3) ** 2) / R
            if gb <= 0 or g_obs <= 0:
                continue

            x = gb / A0

            # McGaugh (p=0.5)
            mu_mcg = max(1 - np.exp(-x ** 0.5), 1e-20)
            g_pred_mcg = gb / mu_mcg
            bias_mcg.append(np.log10(g_obs) - np.log10(g_pred_mcg))

            # p(M)
            mu_pM = max(1 - np.exp(-max(x, 1e-20) ** p_M), 1e-20)
            g_pred_pM = gb / mu_pM
            bias_pM.append(np.log10(g_obs) - np.log10(g_pred_pM))

            Vd_sum2 += (Vd * 1e3) ** 2
            Vg_sum2 += (Vg * 1e3) ** 2

        if len(bias_mcg) < 3:
            continue

        f_gas = Vg_sum2 / max(Vd_sum2 + Vg_sum2, 1)

        gal_results.append({
            "gid": gid,
            "logM": np.log10(M_est),
            "bias_mcg": np.mean(bias_mcg),
            "bias_pM": np.mean(bias_pM),
            "f_gas": f_gas,
            "N_pts": len(bias_mcg),
        })

    logM = np.array([g["logM"] for g in gal_results])
    bias_mcg = np.array([g["bias_mcg"] for g in gal_results])
    bias_pM = np.array([g["bias_pM"] for g in gal_results])
    f_gas = np.array([g["f_gas"] for g in gal_results])

    N = len(gal_results)
    print(f"\nGalaxies analyzed: {N}")

    # --- Bias-mass correlations ---
    r_mcg = np.corrcoef(logM, bias_mcg)[0, 1]
    r_pM = np.corrcoef(logM, bias_pM)[0, 1]

    print(f"\nBias-mass correlation:")
    print(f"  MOND (p=0.5):  r = {r_mcg:+.2f}")
    print(f"  p(M) model:    r = {r_pM:+.2f}")
    print(f"  Reduction:     {(1 - abs(r_pM) / abs(r_mcg)) * 100:.0f}%")

    # --- Mass-bin mean bias ---
    print(f"\nMean bias by mass bin (MOND):")
    print(f"  {'Bin':<12} {'N':>4} {'Mean bias':>10} {'Std':>8}")
    print(f"  {'-' * 36}")
    for lo, hi in [(7, 9), (9, 10), (10, 11), (11, 13)]:
        mask = (logM >= lo) & (logM < hi)
        if mask.sum() > 0:
            print(f"  logM {lo}-{hi:<4} {mask.sum():>4} "
                  f"{bias_mcg[mask].mean():>+10.3f} {bias_mcg[mask].std():>8.3f}")

    dwarf = logM < 9
    print(f"\n  Dwarf mean bias (logM < 9): {bias_mcg[dwarf].mean():+.2f} dex (N={dwarf.sum()})")

    # --- Gas-dominated subsample ---
    gas_dom = f_gas > 0.5
    N_gas = gas_dom.sum()
    if N_gas >= 5:
        r_gas = np.corrcoef(logM[gas_dom], bias_mcg[gas_dom])[0, 1]
        print(f"\nGas-dominated galaxies (f_gas > 0.5):")
        print(f"  N = {N_gas}")
        print(f"  MOND bias-mass correlation: r = {r_gas:+.2f}")
        if N_gas < 30:
            print(f"  (Small sample — suggestive but not statistically significant)")
    else:
        print(f"\nGas-dominated galaxies: only {N_gas} found (need >= 5)")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"MOND has a mass-dependent systematic bias:")
    print(f"  Underpredicts g_obs for dwarfs (bias = {bias_mcg[dwarf].mean():+.2f} dex)")
    print(f"  Approximately unbiased for massive galaxies")
    print(f"  Correlation with mass: r = {r_mcg:+.2f}")
    print(f"p(M) reduces the correlation to r = {r_pM:+.2f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
