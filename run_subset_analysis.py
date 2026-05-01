"""
Subset analysis of p(M) improvement.

Fair comparison: each model gets its own best Y_disk.
Breaks down the +15.5% global improvement by:
  - Mass bin
  - Gas fraction bin
  - Combined bins

Critical finding: the global +15.5% is concentrated in specific subsets,
not a universal effect. This analysis exposes the distribution of improvement.

Usage:
    python run_subset_analysis.py
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sdhg import load_sparc, p_of_M, A0, G, MSUN, KPC


def compute_fg(pts):
    V_gas_sq = sum(abs(p[4]*1e3)**2 for p in pts)
    V_disk_sq = sum(0.5*(p[3]*1e3)**2 for p in pts)
    V_bul_sq = sum(0.7*(p[5]*1e3)**2 for p in pts)
    total = V_gas_sq + V_disk_sq + V_bul_sq
    return V_gas_sq / total if total > 0 else 0


def global_rms(gal_list, Y_disk, model='mcgaugh'):
    resid = []
    for gid, pts, M in gal_list:
        for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
            R = R_kpc * KPC
            g_obs = (Vobs * 1e3)**2 / R
            g_bar = (Y_disk*(Vdisk*1e3)**2 +
                     np.sign(Vgas)*(Vgas*1e3)**2 +
                     0.7*(Vbul*1e3)**2) / R
            if g_bar <= 0 or g_obs <= 0:
                continue
            x = g_bar / A0
            p = p_of_M(M) if model == 'pM' else 0.5
            mu = max(1 - np.exp(-max(x, 1e-20)**p), 1e-20)
            resid.append(np.log10(g_obs) - np.log10(g_bar/mu))
    return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999


def best_y(gal_list, model):
    res = minimize_scalar(
        lambda Y: global_rms(gal_list, Y, model),
        bounds=(0.0, 2.0), method='bounded')
    return res.x, res.fun


def main():
    print("=" * 75)
    print("Subset Analysis: Where does the +15.5% improvement come from?")
    print("  Fair comparison: each model optimizes its own Y_disk")
    print("=" * 75)

    galaxies = load_sparc()
    all_gals = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_last = pts[-1][1] * 1e3
        R_last = pts[-1][0] * KPC
        M = 0.5 * V_last**2 * R_last / G / MSUN
        all_gals.append((gid, pts, M))

    # Baseline: all galaxies
    y_m, r_m = best_y(all_gals, 'mcgaugh')
    y_p, r_p = best_y(all_gals, 'pM')
    imp_all = (r_m - r_p)/r_m*100
    print(f"\n  ALL GALAXIES (N={len(all_gals)}): "
          f"McG RMS={r_m:.4f}, p(M) RMS={r_p:.4f}, improvement={imp_all:+.1f}%")

    # Mass bins
    print(f"\n--- Breakdown by mass bin ---")
    print(f"{'logM range':>15} {'N':>4} {'Y_McG':>7} {'Y_pM':>6} "
          f"{'RMS_McG':>8} {'RMS_pM':>8} {'imp%':>7}")
    print("-" * 60)
    bins_m = [(7, 8.5), (8.5, 9.5), (9.5, 10.0), (10.0, 10.5),
              (10.5, 11.0), (11.0, 13.0)]
    for lo, hi in bins_m:
        sub = [(g, p, m) for g, p, m in all_gals if lo <= np.log10(m) < hi]
        if len(sub) < 3:
            continue
        y_m, r_m = best_y(sub, 'mcgaugh')
        y_p, r_p = best_y(sub, 'pM')
        imp = (r_m - r_p)/r_m*100
        print(f"  {lo:>4.1f}-{hi:>4.1f}      {len(sub):>4} {y_m:>7.3f} "
              f"{y_p:>6.3f} {r_m:>8.4f} {r_p:>8.4f} {imp:>+6.1f}%")

    # Gas fraction bins
    print(f"\n--- Breakdown by gas fraction bin ---")
    print(f"{'f_gas range':>15} {'N':>4} {'Y_McG':>7} {'Y_pM':>6} "
          f"{'RMS_McG':>8} {'RMS_pM':>8} {'imp%':>7}")
    print("-" * 60)
    bins_fg = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    for lo, hi in bins_fg:
        sub = [(g, p, m) for g, p, m in all_gals if lo <= compute_fg(p) < hi]
        if len(sub) < 3:
            continue
        y_m, r_m = best_y(sub, 'mcgaugh')
        y_p, r_p = best_y(sub, 'pM')
        imp = (r_m - r_p)/r_m*100
        print(f"  {lo:>4.2f}-{hi:>4.2f}      {len(sub):>4} {y_m:>7.3f} "
              f"{y_p:>6.3f} {r_m:>8.4f} {r_p:>8.4f} {imp:>+6.1f}%")

    # Mass x Gas fraction 2D
    print(f"\n--- 2D breakdown (mass × gas fraction) ---")
    print(f"{'logM':>6} × {'f_gas':<10} {'N':>4} {'imp%':>7}")
    print("-" * 35)
    for lo_m, hi_m in [(7, 9), (9, 10), (10, 11), (11, 13)]:
        for lo_g, hi_g in [(0.0, 0.3), (0.3, 0.5), (0.5, 1.0)]:
            sub = [(g, p, m) for g, p, m in all_gals
                   if lo_m <= np.log10(m) < hi_m and
                      lo_g <= compute_fg(p) < hi_g]
            if len(sub) < 3:
                continue
            y_m, r_m = best_y(sub, 'mcgaugh')
            y_p, r_p = best_y(sub, 'pM')
            imp = (r_m - r_p)/r_m*100
            label_m = f"{lo_m}-{hi_m}"
            label_g = f"{lo_g:.1f}-{hi_g:.1f}"
            print(f"  {label_m:>5} × {label_g:<10} {len(sub):>4} {imp:>+6.1f}%")

    # Summary
    print(f"\n{'='*75}")
    print("HONEST ASSESSMENT:")
    print(f"{'='*75}")
    print("""
  The global +15.5% improvement is NOT a universal effect.
  With fair comparison (each model optimizes Y_disk), improvement is:
  - Concentrated in low-mass galaxies (logM < 9)
  - Concentrated in mixed gas-fraction galaxies (f_gas 0.3-0.5)
  - Approximately zero for disk-dominated massive galaxies
  - Potentially NEGATIVE for highest-mass galaxies

  Interpretation: p(M) primarily fixes the low-mass end of RAR
  where McGaugh systematically underpredicts g_obs. It does NOT
  improve the fit for most SPARC galaxies (which are intermediate
  mass, disk-dominated).
  """)


if __name__ == "__main__":
    main()
