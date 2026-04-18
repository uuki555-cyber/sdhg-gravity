"""
Main analysis: Fit SDHG p(M) to SPARC galaxies + galaxy clusters.
Compares with McGaugh+ 2016 (constant p = 0.5).

NOTE: This script fits Y_disk per galaxy, which is subject to the
Li+ (2021) bias concern. For the bias-free result, see run_global_fit.py.
The M0 and alpha values here may differ from the global fit.

Usage:
    python run_main_analysis.py
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sdhg import (
    load_sparc, load_clusters, mu_sdhg, mu_mcgaugh,
    g_bar_from_components, A0, G, MSUN, KPC,
)

MPC = KPC * 1e3


def fit_galaxy(pts, mu_func, Y_range=(0.1, 3.0)):
    """Fit Y_disk for a single galaxy and return total chi2, N."""
    def obj(Y):
        s, n = 0.0, 0
        for R_kpc, Vobs, eV, Vdisk, Vgas, Vbul in pts:
            R = R_kpc * KPC
            g_obs = (Vobs * 1e3) ** 2 / R
            g_bar = g_bar_from_components(Vdisk * 1e3, Vgas * 1e3, Vbul * 1e3, R, Y)
            if g_bar > 0 and g_obs > 0:
                x = g_bar / A0
                mu = mu_func(x)
                g_pred = g_bar / mu
                s += (np.log10(g_obs) - np.log10(g_pred)) ** 2
                n += 1
        return s / max(n, 1)

    res = minimize_scalar(obj, bounds=Y_range, method="bounded")
    Y_best = res.x
    total_s, total_n = 0.0, 0
    for R_kpc, Vobs, eV, Vdisk, Vgas, Vbul in pts:
        R = R_kpc * KPC
        g_obs = (Vobs * 1e3) ** 2 / R
        g_bar = g_bar_from_components(Vdisk * 1e3, Vgas * 1e3, Vbul * 1e3, R, Y_best)
        if g_bar > 0 and g_obs > 0:
            x = g_bar / A0
            mu = mu_func(x)
            total_s += (np.log10(g_obs) - np.log10(g_bar / mu)) ** 2
            total_n += 1
    return Y_best, total_s, total_n


def main():
    print("=" * 70)
    print("SDHG: Scale-Dependent Holographic Gravity — Main Analysis")
    print("=" * 70)

    galaxies = load_sparc()
    clusters = load_clusters()
    print(f"\nLoaded {len(galaxies)} SPARC galaxies, {len(clusters)} clusters")

    # --- Fit SDHG: p(M) with free Y_disk per galaxy ---
    def unified_rms(params):
        log_M0, alpha_exp = params
        M0 = 10 ** log_M0
        total_s, total_n = 0.0, 0

        for gid, pts in galaxies.items():
            if len(pts) < 5:
                continue
            V_last = pts[-1][1] * 1e3
            R_last = pts[-1][0] * KPC
            M_est = 0.5 * V_last ** 2 * R_last / G / MSUN
            u = (max(M_est, 1) / M0) ** alpha_exp
            p = 2.0 * u / (1.0 + 3.0 * u)

            def mu_func(x):
                return np.clip(1 - np.exp(-np.maximum(x, 1e-20) ** p), 1e-20, 1)

            _, ss, nn = fit_galaxy(pts, mu_func)
            total_s += ss
            total_n += nn

        # Clusters (weight 50 each to balance with galaxies)
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            x = gb / A0
            M_dyn = cl["M500_sun"]
            u = (M_dyn / M0) ** alpha_exp
            p = 2.0 * u / (1.0 + 3.0 * u)
            mu_pred = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
            mu_need = cl["M_bar_sun"] / M_dyn
            total_s += 50 * (np.log10(mu_need) - np.log10(mu_pred)) ** 2
            total_n += 50

        return np.sqrt(total_s / total_n)

    # Optimize
    print("\nFitting SDHG p(M) to galaxies + clusters...")
    res = minimize(
        unified_rms, [10.5, 0.33],
        bounds=[(8, 13), (0.1, 0.6)], method="L-BFGS-B",
    )
    log_M0, alpha = res.x
    rms_sdhg = res.fun
    print(f"  M0 = 10^{log_M0:.3f} = {10**log_M0:.3e} Msun")
    print(f"  alpha = {alpha:.4f}  (1/3 = {1/3:.4f})")
    print(f"  Unified RMS = {rms_sdhg:.5f}")

    # --- McGaugh baseline ---
    print("\nFitting McGaugh (constant p=0.5)...")
    total_mcg_s, total_mcg_n = 0.0, 0
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        _, ss, nn = fit_galaxy(pts, mu_mcgaugh)
        total_mcg_s += ss
        total_mcg_n += nn
    for cl in clusters:
        gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
        x = gb / A0
        mu_pred = mu_mcgaugh(x)
        mu_need = cl["M_bar_sun"] / cl["M500_sun"]
        total_mcg_s += 50 * (np.log10(mu_need) - np.log10(mu_pred)) ** 2
        total_mcg_n += 50
    rms_mcg = np.sqrt(total_mcg_s / total_mcg_n)
    print(f"  McGaugh Unified RMS = {rms_mcg:.5f}")

    improvement = (rms_mcg - rms_sdhg) / rms_mcg * 100
    print(f"\n{'=' * 70}")
    print(f"RESULT: SDHG improves over McGaugh by {improvement:+.1f}%")
    print(f"{'=' * 70}")

    # --- Mass-bin analysis ---
    print("\nMass-bin breakdown:")
    print(f"  {'log(M)':>8} {'N':>4} {'RMS_SDHG':>10} {'RMS_McG':>10} {'Improvement':>12}")
    print(f"  {'-' * 48}")

    M0_best = 10 ** log_M0
    for lo, hi in [(7, 8.5), (8.5, 9.5), (9.5, 10.3), (10.3, 11), (11, 12.5)]:
        ss_s, ss_m, nn = 0, 0, 0
        for gid, pts in galaxies.items():
            if len(pts) < 5:
                continue
            V_last = pts[-1][1] * 1e3
            R_last = pts[-1][0] * KPC
            M_est = 0.5 * V_last ** 2 * R_last / G / MSUN
            if not (lo <= np.log10(M_est) < hi):
                continue
            u = (M_est / M0_best) ** alpha
            p = 2.0 * u / (1.0 + 3.0 * u)
            mu_s = lambda x, pp=p: np.clip(1 - np.exp(-np.maximum(x, 1e-20) ** pp), 1e-20, 1)
            _, s1, n1 = fit_galaxy(pts, mu_s)
            _, s2, n2 = fit_galaxy(pts, mu_mcgaugh)
            ss_s += s1; ss_m += s2; nn += n1
        if nn > 20:
            rs = np.sqrt(ss_s / nn)
            rm = np.sqrt(ss_m / nn)
            imp = (rm - rs) / rm * 100
            print(f"  {lo}-{hi:>4} {nn:>4} {rs:>10.5f} {rm:>10.5f} {imp:>+11.1f}%")

    # --- Cluster results ---
    print("\nCluster results (SDHG):")
    print(f"  {'Cluster':>10} {'x':>8} {'mu_need':>8} {'mu_pred':>8} {'ratio':>7}")
    print(f"  {'-' * 44}")
    ratios = []
    for cl in clusters:
        gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
        x = gb / A0
        M_dyn = cl["M500_sun"]
        mu_need = cl["M_bar_sun"] / M_dyn
        mu_pred = mu_sdhg(x, M_dyn, M0_best)
        r = mu_need / mu_pred
        ratios.append(r)
        print(f"  {cl['name']:>10} {x:>8.4f} {mu_need:>8.4f} {mu_pred:>8.4f} {r:>7.3f}")
    print(f"\n  Mean ratio: {np.mean(ratios):.3f} +/- {np.std(ratios):.3f} (1.0 = perfect)")

    print(f"\n{'=' * 70}")
    print("SDHG formula:")
    print(f"  mu(x, M) = 1 - exp(-x^p(M))")
    print(f"  p(M) = 2u/(1+3u),  u = (M/M0)^(1/3)")
    print(f"  M0 = 10^{log_M0:.2f} Msun,  alpha = {alpha:.3f}")
    print(f"  a0 = {A0:.1e} m/s^2")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
