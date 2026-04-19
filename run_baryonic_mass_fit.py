"""
Global fit using BARYONIC mass (from Vdisk, Vgas) instead of dynamical mass.

This tests whether the p(M) signal persists when using photometric
baryonic masses, which are independent of the gravitational model.

Usage:
    python run_baryonic_mass_fit.py
"""
import numpy as np
from scipy.optimize import minimize
from sdhg import load_sparc, load_clusters, A0, G, MSUN, KPC


def main():
    print("=" * 70)
    print("Global Fit: Baryonic Mass vs Dynamical Mass")
    print("=" * 70)

    galaxies = load_sparc()
    clusters = load_clusters()

    # Build data with BOTH mass definitions
    gal_data_dyn = []   # dynamical mass: M ~ 0.5 V^2 R / G
    gal_data_bar = []   # baryonic mass from velocity components

    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_last = pts[-1][1] * 1e3
        R_last = pts[-1][0] * KPC
        M_dyn = 0.5 * V_last ** 2 * R_last / G / MSUN

        # Baryonic mass: sum disk + gas contributions at last point
        Vd_last = pts[-1][3] * 1e3
        Vg_last = pts[-1][4] * 1e3
        Vb_last = pts[-1][5] * 1e3
        # M_bar ~ (Y*Vd^2 + Vg^2 + 0.7*Vb^2) * R / G
        M_bar = (0.5 * Vd_last**2 + abs(Vg_last)**2 + 0.7 * Vb_last**2) \
                * R_last / G / MSUN

        for R_kpc, Vo, eV, Vd, Vg, Vb in pts:
            gal_data_dyn.append((Vd*1e3, Vg*1e3, Vb*1e3,
                                 R_kpc*KPC, Vo*1e3, M_dyn))
            gal_data_bar.append((Vd*1e3, Vg*1e3, Vb*1e3,
                                 R_kpc*KPC, Vo*1e3, max(M_bar, 1)))

    def score(gal_data, Y, logM0, alpha):
        M0 = 10 ** logM0
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo, M in gal_data:
            go = Vo ** 2 / R
            gb = (Y * Vd**2 + np.sign(Vg) * Vg**2 + 0.7 * Vb**2) / R
            if gb > 0 and go > 0:
                u = (max(M, 1) / M0) ** alpha
                p = min(max(2 * u / (1 + 3 * u), 0.01), 0.99)
                x = gb / A0
                mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
                s += (np.log10(go) - np.log10(gb / mu)) ** 2
                n += 1
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            u = (cl["M500_sun"] / M0) ** alpha
            p = min(max(2 * u / (1 + 3 * u), 0.01), 0.99)
            x = gb / A0
            mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
            mu_need = cl["M_bar_sun"] / cl["M500_sun"]
            s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
            n += 50
        return np.sqrt(s / n)

    def score_mcg(gal_data, Y):
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo, M in gal_data:
            go = Vo ** 2 / R
            gb = (Y * Vd**2 + np.sign(Vg) * Vg**2 + 0.7 * Vb**2) / R
            if gb > 0 and go > 0:
                x = gb / A0
                mu = max(1 - np.exp(-x ** 0.5), 1e-20)
                s += (np.log10(go) - np.log10(gb / mu)) ** 2
                n += 1
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            x = gb / A0
            mu = max(1 - np.exp(-x ** 0.5), 1e-20)
            mu_need = cl["M_bar_sun"] / cl["M500_sun"]
            s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
            n += 50
        return np.sqrt(s / n)

    def score_varG(gal_data, params):
        Y, logM0, beta = params
        M0 = 10 ** logM0
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo, M in gal_data:
            go = Vo ** 2 / R
            gb = (Y * Vd**2 + np.sign(Vg) * Vg**2 + 0.7 * Vb**2) / R
            gb_eff = gb * (max(M, 1) / M0) ** beta
            if gb_eff > 0 and go > 0:
                x = gb_eff / A0
                mu = max(1 - np.exp(-x ** 0.5), 1e-20)
                s += (np.log10(go) - np.log10(gb_eff / mu)) ** 2
                n += 1
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            gb_eff = gb * (cl["M500_sun"] / M0) ** beta
            x = gb_eff / A0
            mu = max(1 - np.exp(-max(x, 1e-20) ** 0.5), 1e-20)
            mu_need = cl["M_bar_sun"] / cl["M500_sun"]
            s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
            n += 50
        return np.sqrt(s / n)

    for mass_label, gdata in [("Dynamical mass", gal_data_dyn),
                               ("Baryonic mass", gal_data_bar)]:
        print(f"\n--- {mass_label} ---")
        rM = minimize(lambda p: score_mcg(gdata, p[0]),
                      [0.5], bounds=[(0.1, 3)], method='L-BFGS-B')
        rP = minimize(lambda p: score(gdata, p[0], p[1], p[2]),
                      [0.5, 10.5, 0.33], bounds=[(0.1, 3), (8, 13), (0.05, 0.6)],
                      method='L-BFGS-B')
        rG = minimize(lambda p: score_varG(gdata, p),
                      [0.5, 10.5, 0.05], bounds=[(0.1, 3), (8, 13), (-0.3, 0.3)],
                      method='L-BFGS-B')

        imp_p = (rM.fun - rP.fun) / rM.fun * 100
        imp_g = (rM.fun - rG.fun) / rM.fun * 100
        gap = imp_p - imp_g

        print(f"  McGaugh:    RMS = {rM.fun:.5f}")
        print(f"  p(M):       RMS = {rP.fun:.5f} ({imp_p:+.1f}%)  "
              f"alpha={rP.x[2]:.3f} M0=10^{rP.x[1]:.2f}")
        print(f"  G(M) ctrl:  RMS = {rG.fun:.5f} ({imp_g:+.1f}%)")
        print(f"  Li+ gap:    {gap:.1f}% (p(M) - G(M))")

    print(f"\n{'=' * 70}")
    print("If the signal is from Y_disk bias, baryonic mass should")
    print("show a SMALLER gap than dynamical mass.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
