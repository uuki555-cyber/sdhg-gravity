"""
p(Σ) model test: is surface density the true driver of the RAR exponent?

p(Σ) = 2u / (1 + 3u), where u = (Σ/Σ0)^alpha_Σ
Σ is the surface mass density M / (πR_last²).

Compared against p(M) under fair conditions (each model optimizes own Y_disk).

Usage:
    python run_psigma_test.py
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sdhg import load_sparc, load_clusters, p_of_M, A0, G, MSUN, KPC


def p_of_Sigma(Sigma, Sigma0, alpha_S):
    """p(Σ) functional form."""
    u = (Sigma / Sigma0)**alpha_S
    return np.clip(2 * u / (1 + 3 * u), 0.01, 0.95)


def compute_fg(pts):
    V_gas_sq = sum(abs(p[4]*1e3)**2 for p in pts)
    V_disk_sq = sum(0.5*(p[3]*1e3)**2 for p in pts)
    V_bul_sq = sum(0.7*(p[5]*1e3)**2 for p in pts)
    total = V_gas_sq + V_disk_sq + V_bul_sq
    return V_gas_sq / total if total > 0 else 0


def global_rms(gal_list, Y_disk, model='mcgaugh', Sigma0=None, alpha_S=None):
    resid = []
    for gid, pts, M, Sigma in gal_list:
        for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
            R = R_kpc * KPC
            g_obs = (Vobs * 1e3)**2 / R
            g_bar = (Y_disk*(Vdisk*1e3)**2 +
                     np.sign(Vgas)*(Vgas*1e3)**2 +
                     0.7*(Vbul*1e3)**2) / R
            if g_bar <= 0 or g_obs <= 0:
                continue
            x = g_bar / A0
            if model == 'mcgaugh':
                p = 0.5
            elif model == 'pM':
                p = p_of_M(M)
            elif model == 'pSigma':
                p = p_of_Sigma(Sigma, Sigma0, alpha_S)
            mu = max(1 - np.exp(-max(x, 1e-20)**p), 1e-20)
            resid.append(np.log10(g_obs) - np.log10(g_bar/mu))
    return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999


def main():
    print("=" * 75)
    print("p(Σ) model test: surface density as the driver of RAR exponent")
    print("=" * 75)

    galaxies = load_sparc()
    all_gals = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_last = pts[-1][1] * 1e3
        R_last_kpc = pts[-1][0]
        R_last = R_last_kpc * KPC
        M = 0.5 * V_last**2 * R_last / G / MSUN
        Sigma = M / (np.pi * R_last_kpc**2)  # Msun/kpc²
        all_gals.append((gid, pts, M, Sigma))

    # Optimize p(Σ) parameters globally
    print("\nOptimizing p(Σ) parameters...")

    def opt_pSigma(params):
        Y, logS0, alpha_S = params
        return global_rms(all_gals, Y, 'pSigma',
                          Sigma0=10**logS0, alpha_S=alpha_S)

    res = minimize(opt_pSigma, [0.5, 7.5, 0.5],
                   bounds=[(0.0, 2.0), (4.0, 12.0), (0.05, 2.0)],
                   method='L-BFGS-B')
    Y_S, logS0, alpha_S = res.x
    rms_pSigma = res.fun

    print(f"  Y_disk = {Y_S:.3f}")
    print(f"  logΣ0  = {logS0:.2f}  (Σ0 = {10**logS0:.2e} Msun/kpc²)")
    print(f"  α_Σ    = {alpha_S:.3f}")
    print(f"  RMS    = {rms_pSigma:.4f}")

    # Compare with p(M) and McGaugh at best Y
    def opt_mcg(Y):
        return global_rms(all_gals, Y, 'mcgaugh')

    def opt_pM(Y):
        return global_rms(all_gals, Y, 'pM')

    res_m = minimize_scalar(opt_mcg, bounds=(0.0, 2.0), method='bounded')
    res_p = minimize_scalar(opt_pM, bounds=(0.0, 2.0), method='bounded')

    print(f"\n  McGaugh: RMS = {res_m.fun:.4f} (Y={res_m.x:.3f})")
    print(f"  p(M):    RMS = {res_p.fun:.4f} (Y={res_p.x:.3f}) "
          f"imp = {(res_m.fun-res_p.fun)/res_m.fun*100:+.1f}%")
    print(f"  p(Σ):    RMS = {rms_pSigma:.4f} (Y={Y_S:.3f}) "
          f"imp = {(res_m.fun-rms_pSigma)/res_m.fun*100:+.1f}%")

    # Subset analysis: p(Σ) across bins
    print(f"\n--- Subset improvement analysis (p(M) vs p(Σ)) ---")
    print(f"{'Subset':>20} {'N':>4} {'p(M) imp%':>10} {'p(Σ) imp%':>10}")
    print("-" * 50)

    # Mass bins
    bins_m = [(7, 9), (9, 10), (10, 11), (11, 13)]
    for lo, hi in bins_m:
        sub = [(g, p, m, s) for g, p, m, s in all_gals
               if lo <= np.log10(m) < hi]
        if len(sub) < 3:
            continue

        y_m, r_m = minimize_scalar(
            lambda Y: global_rms(sub, Y, 'mcgaugh'),
            bounds=(0.0, 2.0), method='bounded').x, \
            minimize_scalar(lambda Y: global_rms(sub, Y, 'mcgaugh'),
                            bounds=(0.0, 2.0), method='bounded').fun
        y_p, r_p = minimize_scalar(
            lambda Y: global_rms(sub, Y, 'pM'),
            bounds=(0.0, 2.0), method='bounded').x, \
            minimize_scalar(lambda Y: global_rms(sub, Y, 'pM'),
                            bounds=(0.0, 2.0), method='bounded').fun
        y_s, r_s = minimize_scalar(
            lambda Y: global_rms(sub, Y, 'pSigma',
                                 Sigma0=10**logS0, alpha_S=alpha_S),
            bounds=(0.0, 2.0), method='bounded').x, \
            minimize_scalar(lambda Y: global_rms(sub, Y, 'pSigma',
                                                  Sigma0=10**logS0, alpha_S=alpha_S),
                            bounds=(0.0, 2.0), method='bounded').fun

        imp_m = (r_m - r_p)/r_m*100
        imp_s = (r_m - r_s)/r_m*100
        label = f"logM {lo}-{hi}"
        print(f"{label:>20} {len(sub):>4} {imp_m:>+9.1f}% {imp_s:>+9.1f}%")

    # Gas fraction bins
    bins_fg = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    for lo, hi in bins_fg:
        sub = [(g, p, m, s) for g, p, m, s in all_gals
               if lo <= compute_fg(p) < hi]
        if len(sub) < 3:
            continue

        r_m = minimize_scalar(lambda Y: global_rms(sub, Y, 'mcgaugh'),
                              bounds=(0.0, 2.0), method='bounded').fun
        r_p = minimize_scalar(lambda Y: global_rms(sub, Y, 'pM'),
                              bounds=(0.0, 2.0), method='bounded').fun
        r_s = minimize_scalar(lambda Y: global_rms(sub, Y, 'pSigma',
                                                    Sigma0=10**logS0, alpha_S=alpha_S),
                              bounds=(0.0, 2.0), method='bounded').fun

        imp_m = (r_m - r_p)/r_m*100
        imp_s = (r_m - r_s)/r_m*100
        label = f"f_gas {lo:.1f}-{hi:.1f}"
        print(f"{label:>20} {len(sub):>4} {imp_m:>+9.1f}% {imp_s:>+9.1f}%")

    # Check physical scale of Σ0
    print(f"\n--- Physical interpretation ---")
    Sigma0_SI = 10**logS0 * MSUN / KPC**2  # kg/m²
    print(f"  Σ0 = {10**logS0:.2e} Msun/kpc² = {Sigma0_SI:.2e} kg/m²")
    print(f"  G×Σ0 = {G * Sigma0_SI:.2e} m/s²")
    print(f"  a0   = {A0:.2e} m/s²")
    print(f"  G×Σ0 / a0 = {G * Sigma0_SI / A0:.3f}")
    print(f"  → G×Σ0 is at the order of a0 (MOND scale)")


if __name__ == "__main__":
    main()
