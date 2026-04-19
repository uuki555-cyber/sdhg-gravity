"""
Bayesian model comparison: BIC, AIC, and chi-squared for all models.

Compares McGaugh, constant-p, variable-G, and p(M) models
with proper information criteria that penalize extra parameters.

Usage:
    python run_model_comparison.py
"""
import numpy as np
from scipy.optimize import minimize
from sdhg import load_sparc, load_clusters, A0, G, MSUN, KPC


def main():
    print("=" * 70)
    print("Bayesian Model Comparison")
    print("=" * 70)

    galaxies = load_sparc()
    clusters = load_clusters()

    gal_data = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_last = pts[-1][1] * 1e3
        R_last = pts[-1][0] * KPC
        M_est = 0.5 * V_last ** 2 * R_last / G / MSUN
        for R_kpc, Vo, eV, Vd, Vg, Vb in pts:
            gal_data.append((Vd * 1e3, Vg * 1e3, Vb * 1e3,
                             R_kpc * KPC, Vo * 1e3, M_est))

    N = len(gal_data) + len(clusters) * 50  # effective data points

    def score(Y, p_fn):
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo, M in gal_data:
            go = Vo ** 2 / R
            gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
            if gb > 0 and go > 0:
                p = p_fn(M)
                x = gb / A0
                mu = max(1 - np.exp(-max(x, 1e-20) ** min(p, 0.99)), 1e-20)
                s += (np.log10(go) - np.log10(gb / mu)) ** 2
                n += 1
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            p = p_fn(cl["M500_sun"])
            x = gb / A0
            mu = max(1 - np.exp(-max(x, 1e-20) ** min(p, 0.99)), 1e-20)
            mu_need = cl["M_bar_sun"] / cl["M500_sun"]
            s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
            n += 50
        return s, n

    def score_varG(params):
        Y, logM0, beta = params
        M0 = 10 ** logM0
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo, M in gal_data:
            go = Vo ** 2 / R
            gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
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
        return s, n

    # Fit all models
    # A: McGaugh (1 param)
    rA = minimize(lambda p: score(p[0], lambda M: 0.5)[0],
                  [0.5], bounds=[(0.1, 3)], method='L-BFGS-B')
    chi2_A, n_A = score(rA.x[0], lambda M: 0.5)
    k_A = 1

    # B: Constant p (2 params)
    rB = minimize(lambda p: score(p[0], lambda M, pp=p: pp[1])[0],
                  [0.5, 0.5], bounds=[(0.1, 3), (0.1, 0.9)], method='L-BFGS-B')
    chi2_B, n_B = score(rB.x[0], lambda M: rB.x[1])
    k_B = 2

    # C: Variable G(M) (3 params)
    rC = minimize(lambda p: score_varG(p)[0],
                  [0.5, 10.5, 0.05], bounds=[(0.1, 3), (8, 13), (-0.3, 0.3)],
                  method='L-BFGS-B')
    chi2_C, n_C = score_varG(rC.x)
    k_C = 3

    # D: p(M) free alpha (3 params)
    rD = minimize(lambda p: score(p[0], lambda M, M0=10**p[1], a=p[2]:
                  min(max(2*(max(M,1)/M0)**a / (1+3*(max(M,1)/M0)**a), 0.01), 0.99))[0],
                  [0.5, 10.5, 0.33], bounds=[(0.1, 3), (8, 13), (0.1, 0.6)],
                  method='L-BFGS-B')
    chi2_D, n_D = score(rD.x[0], lambda M, M0=10**rD.x[1], a=rD.x[2]:
                  min(max(2*(max(M,1)/M0)**a / (1+3*(max(M,1)/M0)**a), 0.01), 0.99))
    k_D = 3

    # E: p(M) alpha=1/3 fixed (2 params)
    rE = minimize(lambda p: score(p[0], lambda M, M0=10**p[1]:
                  min(max(2*(max(M,1)/M0)**(1/3) / (1+3*(max(M,1)/M0)**(1/3)), 0.01), 0.99))[0],
                  [0.5, 10.5], bounds=[(0.1, 3), (8, 13)], method='L-BFGS-B')
    chi2_E, n_E = score(rE.x[0], lambda M, M0=10**rE.x[1]:
                  min(max(2*(max(M,1)/M0)**(1/3) / (1+3*(max(M,1)/M0)**(1/3)), 0.01), 0.99))
    k_E = 2

    # Information criteria
    models = [
        ("McGaugh (p=0.5)", k_A, chi2_A, n_A),
        ("Constant p", k_B, chi2_B, n_B),
        ("Variable G(M)", k_C, chi2_C, n_C),
        ("p(M) alpha=free", k_D, chi2_D, n_D),
        ("p(M) alpha=1/3", k_E, chi2_E, n_E),
    ]

    print(f"\nN_eff = {N} data points")
    print(f"\n{'Model':<20} {'k':>3} {'chi2':>10} {'RMS':>8} {'AIC':>10} {'BIC':>10} {'dAIC':>7} {'dBIC':>7}")
    print("-" * 78)

    aics = []
    bics = []
    for name, k, chi2, n in models:
        rms = np.sqrt(chi2 / n)
        aic = chi2 + 2 * k
        bic = chi2 + k * np.log(N)
        aics.append(aic)
        bics.append(bic)

    aic_min = min(aics)
    bic_min = min(bics)

    for i, (name, k, chi2, n) in enumerate(models):
        rms = np.sqrt(chi2 / n)
        print(f"{name:<20} {k:>3} {chi2:>10.2f} {rms:>8.5f} "
              f"{aics[i]:>10.2f} {bics[i]:>10.2f} "
              f"{aics[i]-aic_min:>+7.1f} {bics[i]-bic_min:>+7.1f}")

    print(f"\n{'=' * 70}")
    best_aic = [m[0] for m in models][np.argmin(aics)]
    best_bic = [m[0] for m in models][np.argmin(bics)]
    print(f"Best by AIC: {best_aic}")
    print(f"Best by BIC: {best_bic}")

    # Bayes factor approximation
    dBIC_mcg_pM = bics[0] - bics[4]  # McGaugh vs p(M) alpha=1/3
    print(f"\nBayes factor (McGaugh vs p(M) alpha=1/3):")
    print(f"  dBIC = {dBIC_mcg_pM:.1f}")
    if dBIC_mcg_pM > 10:
        print(f"  Very strong evidence for p(M)")
    elif dBIC_mcg_pM > 6:
        print(f"  Strong evidence for p(M)")
    elif dBIC_mcg_pM > 2:
        print(f"  Positive evidence for p(M)")
    else:
        print(f"  Inconclusive")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
