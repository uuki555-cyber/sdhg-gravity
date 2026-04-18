"""
Global fit: Test p(M) without any per-galaxy fitting.

This script addresses the Li, Lelli & McGaugh (2021, A&A) concern that
fitting a parameter per galaxy can create spurious mass dependence.

Here, Y_disk is a SINGLE global value shared by ALL galaxies.
No per-galaxy fitting. No flat priors. No room for Li+ bias.

We compare four models with the same data:
  C: McGaugh (p=0.5, Y global)          — 1 parameter
  D: Constant p (Y, p global)           — 2 parameters
  B: Variable G(M) (Y, M0, beta)        — 3 parameters (control)
  A: SDHG p(M) (Y, M0, alpha)           — 3 parameters

If A >> B: p(M) is a real signal, not an artifact of G variation.

Usage:
    python run_global_fit.py
"""
import numpy as np
from scipy.optimize import minimize
from sdhg import load_sparc, load_clusters, load_little_things, A0, G, MSUN, KPC


def main():
    print("=" * 70)
    print("Global Fit: p(M) vs G(M) — No per-galaxy fitting")
    print("=" * 70)

    galaxies = load_sparc()
    clusters = load_clusters()

    # Flatten all galaxy data points
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

    print(f"Data: {len(gal_data)} galaxy points + {len(clusters)} clusters")

    # --- Scoring function ---
    def score_pM(Y, p_fn):
        """RMS for a model defined by a p(M) function."""
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo, M in gal_data:
            go = Vo ** 2 / R
            gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
            if gb > 0 and go > 0:
                p = p_fn(M)
                x = gb / A0
                mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
                s += (np.log10(go) - np.log10(gb / mu)) ** 2
                n += 1
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            p = p_fn(cl["M500_sun"])
            x = gb / A0
            mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
            mu_need = cl["M_bar_sun"] / cl["M500_sun"]
            s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
            n += 50
        return np.sqrt(s / n)

    def score_varG(params):
        """RMS for variable-G model: g_bar_eff = g_bar * (M/M0)^beta."""
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
        return np.sqrt(s / n)

    # --- Fit models ---
    # C: McGaugh
    rC = minimize(lambda p: score_pM(p[0], lambda M: 0.5),
                  [0.5], bounds=[(0.1, 3)], method="L-BFGS-B")

    # D: Constant p
    rD = minimize(lambda p: score_pM(p[0], lambda M, pp=p: pp[1]),
                  [0.5, 0.5], bounds=[(0.1, 3), (0.1, 0.9)], method="L-BFGS-B")

    # A: SDHG p(M)
    def sdhg_obj(params):
        Y, logM0, alpha = params
        M0 = 10 ** logM0
        return score_pM(Y, lambda M: 2 * (max(M, 1) / M0) ** alpha /
                        (1 + 3 * (max(M, 1) / M0) ** alpha))

    rA = minimize(sdhg_obj, [0.5, 10.5, 0.33],
                  bounds=[(0.1, 3), (8, 13), (0.1, 0.6)], method="L-BFGS-B")

    # B: Variable G(M) — same number of params as A
    rB = minimize(score_varG, [0.5, 10.5, 0.05],
                  bounds=[(0.1, 3), (8, 13), (-0.3, 0.3)], method="L-BFGS-B")

    # --- Results ---
    iA = (rC.fun - rA.fun) / rC.fun * 100
    iB = (rC.fun - rB.fun) / rC.fun * 100
    iD = (rC.fun - rD.fun) / rC.fun * 100

    print(f"\n{'Model':<35} {'Params':>6} {'RMS':>10} {'vs McGaugh':>11}")
    print("-" * 65)
    print(f"{'C: McGaugh (p=0.5)':<35} {'1':>6} {rC.fun:>10.5f} {'baseline':>11}")
    print(f"{'D: Constant p':<35} {'2':>6} {rD.fun:>10.5f} {iD:>+10.1f}%")
    print(f"{'B: Variable G(M) [control]':<35} {'3':>6} {rB.fun:>10.5f} {iB:>+10.1f}%")
    print(f"{'A: SDHG p(M)':<35} {'3':>6} {rA.fun:>10.5f} {iA:>+10.1f}%")

    print(f"\n  SDHG params: Y={rA.x[0]:.3f}, M0=10^{rA.x[1]:.2f}, alpha={rA.x[2]:.3f}")
    print(f"  VarG params: Y={rB.x[0]:.3f}, M0=10^{rB.x[1]:.2f}, beta={rB.x[2]:.4f}")

    print(f"\n{'=' * 70}")
    gap = iA - iB
    if gap > 3:
        print(f"  RESULT: p(M) beats G(M) by {gap:.1f}%.")
        print(f"  Li+ (2021) Y_disk degeneracy is disfavored as sole explanation.")
        print(f"  Definitive exclusion requires more gas-dominated dwarf data.")
    elif gap > 1:
        print(f"  RESULT: p(M) exceeds G(M) by {gap:.1f}%. Suggestive but not definitive.")
    else:
        print(f"  RESULT: p(M) and G(M) are similar ({gap:.1f}%). Inconclusive.")
    print(f"{'=' * 70}")

    # --- Cross-validation ---
    print(f"\nCross-validation: SPARC-trained -> LITTLE THINGS")
    lt = load_little_things()
    oh = {"cvidwa": 2.6e7, "ddo47": 3.5e8, "ddo50": 6e8, "ddo52": 3.3e8,
          "ddo53": 6.5e7, "ddo87": 4e8, "ddo101": 1.5e8, "ddo126": 2e8,
          "ddo133": 2.5e8, "ddo154": 5e8, "ddo168": 4.5e8, "ddo210": 2e7,
          "ddo216": 3e7, "ngc1569": 5e8, "ngc2366": 8e8, "ugc8508": 5e7,
          "wlm": 1e8}

    M0_cal = 10 ** rA.x[1]
    alpha_cal = rA.x[2]
    rs, rm, n = 0, 0, 0
    for gn, pts in lt.items():
        M = oh.get(gn)
        if not M:
            continue
        Rl = pts[-1][0]
        u = (M / M0_cal) ** alpha_cal
        p = 2 * u / (1 + 3 * u)
        for Rk, Vc, eV, Sig in pts:
            R = Rk * KPC
            go = (Vc * 1e3) ** 2 / R
            Me = M * MSUN * (Rk / Rl) ** 1.5
            gb = G * Me / R ** 2
            if gb > 0 and go > 0:
                x = gb / A0
                ms = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
                mm = max(1 - np.exp(-x ** 0.5), 1e-20)
                rs += (np.log10(go) - np.log10(gb / ms)) ** 2
                rm += (np.log10(go) - np.log10(gb / mm)) ** 2
                n += 1

    rs = np.sqrt(rs / n)
    rm = np.sqrt(rm / n)
    cv = (rm - rs) / rm * 100
    print(f"  SDHG: {rs:.5f}, McGaugh: {rm:.5f}, improvement: {cv:+.1f}%")
    if cv > 3:
        print("  Cross-validation: CONFIRMED")
    elif cv > -3:
        print("  Cross-validation: NEUTRAL (within noise)")
    else:
        print("  Cross-validation: NEGATIVE (overfitting suspected)")


if __name__ == "__main__":
    main()
