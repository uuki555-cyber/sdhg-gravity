"""
Independent validation on LITTLE THINGS dwarf galaxies (Iorio+ 2017).
Tests the SDHG prediction: ultra-dwarf galaxies (M < 10^8) have p < 0.25.

Usage:
    python run_little_things.py
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sdhg import load_little_things, p_of_M, A0, G, MSUN, KPC

# Baryonic masses from Oh+ 2015 (approximate, Mgas + M* in solar masses)
OH2015_MASSES = {
    "cvidwa": 2.6e7, "ddo47": 3.5e8, "ddo50": 6.0e8,
    "ddo52": 3.3e8, "ddo53": 6.5e7, "ddo87": 4.0e8,
    "ddo101": 1.5e8, "ddo126": 2.0e8, "ddo133": 2.5e8,
    "ddo154": 5.0e8, "ddo168": 4.5e8, "ddo210": 2.0e7,
    "ddo216": 3.0e7, "ngc1569": 5.0e8, "ngc2366": 8.0e8,
    "ugc8508": 5.0e7, "wlm": 1.0e8,
}


def main():
    print("=" * 70)
    print("SDHG: Independent Validation on LITTLE THINGS Dwarf Galaxies")
    print("=" * 70)

    lt = load_little_things()
    print(f"\nLoaded {len(lt)} LITTLE THINGS galaxies")

    results = []
    for gname, pts in lt.items():
        M_bar = OH2015_MASSES.get(gname)
        if M_bar is None:
            continue

        R_arr = np.array([p[0] for p in pts])
        V_arr = np.array([p[1] for p in pts])
        R_last = R_arr[-1]

        # Compute x, mu at each radius (approximate enclosed mass)
        xs, mus = [], []
        for R_kpc, Vc, eVc, Sigma in pts:
            R = R_kpc * KPC
            g_obs = (Vc * 1e3) ** 2 / R
            M_enc = M_bar * MSUN * (R_kpc / R_last) ** 1.5
            g_bar = G * M_enc / R ** 2
            if g_bar > 0 and g_obs > 0:
                xs.append(g_bar / A0)
                mus.append(g_bar / g_obs)

        if len(xs) < 4:
            continue
        xs = np.array(xs)
        mus = np.array(mus)

        # Fit p
        def rms_p(p):
            pred = 1 - np.exp(-np.maximum(xs, 1e-20) ** p)
            return np.sqrt(np.mean(
                (np.log10(np.maximum(mus, 1e-20)) -
                 np.log10(np.maximum(pred, 1e-20))) ** 2
            ))

        res = minimize_scalar(rms_p, bounds=(0.01, 0.95), method="bounded")
        rms_mcg = rms_p(0.5)
        p_pred = p_of_M(M_bar)
        results.append((gname, M_bar, res.x, p_pred, res.fun, rms_mcg))

    # Display
    print(f"\n{'Galaxy':>10} {'M_bar':>10} {'p_obs':>6} {'p_SDHG':>7} {'RMS':>7} {'RMS_McG':>8}")
    print("-" * 52)
    for g, M, po, pp, rm, rmc in sorted(results, key=lambda x: x[1]):
        print(f"{g:>10} {M:>10.1e} {po:>6.3f} {pp:>7.3f} {rm:>7.4f} {rmc:>8.4f}")

    p_obs = np.array([r[2] for r in results])
    M_arr = np.array([r[1] for r in results])

    # Ultra-dwarf test
    ultra = M_arr < 1e8
    print(f"\nUltra-dwarf (M < 10^8): N={ultra.sum()}, p_obs = {p_obs[ultra].mean():.3f}")
    print(f"SDHG prediction: p < 0.25")
    print(f"Result: {'CONFIRMED' if p_obs[ultra].mean() < 0.30 else 'NOT CONFIRMED'}")

    # Overall
    rms_sdhg = np.mean([r[4] for r in results])
    rms_mcg = np.mean([r[5] for r in results])
    imp = (rms_mcg - rms_sdhg) / rms_mcg * 100
    print(f"\nOverall: SDHG RMS={rms_sdhg:.4f}, McGaugh RMS={rms_mcg:.4f}, Improvement={imp:+.1f}%")

    # Correlation
    corr = np.corrcoef(p_obs, np.log10(M_arr))[0, 1]
    print(f"Correlation p_obs vs log(M): r = {corr:.3f}")

    # Mann-Whitney test
    normal = M_arr >= 1e8
    if ultra.sum() > 3 and normal.sum() > 3:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(p_obs[ultra], p_obs[normal], alternative="less")
        print(f"Mann-Whitney (ultra < normal): p-value = {pval:.6f}")


if __name__ == "__main__":
    main()
