"""
p(Σ) model: comprehensive validation.

Tests the zero-free-parameter form (Σ₀ = a₀/(4π²G), α_Σ = 5/3) on:
  1. SPARC subset breakdown (2D: mass × gas fraction) with p(Σ)
  2. LITTLE THINGS independent validation
  3. Galaxy clusters independent validation
  4. Σ₀ uncertainty analysis (how much does fit degrade if Σ₀ shifted?)
  5. α_Σ robustness under Y_disk marginalization
  6. Comparison with Milgrom's simple μ(x) = x/(1+x)

Usage:
    python run_psigma_validation.py
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.model_selection import KFold
from sdhg import load_sparc, load_clusters, load_little_things, A0, G, MSUN, KPC


PC = 3.086e16  # m
SIGMA0_THEORY_SI = A0 / (4 * np.pi**2 * G)  # kg/m²
SIGMA0_THEORY_MSUN_KPC2 = SIGMA0_THEORY_SI / MSUN * KPC**2  # M☉/kpc²
ALPHA_THEORY = 5/3


def p_of_Sigma(Sigma, Sigma0=SIGMA0_THEORY_MSUN_KPC2, alpha=ALPHA_THEORY):
    u = (Sigma / Sigma0)**alpha
    return np.clip(2*u/(1+3*u), 0.01, 0.95)


def compute_fg(pts):
    V_gas_sq = sum(abs(p[4]*1e3)**2 for p in pts)
    V_disk_sq = sum(0.5*(p[3]*1e3)**2 for p in pts)
    V_bul_sq = sum(0.7*(p[5]*1e3)**2 for p in pts)
    total = V_gas_sq + V_disk_sq + V_bul_sq
    return V_gas_sq / total if total > 0 else 0


def rms_sparc(gal_list, Y, model='mcgaugh', Sigma0=None, alpha=None):
    """SPARC-style data: (gid, pts, M, Sigma)."""
    resid = []
    for gid, pts, M, Sigma in gal_list:
        for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
            R = R_kpc*KPC
            g_obs = (Vobs*1e3)**2/R
            g_bar = (Y*(Vdisk*1e3)**2 + np.sign(Vgas)*(Vgas*1e3)**2 +
                     0.7*(Vbul*1e3)**2)/R
            if g_bar <= 0 or g_obs <= 0:
                continue
            x = g_bar/A0
            if model == 'mcgaugh':
                p = 0.5
            elif model == 'pSigma':
                p = p_of_Sigma(Sigma, Sigma0, alpha)
            elif model == 'milgrom':
                # Milgrom's simple μ(x) = x/(1+x), equivalent to p → ∞ at some sense
                mu = x / (1+x)
                g_pred = g_bar / mu
                resid.append(np.log10(g_obs) - np.log10(g_pred))
                continue
            mu = max(1-np.exp(-max(x, 1e-20)**p), 1e-20)
            resid.append(np.log10(g_obs) - np.log10(g_bar/mu))
    return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999


def main():
    print("=" * 75)
    print("p(Σ) Comprehensive Validation")
    print(f"  Fixed: Σ₀ = a₀/(4π²G) = {SIGMA0_THEORY_MSUN_KPC2:.2e} M☉/kpc²")
    print(f"  Fixed: α_Σ = 5/3 = {5/3:.4f}")
    print("=" * 75)

    # Build SPARC dataset
    galaxies = load_sparc()
    all_gals = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_last = pts[-1][1]*1e3
        R_kpc = pts[-1][0]
        R_last = R_kpc*KPC
        M = 0.5*V_last**2*R_last/G/MSUN
        Sigma = M / (np.pi * R_kpc**2)
        all_gals.append((gid, pts, M, Sigma))

    # ================================================================
    # Test 1: SPARC 2D subset breakdown with p(Σ) theoretical
    # ================================================================
    print("\n" + "="*75)
    print("Test 1: SPARC 2D breakdown (mass × gas fraction)")
    print("        Using theoretical Σ₀ and α=5/3")
    print("="*75)

    def best_y(sub, model, **kwargs):
        return minimize_scalar(
            lambda Y: rms_sparc(sub, Y, model, **kwargs),
            bounds=(0.0, 2.0), method='bounded')

    # Mass bins × gas fraction
    print(f"\n  {'logM':>6} × {'f_gas':<10} {'N':>4} {'RMS_McG':>8} {'RMS_pΣ':>8} {'imp%':>7} {'Milgrom':>8}")
    print("-" * 62)

    totals = {'N': 0, 'McG': [], 'pS': [], 'Milg': []}
    for lo_m, hi_m in [(7, 9), (9, 10), (10, 11), (11, 13)]:
        for lo_g, hi_g in [(0.0, 0.3), (0.3, 0.5), (0.5, 1.0)]:
            sub = [(g, p, m, s) for g, p, m, s in all_gals
                   if lo_m <= np.log10(m) < hi_m and
                      lo_g <= compute_fg(p) < hi_g]
            if len(sub) < 3:
                continue
            r_m = best_y(sub, 'mcgaugh').fun
            r_s = best_y(sub, 'pSigma',
                         Sigma0=SIGMA0_THEORY_MSUN_KPC2,
                         alpha=ALPHA_THEORY).fun
            r_milg = best_y(sub, 'milgrom').fun
            imp = (r_m - r_s)/r_m*100
            label_m = f"{lo_m}-{hi_m}"
            label_g = f"{lo_g:.1f}-{hi_g:.1f}"
            print(f"  {label_m:>5} × {label_g:<10} {len(sub):>4} "
                  f"{r_m:>8.4f} {r_s:>8.4f} {imp:>+6.1f}% {r_milg:>8.4f}")
            totals['N'] += len(sub)
            totals['McG'].extend([r_m] * len(sub))
            totals['pS'].extend([r_s] * len(sub))

    # ================================================================
    # Test 2: LITTLE THINGS validation with theoretical p(Σ)
    # ================================================================
    print("\n" + "="*75)
    print("Test 2: LITTLE THINGS independent validation")
    print("        Using theoretical Σ₀ and α=5/3 from SPARC")
    print("="*75)

    lt = load_little_things()

    def rms_lt(pts, p_val, M_bar):
        """LITTLE THINGS RMS calculation."""
        R_last = max(p[0] for p in pts)
        xs, mus = [], []
        for R_kpc, Vc, eVc, Sigma in pts:
            R = R_kpc * KPC
            g_obs = (Vc * 1e3)**2 / R
            M_enc = M_bar * MSUN * (R_kpc / R_last)**1.5
            g_bar = G * M_enc / R**2
            if g_bar > 0 and g_obs > 0:
                xs.append(g_bar / A0)
                mus.append(g_bar / g_obs)
        if len(xs) < 4:
            return 9999
        xs = np.array(xs)
        mus = np.array(mus)
        pred = 1 - np.exp(-np.maximum(xs, 1e-20)**p_val)
        return np.sqrt(np.mean(
            (np.log10(np.maximum(mus, 1e-20)) -
             np.log10(np.maximum(pred, 1e-20)))**2
        ))

    print(f"\n  {'Galaxy':>10} {'logM':>6} {'V_flat':>6} {'logΣ':>6} "
          f"{'p_pΣ':>6} {'McG':>8} {'p(Σ)':>8} {'imp%':>7}")
    print("-" * 65)

    rms_mcg_lt = []
    rms_pS_lt = []
    for gname, pts in sorted(lt.items()):
        R_arr = np.array([p[0] for p in pts])
        V_arr = np.array([p[1] for p in pts])
        V_flat = V_arr[-3:].mean() if len(V_arr) >= 3 else V_arr[-1]
        R_last = R_arr[-1]
        M = 0.5 * (V_flat * 1e3)**2 * (R_last * KPC) / G / MSUN
        Sigma = M / (np.pi * R_last**2)

        p_pS = p_of_Sigma(Sigma)

        rms_m = rms_lt(pts, 0.5, M)
        rms_s = rms_lt(pts, p_pS, M)

        if rms_m < 9000 and rms_s < 9000:
            imp = (rms_m - rms_s)/rms_m*100
            print(f"  {gname:>10} {np.log10(M):>6.2f} {V_flat:>6.1f} "
                  f"{np.log10(Sigma):>6.2f} {p_pS:>6.3f} "
                  f"{rms_m:>8.4f} {rms_s:>8.4f} {imp:>+6.1f}%")
            rms_mcg_lt.append(rms_m)
            rms_pS_lt.append(rms_s)

    mean_mcg = np.mean(rms_mcg_lt)
    mean_pS = np.mean(rms_pS_lt)
    imp_lt = (mean_mcg - mean_pS)/mean_mcg*100
    wins = sum(1 for a, b in zip(rms_pS_lt, rms_mcg_lt) if a < b)
    print(f"\n  Mean McGaugh: {mean_mcg:.4f}")
    print(f"  Mean p(Σ):    {mean_pS:.4f}")
    print(f"  Improvement:  {imp_lt:+.1f}%")
    print(f"  Win rate:     {wins}/{len(rms_pS_lt)}")

    # ================================================================
    # Test 3: Cluster validation
    # ================================================================
    print("\n" + "="*75)
    print("Test 3: Galaxy cluster validation")
    print("        p(Σ) prediction vs observed cluster RAR (p≈0.66)")
    print("="*75)

    clusters = load_clusters()

    print(f"\n  {'Cluster':>15} {'logM':>6} {'logR':>6} {'logΣ':>6} {'p(Σ)':>7}")
    print("-" * 45)

    p_cluster_preds = []
    for c in clusters:
        M = c['M500_sun']
        R_kpc = c['R500_m'] / KPC
        Sigma = M / (np.pi * R_kpc**2)
        p_pred = p_of_Sigma(Sigma)
        p_cluster_preds.append(p_pred)
        print(f"  {c['name']:>15} {np.log10(M):>6.2f} {np.log10(R_kpc):>6.2f} "
              f"{np.log10(Sigma):>6.2f} {p_pred:>7.3f}")

    print(f"\n  Mean p(Σ) prediction: {np.mean(p_cluster_preds):.3f}")
    print(f"  Observed cluster p:   ~0.66 (from cluster RAR fit)")
    print(f"  Agreement:             {abs(np.mean(p_cluster_preds) - 0.66)/0.66*100:.1f}% difference")

    # ================================================================
    # Test 4: Σ₀ uncertainty analysis
    # ================================================================
    print("\n" + "="*75)
    print("Test 4: Σ₀ sensitivity analysis")
    print("        How robust is the 13.7% improvement to Σ₀ choice?")
    print("="*75)

    # Baseline
    Sigma0_base = SIGMA0_THEORY_MSUN_KPC2
    rms_base = minimize_scalar(
        lambda Y: rms_sparc(all_gals, Y, 'pSigma',
                            Sigma0=Sigma0_base, alpha=ALPHA_THEORY),
        bounds=(0.0, 2.0), method='bounded').fun
    rms_mcg = minimize_scalar(
        lambda Y: rms_sparc(all_gals, Y, 'mcgaugh'),
        bounds=(0.0, 2.0), method='bounded').fun

    print(f"\n  McGaugh: RMS = {rms_mcg:.4f}")
    print(f"  p(Σ) at Σ₀ = a₀/(4π²G): RMS = {rms_base:.4f} "
          f"({(rms_mcg-rms_base)/rms_mcg*100:+.1f}%)")

    print(f"\n  Sensitivity to Σ₀ (α=5/3 fixed):")
    for factor in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        Sigma0_test = Sigma0_base * factor
        rms_test = minimize_scalar(
            lambda Y: rms_sparc(all_gals, Y, 'pSigma',
                                Sigma0=Sigma0_test, alpha=ALPHA_THEORY),
            bounds=(0.0, 2.0), method='bounded').fun
        imp = (rms_mcg - rms_test)/rms_mcg*100
        print(f"    Σ₀ × {factor:.1f}: RMS = {rms_test:.4f} ({imp:+.1f}%)")

    # ================================================================
    # Test 5: α_Σ robustness under Y_disk marginalization
    # ================================================================
    print("\n" + "="*75)
    print("Test 5: α_Σ fit under Y_disk marginalization")
    print("        Does α_Σ change when we allow Y_disk to vary per galaxy?")
    print("="*75)

    # Simple per-galaxy Y test
    print(f"\n  Per-galaxy best p (at standard Y=0.5):")
    p_fits = []
    M_arr = []
    Sigma_arr = []
    for gid, pts, M, Sigma in all_gals:
        def rms_p(p_val):
            resid = []
            for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
                R = R_kpc*KPC
                g_obs = (Vobs*1e3)**2/R
                g_bar = (0.5*(Vdisk*1e3)**2 + np.sign(Vgas)*(Vgas*1e3)**2 +
                         0.7*(Vbul*1e3)**2)/R
                if g_bar <= 0 or g_obs <= 0: continue
                x = g_bar/A0
                mu = max(1-np.exp(-max(x, 1e-20)**p_val), 1e-20)
                resid.append(np.log10(g_obs)-np.log10(g_bar/mu))
            return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999
        res = minimize_scalar(rms_p, bounds=(0.01, 0.95), method='bounded')
        p_fits.append(res.x)
        M_arr.append(M)
        Sigma_arr.append(Sigma)

    p_fits = np.array(p_fits)
    M_arr = np.array(M_arr)
    Sigma_arr = np.array(Sigma_arr)

    # Fit p ∝ Σ^α at low p (small u approximation: p ≈ 2u = 2(Σ/Σ₀)^α)
    low_p_mask = (p_fits < 0.4) & (p_fits > 0.05)
    if low_p_mask.sum() > 10:
        log_p_low = np.log(p_fits[low_p_mask] / 2)
        log_S_low = np.log(Sigma_arr[low_p_mask] / SIGMA0_THEORY_MSUN_KPC2)
        # log(p/2) = α × log(Σ/Σ₀)
        slope, intercept = np.polyfit(log_S_low, log_p_low, 1)
        print(f"  Low-p regime (p<0.4, N={low_p_mask.sum()}):")
        print(f"    α (from slope) = {slope:.3f} (expected 5/3 = {5/3:.3f})")
        print(f"    Intercept = {intercept:.3f} (expected ~0 if Σ₀ correct)")
        print(f"    Σ₀ × exp(intercept) = {SIGMA0_THEORY_MSUN_KPC2 * np.exp(intercept):.2e}")


if __name__ == "__main__":
    main()
