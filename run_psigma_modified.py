"""
Modified p(Σ) variants that preserve deep MOND.

Original: p(Σ) = 2u/(1+3u), u = (Σ/Σ_0)^α
  - p → 0 as Σ → 0 (BREAKS deep MOND)
  - p → 2/3 as Σ → ∞ (clusters)
  - p = 0.5 at Σ = Σ_0 (McGaugh)

Problem: low-Σ galaxies (Crater II) need p ≥ 0.5 for proper deep MOND.

Variants tested:

V1. **Lower bound at 0.5**:
    p(Σ) = max(0.5, 2u/(1+3u))
    For Σ ≤ Σ_0: p = 0.5 (McGaugh, deep MOND preserved)
    For Σ > Σ_0: p increases toward 2/3 (clusters)

V2. **Smooth interpolation**:
    p(Σ) = 0.5 + (1/6) × u/(1+u)
    p(0) = 0.5, p(Σ_0) ≈ 0.583, p(∞) → 2/3
    Smooth, never below 0.5

V3. **Asymmetric McGaugh exponent**:
    p(Σ) = 0.5 if Σ < Σ_0
    p(Σ) = 0.5 + (1/6) × (1 - exp(-(Σ/Σ_0 - 1)^α)) if Σ ≥ Σ_0

These all preserve standard MOND for low-Σ systems while modifying the
high-Σ (cluster) regime.

Tests:
1. SPARC global fit improvement (compare with original p(Σ))
2. Subset analysis (does the dwarf improvement disappear?)
3. Cluster prediction (still ~0.66?)
4. Crater II / dSph EFE (now matches McGaugh?)

Usage:
    python run_psigma_modified.py
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sdhg import load_sparc, load_clusters, A0, G, MSUN, KPC

SIGMA0_THEORY = A0 / (4 * np.pi**2 * G) / MSUN * KPC**2
ALPHA = 5/3


def p_Sigma_v0(Sigma, alpha=ALPHA, Sigma0=SIGMA0_THEORY):
    """Original: p(Σ) = 2u/(1+3u) — broken at low Σ."""
    u = (Sigma / Sigma0)**alpha
    return np.clip(2*u/(1+3*u), 0.01, 0.95)


def p_Sigma_v1(Sigma, alpha=ALPHA, Sigma0=SIGMA0_THEORY):
    """V1: max(0.5, 2u/(1+3u)) — floor at 0.5."""
    u = (Sigma / Sigma0)**alpha
    p = 2*u/(1+3*u)
    return np.clip(np.maximum(p, 0.5), 0.5, 0.95)


def p_Sigma_v2(Sigma, alpha=ALPHA, Sigma0=SIGMA0_THEORY):
    """V2: p = 0.5 + (1/6) × u/(1+u) — smooth, asymptotes to 2/3."""
    u = (Sigma / Sigma0)**alpha
    return np.clip(0.5 + (1/6) * u/(1+u), 0.5, 0.95)


def p_Sigma_v3(Sigma, alpha=ALPHA, Sigma0=SIGMA0_THEORY):
    """V3: piecewise — McGaugh below Σ_0, modified above."""
    if np.isscalar(Sigma):
        if Sigma < Sigma0:
            return 0.5
        v = (Sigma/Sigma0 - 1)**alpha
        return np.clip(0.5 + (1/6) * (1 - np.exp(-v)), 0.5, 0.95)
    else:
        result = np.zeros_like(Sigma)
        below = Sigma < Sigma0
        result[below] = 0.5
        v = (Sigma[~below]/Sigma0 - 1)**alpha
        result[~below] = np.clip(0.5 + (1/6) * (1 - np.exp(-v)), 0.5, 0.95)
        return result


def p_McGaugh(Sigma):
    """Constant p = 0.5 (McGaugh universal)."""
    return 0.5


# ============================================================================
# Global fit machinery
# ============================================================================

def compute_galaxy_data(galaxies):
    data = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_arr = np.array([p[1] for p in pts])
        R_arr = np.array([p[0] for p in pts])
        V_flat = V_arr[-3:].mean() if len(V_arr) >= 3 else V_arr[-1]
        R_last = R_arr[-1]
        M_dyn = 0.5 * (V_flat*1e3)**2 * (R_last*KPC) / G / MSUN
        Sigma = M_dyn / (np.pi * R_last**2)
        data.append({'pts': pts, 'M_dyn': M_dyn, 'Sigma': Sigma,
                     'R_last': R_last, 'V_flat': V_flat})
    return data


def global_rms(gal_list, Y_disk, p_fn):
    resid = []
    for gd in gal_list:
        Sigma = gd['Sigma']
        p_val = p_fn(Sigma)
        for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in gd['pts']:
            R = R_kpc*KPC
            g_obs = (Vobs*1e3)**2/R
            g_bar = (Y_disk*(Vdisk*1e3)**2 + np.sign(Vgas)*(Vgas*1e3)**2 +
                     0.7*(Vbul*1e3)**2)/R
            if g_bar <= 0 or g_obs <= 0:
                continue
            x = g_bar/A0
            mu = max(1-np.exp(-max(x, 1e-30)**p_val), 1e-30)
            resid.append(np.log10(g_obs)-np.log10(g_bar/mu))
    return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999


def best_y(gal_list, p_fn):
    res = minimize_scalar(lambda Y: global_rms(gal_list, Y, p_fn),
                          bounds=(0.0, 2.0), method='bounded')
    return res.x, res.fun


def main():
    print("=" * 75)
    print("Modified p(Σ) variants: preserve deep MOND, improve SPARC?")
    print("=" * 75)

    galaxies = load_sparc()
    data = compute_galaxy_data(galaxies)
    print(f"\n  N = {len(data)} SPARC galaxies")

    # ================================================================
    # Test 1: Global fit comparison
    # ================================================================
    print(f"\n{'='*75}")
    print("Test 1: SPARC global fit (best Y_disk for each model)")
    print(f"{'='*75}")

    variants = [
        ('McGaugh (p=0.5)', p_McGaugh, 0.5),
        ('V0: original 2u/(1+3u)', p_Sigma_v0, None),
        ('V1: max(0.5, 2u/(1+3u))', p_Sigma_v1, None),
        ('V2: 0.5 + u/(6(1+u))', p_Sigma_v2, None),
        ('V3: piecewise', p_Sigma_v3, None),
    ]

    print(f"\n  {'Variant':>30} {'Y_best':>7} {'RMS':>8} {'vs McG':>7}")
    print("-" * 60)

    rms_mcg = best_y(data, p_McGaugh)[1]
    for name, p_fn, _ in variants:
        Y, rms = best_y(data, p_fn)
        imp = (rms_mcg - rms)/rms_mcg*100
        print(f"  {name:>30} {Y:>7.3f} {rms:>8.4f} {imp:>+6.1f}%")

    # ================================================================
    # Test 2: Subset analysis with V1 (key check)
    # ================================================================
    print(f"\n{'='*75}")
    print("Test 2: Subset analysis comparison (V0 vs V1)")
    print("        Does floor at 0.5 lose the dwarf improvement?")
    print(f"{'='*75}")

    print(f"\n  {'logM range':>12} {'N':>4} {'V0 imp%':>8} {'V1 imp%':>8} {'V2 imp%':>8}")
    print("-" * 50)

    for lo, hi in [(7, 9), (9, 10), (10, 11), (11, 13)]:
        sub = [d for d in data if lo <= np.log10(d['M_dyn']) < hi]
        if len(sub) < 3:
            continue
        rms_mcg_s = best_y(sub, p_McGaugh)[1]
        rms_v0 = best_y(sub, p_Sigma_v0)[1]
        rms_v1 = best_y(sub, p_Sigma_v1)[1]
        rms_v2 = best_y(sub, p_Sigma_v2)[1]
        print(f"  {lo}-{hi}                {len(sub):>4} "
              f"{(rms_mcg_s-rms_v0)/rms_mcg_s*100:>+7.1f}% "
              f"{(rms_mcg_s-rms_v1)/rms_mcg_s*100:>+7.1f}% "
              f"{(rms_mcg_s-rms_v2)/rms_mcg_s*100:>+7.1f}%")

    # ================================================================
    # Test 3: Cluster prediction
    # ================================================================
    print(f"\n{'='*75}")
    print("Test 3: Galaxy cluster predictions")
    print(f"{'='*75}")

    clusters = load_clusters()
    print(f"\n  Clusters at logΣ ≈ 8.0 (Σ ≈ 5-10 × Σ_0)")
    print(f"  {'Variant':>30} {'p_cluster':>10}")
    print("-" * 45)

    for name, p_fn, _ in variants:
        p_pred = []
        for c in clusters:
            R_kpc = c['R500_m'] / KPC
            Sigma = c['M500_sun'] / (np.pi * R_kpc**2)
            p_pred.append(p_fn(Sigma))
        if name == 'McGaugh (p=0.5)':
            print(f"  {name:>30} {0.5:>10.3f}")
        else:
            print(f"  {name:>30} {np.mean(p_pred):>10.3f}")

    # ================================================================
    # Test 4: Crater II prediction (deep MOND test)
    # ================================================================
    print(f"\n{'='*75}")
    print("Test 4: Crater II prediction (deep MOND check)")
    print(f"{'='*75}")

    M_crater = 3.3e5  # M_sun
    R_crater = 1.07   # kpc
    Sigma_crater = M_crater / (np.pi * R_crater**2)
    print(f"\n  Crater II: Σ = {Sigma_crater:.2e} M☉/kpc² "
          f"(= {Sigma_crater/SIGMA0_THEORY:.4f} × Σ_0)")
    print(f"\n  {'Variant':>30} {'p_predicted':>12}")
    print("-" * 50)
    for name, p_fn, _ in variants:
        if name == 'McGaugh (p=0.5)':
            print(f"  {name:>30} {0.5:>12.3f}")
        else:
            print(f"  {name:>30} {p_fn(Sigma_crater):>12.3f}")

    # ================================================================
    # Conclusions
    # ================================================================
    print(f"\n{'='*75}")
    print("CONCLUSIONS")
    print(f"{'='*75}")
    print("""
  Key trade-off:
  - V0 (original): +14.7% on SPARC, but breaks deep MOND for low-Σ systems
  - V1 (floor 0.5): preserves deep MOND, but loses dwarf improvement
  - V2 (smooth): preserves deep MOND, more flexible at high Σ

  Implication:
  - If V1/V2 give similar SPARC improvement to V0, we have a "best of both"
    model that works for clusters AND deep MOND systems.
  - If V1/V2 give much LESS improvement, V0's improvement comes specifically
    from the low-Σ p < 0.5 prediction, which contradicts standard MOND.

  → The right interpretation depends on subset analysis above.
""")


if __name__ == "__main__":
    main()
