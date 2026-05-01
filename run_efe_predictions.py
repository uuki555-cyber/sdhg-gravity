"""
EFE-corrected p(Σ) predictions for UDGs and Milky Way satellites.

External Field Effect (EFE) is essential for satellite/group galaxies in MOND.
McGaugh (2016, 2025) and Famaey & McGaugh (2012) prescription:

  In the deep-MOND regime (g_in << g_ex << a_0), the effective gravity is
  rescaled by the external field. The asymptotic isothermal velocity dispersion
  is:

  σ⁴_iso = (4/9) G M a_0  (no EFE)
  σ⁴_EFE ≈ (4/9) G M a_0 × ν_e × ε_e  (with EFE)

  where ν_e = ν(g_ex/a_0) is the interpolation function evaluated at the
  external field, and ε_e captures dimensional factors.

For p(Σ), the modification is:
  - p(Σ) controls μ(x) = 1 - exp(-x^p)
  - In the deep-MOND limit at low x (= g_bar/a_0 << 1):
    μ ≈ x^p, so g_obs ≈ g_bar / x^p = a_0^p × g_bar^(1-p)
  - For p = 0.5: g_obs ≈ √(g_bar a_0) (standard deep MOND)
  - For p < 0.5: g_obs grows slower with g_bar
  - For p > 0.5: g_obs grows faster

EFE in p(Σ) framework:
  When g_bar << g_ex, the effective dynamics is set by g_ex.
  We approximate the EFE-corrected velocity dispersion using:
    σ²_EFE = σ²_iso × √(g_in / (g_in + g_ex))
  (Famaey-McGaugh interpolation in the deep-MOND limit)

Tests:
  - Crater II: σ_observed = 2.7 km/s
  - Antlia II: σ_observed = 5.7 km/s (tidally disrupted)
  - DF-2 / DF-4: low σ (8.5 / 4.2 km/s) — group EFE
  - Leo P: V_circ = 15 km/s (truly isolated, V_c not σ)
  - DF-44: σ = 33 km/s (Coma cluster)

Usage:
    python run_efe_predictions.py
"""
import numpy as np
from sdhg import A0, G, MSUN, KPC

PC = 3.086e16  # m
SIGMA0_THEORY = A0 / (4 * np.pi**2 * G) / MSUN * KPC**2
ALPHA_THEORY = 5/3


def p_Sigma(Sigma, Sigma0=SIGMA0_THEORY, alpha=ALPHA_THEORY):
    u = (Sigma / Sigma0)**alpha
    return float(np.clip(2*u/(1+3*u), 0.01, 0.95))


def sigma_iso_deep_mond(M_baryon_kg, a_0):
    """Deep MOND isolated isothermal velocity dispersion.

    σ⁴ = (4/9) G M a_0
    """
    sigma_4 = (4/9) * G * M_baryon_kg * a_0
    return sigma_4**(1/4)


def sigma_iso_pSigma(M_baryon, R_eff, p_val):
    """Isolated p(Σ) prediction for velocity dispersion.

    Generalized: σ²(R_eff) ≈ g_obs(R_eff) × R_eff / 2.5
    where g_obs = g_bar / μ, μ = 1 - exp(-x^p)
    """
    M_kg = M_baryon * MSUN
    R_m = R_eff * KPC
    g_bar = G * M_kg / R_m**2
    x = g_bar / A0
    if x <= 0:
        return 0
    mu = max(1 - np.exp(-max(x, 1e-30)**p_val), 1e-30)
    g_obs = g_bar / mu
    sigma_squared = g_obs * R_m / 2.5  # King-like virial
    return np.sqrt(sigma_squared) / 1e3  # km/s


def efe_correction_factor(g_in, g_ex, a_0):
    """EFE correction factor in deep-MOND limit.

    Famaey-McGaugh (2012) external field effect approximation:
    For g_in << g_ex << a_0, the effective acceleration is suppressed.

    Simple approximation:
      g_obs_EFE/g_obs_iso ≈ √(g_in / (g_in + g_ex))   [Bekenstein-Milgrom-like]

    More detailed: depends on direction (perpendicular vs parallel external field)
    """
    if g_ex <= 0:
        return 1.0
    return np.sqrt(g_in / (g_in + g_ex))


def predict_udg(name, M_bar, R_eff, host_M=None, host_distance=None,
                p_method='pSigma'):
    """Predict velocity dispersion for a UDG/satellite, with optional EFE.

    Parameters:
      M_bar: baryonic mass in M_sun
      R_eff: effective radius in kpc
      host_M: if satellite, host galaxy mass in M_sun (sets external field)
      host_distance: distance from host in kpc
      p_method: 'pSigma' or 'mcgaugh'
    """
    Sigma = M_bar / (np.pi * R_eff**2)
    p = p_Sigma(Sigma) if p_method == 'pSigma' else 0.5

    # Internal acceleration
    M_kg = M_bar * MSUN
    R_m = R_eff * KPC
    g_in = G * M_kg / R_m**2

    # Isolated prediction
    sigma_iso = sigma_iso_pSigma(M_bar, R_eff, p)

    # External field
    if host_M is not None and host_distance is not None:
        host_kg = host_M * MSUN
        host_dist_m = host_distance * KPC
        g_ex_newton = G * host_kg / host_dist_m**2
        # MOND-corrected external field (deep MOND if g_ex_newton < a_0)
        if g_ex_newton < A0:
            g_ex = np.sqrt(g_ex_newton * A0)
        else:
            g_ex = g_ex_newton

        efe_factor = efe_correction_factor(g_in, g_ex, A0)
        sigma_efe = sigma_iso * np.sqrt(efe_factor)
    else:
        sigma_efe = sigma_iso
        g_ex = 0
        efe_factor = 1.0

    return {
        'name': name,
        'M_bar': M_bar,
        'R_eff': R_eff,
        'Sigma': Sigma,
        'logSigma': np.log10(Sigma),
        'p': p,
        'g_in': g_in,
        'g_ex': g_ex,
        'g_in_over_a0': g_in/A0,
        'g_ex_over_a0': g_ex/A0 if g_ex > 0 else 0,
        'efe_factor': efe_factor,
        'sigma_iso': sigma_iso,
        'sigma_efe': sigma_efe,
    }


def main():
    print("=" * 80)
    print("EFE-corrected p(Σ) predictions for UDGs and satellites")
    print("=" * 80)

    # Σ_0 details
    print(f"\n  Σ_0 = a_0/(4π²G) = {SIGMA0_THEORY:.2e} M☉/kpc²")
    print(f"      = {SIGMA0_THEORY/1e6:.2f} M☉/pc²")
    print(f"  α   = {ALPHA_THEORY}")

    # ================================================================
    # Test 1: Milky Way satellites with EFE (host: MW M ~ 1e12 M_sun)
    # ================================================================
    print(f"\n{'='*80}")
    print("Test 1: Milky Way satellites (with EFE from MW)")
    print(f"{'='*80}")

    MW_mass = 1e12  # baryonic + DM, total

    # (name, M_bar [M_sun], R_eff [kpc], dist_from_MW [kpc], σ_obs [km/s])
    mw_satellites = [
        ('Crater II',  3.3e5,  1.07,  117,   2.7),
        ('Antlia II',  1.0e6,  2.9,   130,   5.7),
        ('Sculptor',   2.3e6,  0.28,   86,   9.2),  # known classic dSph
        ('Fornax',     2.0e7,  0.71,  149,  11.7),
        ('Draco',      2.9e5,  0.22,   76,   9.1),
    ]

    print(f"\n  {'Name':>12} {'M_bar':>9} {'R_eff':>5} {'Σ':>8} {'p(Σ)':>6} "
          f"{'σ_iso':>6} {'σ_EFE':>6} {'σ_obs':>6} {'EFE':>5}")
    print("-" * 75)
    for name, M_bar, R_eff, dist, sigma_obs in mw_satellites:
        result = predict_udg(name, M_bar, R_eff, host_M=MW_mass, host_distance=dist)
        print(f"  {name:>12} {M_bar:>9.1e} {R_eff:>5.2f} "
              f"{result['logSigma']:>8.2f} {result['p']:>6.3f} "
              f"{result['sigma_iso']:>6.2f} {result['sigma_efe']:>6.2f} "
              f"{sigma_obs:>6.2f} {result['efe_factor']:>5.3f}")

    # ================================================================
    # Test 2: NGC1052 group satellites (DF-2, DF-4)
    # ================================================================
    print(f"\n{'='*80}")
    print("Test 2: NGC1052 group satellites (DF-2, DF-4)")
    print(f"{'='*80}")

    NGC1052_mass = 1e11  # rough estimate

    ngc1052_sats = [
        ('NGC1052-DF2', 2e8, 2.0, 80,    8.5),  # group context, ~80 kpc?
        ('NGC1052-DF4', 1.5e8, 1.6, 80,  4.2),
    ]

    print(f"\n  {'Name':>12} {'M_bar':>9} {'R_eff':>5} {'Σ':>8} {'p(Σ)':>6} "
          f"{'σ_iso':>6} {'σ_EFE':>6} {'σ_obs':>6} {'EFE':>5}")
    print("-" * 75)
    for name, M_bar, R_eff, dist, sigma_obs in ngc1052_sats:
        result = predict_udg(name, M_bar, R_eff,
                             host_M=NGC1052_mass, host_distance=dist)
        print(f"  {name:>12} {M_bar:>9.1e} {R_eff:>5.2f} "
              f"{result['logSigma']:>8.2f} {result['p']:>6.3f} "
              f"{result['sigma_iso']:>6.2f} {result['sigma_efe']:>6.2f} "
              f"{sigma_obs:>6.2f} {result['efe_factor']:>5.3f}")

    # ================================================================
    # Test 3: Coma cluster UDGs (DF-44)
    # ================================================================
    print(f"\n{'='*80}")
    print("Test 3: Coma cluster UDGs (high external field)")
    print(f"{'='*80}")

    Coma_mass = 1.5e15  # M_500
    Coma_R500 = 1.5  # Mpc (= 1500 kpc)

    coma_udgs = [
        ('DF-44', 3e8, 4.7, 500, 33),  # 500 kpc from Coma center?
    ]

    print(f"\n  {'Name':>10} {'M_bar':>9} {'R_eff':>5} {'Σ':>8} {'p(Σ)':>6} "
          f"{'σ_iso':>6} {'σ_EFE':>6} {'σ_obs':>6} {'EFE':>5}")
    print("-" * 75)
    for name, M_bar, R_eff, dist, sigma_obs in coma_udgs:
        result = predict_udg(name, M_bar, R_eff,
                             host_M=Coma_mass, host_distance=dist)
        print(f"  {name:>10} {M_bar:>9.1e} {R_eff:>5.2f} "
              f"{result['logSigma']:>8.2f} {result['p']:>6.3f} "
              f"{result['sigma_iso']:>6.2f} {result['sigma_efe']:>6.2f} "
              f"{sigma_obs:>6.2f} {result['efe_factor']:>5.3f}")

    # ================================================================
    # Test 4: Truly isolated dwarfs (no EFE)
    # ================================================================
    print(f"\n{'='*80}")
    print("Test 4: Isolated dwarfs (no significant EFE)")
    print(f"{'='*80}")

    isolated = [
        ('Leo P (V_c)', 0.57e6, 0.5, None, 15.0),  # V_c not σ
    ]

    print(f"\n  {'Name':>15} {'M_bar':>9} {'R_eff':>5} {'Σ':>8} {'p(Σ)':>6} "
          f"{'σ_iso':>6} {'σ_obs':>6}")
    print("-" * 60)
    for name, M_bar, R_eff, dist, sigma_obs in isolated:
        result = predict_udg(name, M_bar, R_eff)
        print(f"  {name:>15} {M_bar:>9.1e} {R_eff:>5.2f} "
              f"{result['logSigma']:>8.2f} {result['p']:>6.3f} "
              f"{result['sigma_iso']:>6.2f} {sigma_obs:>6.2f}")

    # ================================================================
    # Test 5: Comparison with McGaugh (p=0.5)
    # ================================================================
    print(f"\n{'='*80}")
    print("Test 5: p(Σ) vs McGaugh predictions side-by-side (with EFE)")
    print(f"{'='*80}")

    print(f"\n  {'Name':>12} {'σ_obs':>6} {'McG_iso':>7} {'McG_EFE':>8} "
          f"{'pΣ_iso':>7} {'pΣ_EFE':>7} {'Best':>10}")
    print("-" * 75)

    test_galaxies = [
        ('Crater II', 3.3e5, 1.07, MW_mass, 117, 2.7),
        ('Antlia II', 1.0e6, 2.9, MW_mass, 130, 5.7),
        ('Sculptor', 2.3e6, 0.28, MW_mass, 86, 9.2),
        ('Fornax', 2.0e7, 0.71, MW_mass, 149, 11.7),
        ('NGC1052-DF2', 2e8, 2.0, NGC1052_mass, 80, 8.5),
        ('DF-44', 3e8, 4.7, Coma_mass, 500, 33),
        ('Leo P', 0.57e6, 0.5, None, None, 15.0),  # V_c not σ
    ]

    for name, M, R, host, dist, obs in test_galaxies:
        # McGaugh (p=0.5)
        r_mcg = predict_udg(name, M, R, host_M=host, host_distance=dist,
                            p_method='mcgaugh')
        # p(Σ)
        r_pS = predict_udg(name, M, R, host_M=host, host_distance=dist,
                           p_method='pSigma')

        # Determine best
        err_mcg = abs(r_mcg['sigma_efe'] - obs)
        err_pS = abs(r_pS['sigma_efe'] - obs)
        best = 'p(Σ)' if err_pS < err_mcg else 'McG' if err_mcg < err_pS else 'tie'

        print(f"  {name:>12} {obs:>6.2f} {r_mcg['sigma_iso']:>7.2f} "
              f"{r_mcg['sigma_efe']:>8.2f} {r_pS['sigma_iso']:>7.2f} "
              f"{r_pS['sigma_efe']:>7.2f} {best:>10}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*80}")
    print("HONEST INTERPRETATION:")
    print(f"{'='*80}")
    print("""
  1. EFE matters significantly for satellite galaxies. Both McGaugh and p(Σ)
     produce different predictions with vs without EFE.

  2. For Crater II (well-studied EFE case), MOND+EFE works well.
     p(Σ)+EFE gives similar values via different mechanism (low Σ → low p).
     Hard to distinguish from data; this is a coincidence-friendly case.

  3. For DF-2 (notoriously low σ), even MOND+EFE gives ~13 km/s vs observed
     ~8.5. McGaugh argues non-equilibrium dynamics. p(Σ)+EFE doesn't resolve.

  4. For DF-44 (Coma cluster member), the high external field (Coma is dense)
     suppresses σ, making both McGaugh+EFE and p(Σ)+EFE similar.

  5. For truly isolated systems (Leo P), the prediction is V_c (not σ).
     Different formalism needed.

  6. DISTINGUISHING TEST: A truly isolated low-Σ galaxy (Σ << Σ_0, no EFE)
     where p(Σ) predicts p << 0.5, vs McGaugh predicting p = 0.5.
     Such galaxies are rare; isolated UDGs would qualify but few are
     well-measured.

  DON'T overclaim: p(Σ) does NOT clearly win against McGaugh+EFE for
  the systems tested. The signature distinguishing prediction is in
  isolated low-Σ equilibrium systems.
""")


if __name__ == "__main__":
    main()
