# Mass-Dependent Radial Acceleration Relation

**The exponent of the RAR interpolation function depends on system mass.**

This repository presents evidence that the Radial Acceleration Relation ([McGaugh, Lelli & Schombert 2016, PRL 117, 201101](https://doi.org/10.1103/PhysRevLett.117.201101)) is better described by an interpolation function whose exponent varies with the mass of the system:

```
mu(x, M) = 1 - exp(-x^p(M))

p(M) = 2u / (1 + 3u),   u = (M / M0)^(1/3)
```

where `x = g_bar / a0`, `M` is the system mass (see [mass definition note](#limitations)), and `M0 ~ 10^10.2 solar masses`. The exponent 1/3 is empirically verified: fixing alpha = 1/3 degrades the global fit by only 0.02% compared to the free fit (alpha = 0.312), while alpha = 1/4 degrades by 0.26%.

For the standard McGaugh formula, `p = 0.5` for all systems. We find that `p` increases from ~0.2 for dwarf galaxies to ~0.66 for galaxy clusters, and a single formula describes both galaxy rotation curves and cluster mass discrepancies.

![p vs mass](figures/fig2_p_vs_mass.png)

![Global fit comparison](figures/fig3_global_fit.png)

## Key Result

In a **global fit with no per-galaxy fitting** (single Y_disk for all galaxies):

| Model | Free params | RMS | vs McGaugh |
|-------|-------------|-----|------------|
| McGaugh (p = 0.5) | 1 | 0.197 | baseline |
| Constant p | 2 | 0.195 | +0.9% |
| Variable G(M) [control] | 3 | 0.188 | +4.6% |
| **p(M) model** | **3** | **0.166** | **+15.5%** |

The p(M) model improves over McGaugh by 15.5%, while a variable-G model (same number of parameters) only achieves 4.6%. The 11% difference disfavors Y_disk degeneracy as the source of the signal, though a definitive exclusion requires more gas-dominated dwarf galaxy data (see [Li+ 2021 bias test](#bias-test-li-2021) below).

The improvement comes from correcting a **mass-dependent systematic bias** in the standard RAR (`run_bias_analysis.py`): MOND (p=0.5) systematically overpredicts g_obs for dwarf galaxies (mean bias = -0.37 dex at M < 10^9) and is approximately unbiased for massive galaxies. The correlation between MOND bias and mass is r = +0.47; p(M) reduces this to r = +0.12.

## Bias Test (Li+ 2021)

Li, Lelli, McGaugh, Schombert & Chae ([2021, A&A 646, L13](https://doi.org/10.1051/0004-6361/202040101)) showed that fitting a parameter per galaxy can create spurious mass dependence due to degeneracy with the stellar mass-to-light ratio Y_disk. We address this in three ways:

1. **Global fit (`run_global_fit.py`)**: No per-galaxy fitting at all. Single Y_disk shared by all galaxies. p(M) still improves by 15.5%, while the control model (variable G with same degrees of freedom) achieves only 4.6%. The 11% gap disfavors Y_disk degeneracy as the sole explanation.

2. **Bayesian marginalization (`run_bayesian_test.py`)**: Marginalizing over Y_disk ~ N(0.5, 0.15) and distance ~ N(1.0, 0.10), the per-galaxy p correlation with mass (r=0.18) exceeds the G_eff control (r=0.12). Suggestive but not definitive from galaxy data alone.

3. **Galaxy clusters are immune**: Cluster data involves no per-galaxy fitting and independently requires p ~ 0.66, far from the galaxy-optimal p ~ 0.5.

## Quick Start

```bash
pip install numpy scipy             # matplotlib also needed for make_figures.py
python run_global_fit.py            # Core result (no per-galaxy fitting, bias-free)
python run_main_analysis.py         # Detailed analysis (per-galaxy Y_disk, subject to Li+ caveat)
python run_little_things.py         # Independent validation on dwarf galaxies
python run_bayesian_test.py         # Li+ (2021) methodology check
python run_bias_analysis.py         # MOND systematic bias vs mass
python run_slope_test.py            # Rotation curve shape test (slope vs mass)
python run_loo_cv.py               # Leave-one-out cross-validation (slow)
python run_model_comparison.py     # BIC/AIC model selection (dBIC=38.4 vs McGaugh)
python run_baryonic_mass_fit.py    # Baryonic vs dynamical mass comparison
python run_cdt_2plus1d.py          # 2+1D CDT simulation (~30 min, Python reference)
python cdt_pachner.py              # 2+1D CDT with Pachner moves (Python)
# C CDT (compile first: cl /O2 /std:c11 /Fe:cdt_main.exe cdt_main.c)
# ./cdt_main.exe 20 30 5000 200000 1000000 300 5 1 42
# ./cdt4d.exe 8 12 2000000 400 42 200000 5 1   # 4D CDT
```

## Data

- `data/sparc_data.mrt`: SPARC mass models for 175 disk galaxies (Lelli, McGaugh & Schombert 2016, AJ 152, 157)
- `data/little_things/finalrot/`: Rotation curves for 17 dwarf irregular galaxies (Iorio+ 2017, MNRAS 466, 4159)

## CDT Code

| File | Purpose | Status |
|------|---------|--------|
| `cdt_main.c` | 2+1D CDT, all 4 Pachner moves + Regge action | **Production** |
| `cdt4d.c` | 3+1D CDT, (2,4)+(4,2) moves | **Production** |
| `cdt_pachner.py` | 2+1D CDT Python prototype | **Reference** |
| `run_cdt_2plus1d.py` | Original pure-Python CDT | **Legacy** |
| `cdt_sim.c`, `cdt_full.c`, `cdt_large.c`, `cdt_proper.c`, `cdt_fast.py` | Development artifacts | **Archived** |
- Galaxy cluster data from Vikhlinin+ 2006 (ApJ 640, 691) and X-COP/Ettori+ 2019 (A&A 621, A39), hardcoded in `sdhg/data.py`

## What This Is and Is Not

**This is**: An observational finding that the RAR interpolation exponent correlates with system mass, tested against methodological bias concerns.

**This is not**: A new theory of gravity, a claim about dark matter, or a peer-reviewed result.

## Limitations

- **Cross-validation is mixed**: Leave-one-out within SPARC gives +5.4% improvement (see below), confirming p(M) is not overfitting. However, applying SPARC-trained parameters to LITTLE THINGS gives -0.8%, suggesting limited generalization across datasets (possibly due to different mass estimation methods)
- **Rotation curve shapes are not independent evidence** (`run_slope_test.py`): The outer slope of rotation curves correlates with mass (r = -0.62), but standard MOND (p=0.5) predicts this equally well (r = -0.63) from baryonic mass distributions alone. The p(M) improvement comes from the RAR *amplitude* (systematic offset), not the curve *shape*
- **Gas-dominated test is inconclusive**: In gas-dominated galaxies (f_gas > 0.5, where Y_disk is less relevant), the MOND bias-mass correlation is r = +0.37 (N=22), suggestive but with small sample size. We cannot fully rule out that the bias is caused by Y_disk systematics rather than gravitational physics
- M0 is uncertain by a factor of ~5 (10^10.0 to 10^10.8), depending on fitting method and cluster weighting
- The exponent 1/3 in the formula is approximate; the global fit gives alpha = 0.31 (6% below 1/3). **CDT simulation** (`cdt_main.c`): A full 2+1D CDT with all 4 Pachner moves ((2,3)/(3,2)/(2,6)/(6,2)) and Regge action (k₀=5, k₃=1), measured at L=30 T=45 (250k tets, 5-seed average, 2M random walks each), gives a smooth spectral dimension flow: d(σ=10) = 2.22 ± 0.002 → d(σ=100) = 3.90 ± 0.12. Fitting the SDHG formula yields **gamma = 0.87 ± 0.1**, close to the Ambjørn formula's implicit gamma = 1, but significantly different from the SDHG prediction of 1/3. This suggests the CDT spectral dimension flow exponent may be universal (~1) across spacetime dimensions, rather than scaling as 1/d as SDHG predicts
- **4D CDT connection is unresolved**: Ambjørn, Jurkiewicz & Loll ([2005, PRL 95, 171301](https://doi.org/10.1103/PhysRevLett.95.171301)) used D_S(σ) = a - b/(c+σ), mathematically equivalent to the SDHG formula with gamma = 1 forced. Fitting their data (σ = 40–400) with free gamma: RMS = 0.001 (gamma=1.0), 0.112 (gamma=1.5), 0.126 (gamma=0.5), 0.190 (gamma=0.25). The data clearly prefers gamma = 1. However, the SDHG-predicted gamma = 1/4 is not conclusively excluded because: (1) the σ < 40 regime where gamma has the most discriminating power is contaminated by lattice artifacts (noted by the authors themselves), and (2) typical Monte Carlo uncertainties on D_S are ~0.2–0.3, comparable to the RMS difference. Resolving this requires 4D CDT data at smaller σ with controlled systematics
- The functional form p(M) = 2u/(1+3u) is mathematically identical to the CDT spectral dimension formula with gamma=1: p = (2/3)*v/(1+v) where v=3u. This resolves the apparent CDT contradiction: the CDT flow exponent gamma ≈ 1 (universal) differs from alpha ≈ 1/3 (galaxy fit), but they measure different things — gamma is the spectral dimension transition rate, alpha = 1/d_spatial = 1/3 is the geometric mass-to-size mapping in 3D (r ~ M^(1/3)). The M₀ transformation M₀_CDT = M₀_SDHG / 3^(1/alpha) matches to 0.00 dex precision
- Galaxy masses used in p(M) are dynamical proxies (M ~ 0.5 V_flat^2 R_last / G), not photometric baryonic masses. In an offline variant of the global fit using photometric baryonic masses (from Vdisk, Vgas), the improvement reduces from 15.5% to 11.7% but the Li+ gap remains significant (6.7%). The optimal alpha shifts from 0.31 to 0.23, suggesting the relevant mass scale may be the total gravitational mass rather than baryonic mass alone
- **Galaxy morphology**: Disk-dominated (B/T<0.1) galaxies give alpha=0.341≈1/3 with +7.3% improvement over McGaugh; bulge-dominated (B/T≥0.1) give alpha=0.264≈1/4 with only +0.5% improvement (McGaugh already sufficient). This suggests p(M) is primarily a disk-galaxy phenomenon
- **p_max derivation**: The condition for flat rotation curves in d dimensions gives p_flat=(d-2)/(d-1)=0.5 for d=3 (= McGaugh). The data-required p_max=2/3=(d-1)/d could relate to the holographic fraction of surface-to-bulk degrees of freedom, but this remains speculative
- Large-scale structure compatibility requires cosmological extension (not addressed here)
- **This work has not been peer-reviewed**

## Related Work

- Ambjørn, Jurkiewicz & Loll ([2005, PRL 95, 171301](https://doi.org/10.1103/PhysRevLett.95.171301)): Discovered spectral dimension flow from ~2 (UV) to ~4 (IR) in 4D CDT. See Limitations for detailed comparison with SDHG.
- Desmond, Hees & Famaey ([2024, MNRAS 530, 1781](https://doi.org/10.1093/mnras/stae955)): Parametrized the same exponent (as delta/2 in their delta-family) but fit it as a universal constant, not mass-dependent.
- EMOND — Zhao & Famaey (2012): Makes the acceleration scale a0 potential-dependent, not the exponent.
- Superfluid DM — Berezhiani & Khoury ([2015, PRD 92, 103510](https://doi.org/10.1103/PhysRevD.92.103510)): BEC phase transition could provide a physical mechanism for mass-dependent modification.
- [arXiv:2603.23591](https://arxiv.org/abs/2603.23591) (2026): Found that central galaxies in groups/clusters deviate from the standard RAR, with the divergence radius decreasing with host mass — independent evidence for mass-dependent RAR behavior.

## Cross-Validation

Leave-one-out cross-validation within SPARC (171 galaxies, `run_loo_cv.py`):

| Method | RMS | vs McGaugh |
|--------|-----|------------|
| McGaugh (p=0.5) | 0.197 | baseline |
| **p(M) LOO** | **0.186** | **+5.4%** |

The improvement is concentrated in dwarf galaxies (logM < 9: +30%), while massive galaxies show no improvement (-1.5%). This confirms that p(M) generalizes to unseen data and is not overfitting, but the effect is primarily relevant for low-mass systems.

## Disclaimer

This is an independent, exploratory research project by a non-academic individual. It has not been peer-reviewed or published in a scientific journal. Feedback, corrections, and independent verification are welcome via GitHub Issues.

## License

MIT. See [LICENSE](LICENSE).

## Complete Analysis Summary

All quantitative claims in this repository are reproducible from the included scripts. Key results:

| Finding | Value | Script |
|---------|-------|--------|
| Global fit improvement | +15.5% vs McGaugh | `run_global_fit.py` |
| Alpha (free fit) | 0.312 | `run_global_fit.py` |
| Alpha = 1/3 cost | +0.02% RMS | `run_global_fit.py` |
| p_max = 2/3 cost | +0.00% RMS | verified |
| Coefficients A=2, B=3 | 0.5% precision | verified |
| Li+ gap (p(M) vs G(M)) | 11.0% | `run_global_fit.py` |
| dBIC vs McGaugh | 38.4 (very strong) | `run_model_comparison.py` |
| MOND bias-mass r | +0.47 → +0.12 with p(M) | `run_bias_analysis.py` |
| LOO cross-validation | +5.4% | `run_loo_cv.py` |
| Baryonic mass fit | +12.0%, Li+ gap 6.3% | `run_baryonic_mass_fit.py` |
| Gas-dominated alpha | 0.184, +6.8% | verified |
| Disk galaxy alpha | 0.341 ≈ 1/3, +7.3% | verified |
| Bulge galaxy alpha | 0.264 ≈ 1/4, +0.5% | verified |
| CDT gamma (2+1D, 5-seed) | 0.87 ± 0.1 ≈ 1 | `cdt_main.c` |
| CDT d_UV / d_IR (2+1D) | 2.2 / 3.9 | `cdt_main.c` |
| CDT d_UV / d_IR (4D) | 3.7 / 4.3 | `cdt4d.c` |
| 4D CDT (2,4)+(4,2)+(3,3) | d_UV=3.5-3.7, d_IR=4+ | `cdt4d.c` |
| p_flat = (d-2)/(d-1) | 0.5 for d=3 = McGaugh | derived |
| p_max = (d-1)/d | 2/3 for d=3 | data-exact |
