# Mass-Dependent Radial Acceleration Relation

**The exponent of the RAR interpolation function depends on baryonic mass.**

This repository presents evidence that the Radial Acceleration Relation (McGaugh, Lelli & Schombert 2016) is better described by an interpolation function whose exponent varies with the baryonic mass of the system:

```
mu(x, M) = 1 - exp(-x^p(M))

p(M) = 2u / (1 + 3u),   u = (M / M0)^(1/3)
```

where `x = g_bar / a0`, `M` is the baryonic mass, and `M0 ~ 10^10.2 solar masses`.

For the standard McGaugh formula, `p = 0.5` for all systems. We find that `p` increases from ~0.2 for dwarf galaxies to ~0.66 for galaxy clusters, unifying their dynamics with a single formula.

![p vs mass](figures/fig2_p_vs_mass.png)

![Global fit comparison](figures/fig3_global_fit.png)

## Key Result

In a **global fit with no per-galaxy fitting** (single Y_disk for all galaxies), comparing models with the same number of free parameters:

| Model | Free params | RMS | vs McGaugh |
|-------|-------------|-----|------------|
| McGaugh (p = 0.5) | 1 | 0.197 | baseline |
| Constant p | 2 | 0.195 | +0.9% |
| Variable G(M) [control] | 3 | 0.188 | +4.6% |
| **p(M) model** | **3** | **0.166** | **+15.5%** |

The p(M) model improves over McGaugh by 15.5%, while a variable-G model (same number of parameters) only achieves 4.6%. The 11% difference rules out Y_disk degeneracy as the source of the signal (see [Li+ 2021 bias test](#bias-test) below).

## Bias Test (Li+ 2021)

Li, Lelli & McGaugh ([2021, A&A 653, A170](https://doi.org/10.1051/0004-6361/202040101)) showed that fitting a parameter per galaxy can create spurious mass dependence due to degeneracy with the stellar mass-to-light ratio Y_disk. We address this in three ways:

1. **Global fit (`run_global_fit.py`)**: No per-galaxy fitting at all. Single Y_disk shared by all galaxies. p(M) still improves by 15.5%, while the control model (variable G with same degrees of freedom) achieves only 4.6%. **The 11% gap cannot be an artifact.**

2. **Bayesian marginalization (`run_bayesian_test.py`)**: Marginalizing over Y_disk ~ N(0.5, 0.15) and distance ~ N(1.0, 0.10), the per-galaxy p correlation with mass (r=0.18) exceeds the G_eff control (r=0.12). Suggestive but not definitive from galaxy data alone.

3. **Galaxy clusters are immune**: Cluster data involves no per-galaxy fitting and independently requires p ~ 0.66, far from the galaxy-optimal p ~ 0.5.

## Quick Start

```bash
pip install numpy scipy
python run_global_fit.py            # Core result (no per-galaxy fitting, bias-free)
python run_main_analysis.py         # Detailed analysis (per-galaxy Y_disk, subject to Li+ caveat)
python run_little_things.py         # Independent validation on dwarf galaxies
python run_bayesian_test.py         # Li+ (2021) methodology check
```

## Data

- `data/sparc_data.mrt`: SPARC mass models for 175 disk galaxies (Lelli, McGaugh & Schombert 2016, AJ 152, 157)
- `data/little_things/finalrot/`: Rotation curves for 17 dwarf irregular galaxies (Iorio+ 2017, MNRAS 466, 4159)
- Galaxy cluster data from Vikhlinin+ 2006 (ApJ 640, 691) and X-COP/Ettori+ 2019 (A&A 621, A39), hardcoded in `sdhg/data.py`

## What This Is and Is Not

**This is**: An observational finding that the RAR interpolation exponent correlates with system mass, tested against methodological bias concerns.

**This is not**: A new theory of gravity, a claim about dark matter, or a peer-reviewed result.

## Limitations

- **Cross-validation is neutral**: Applying SPARC-trained parameters to LITTLE THINGS gives -0.8% (no improvement). This may be due to crude enclosed-mass estimates for dwarfs, but it means we cannot yet confirm that p(M) generalizes to unseen data
- M0 is uncertain by a factor of ~5 (10^10.0 to 10^10.8), depending on fitting method and cluster weighting
- The functional form p(M) = 2u/(1+3u) is empirical; theoretical derivation is speculative
- Large-scale structure compatibility requires cosmological extension (not addressed here)
- **This work has not been peer-reviewed**

## Related Work

- Desmond & Famaey ([2024, MNRAS 530, 1781](https://doi.org/10.1093/mnras/stae713)): Parametrized the same exponent (as delta/2 in their delta-family) but fit it as a universal constant, not mass-dependent.
- EMOND — Zhao & Famaey (2012): Makes the acceleration scale a0 potential-dependent, not the exponent.
- Superfluid DM — Berezhiani & Khoury ([2015, PRD 92, 103510](https://doi.org/10.1103/PhysRevD.92.103510)): BEC phase transition could provide a physical mechanism for mass-dependent modification.

## License

MIT. See [LICENSE](LICENSE).

## Disclaimer

This is an independent, exploratory research project by a non-academic individual. It has not been peer-reviewed or published in a scientific journal. Feedback, corrections, and independent verification are welcome via GitHub Issues.
