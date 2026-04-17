# SDHG: Mass-Dependent Radial Acceleration Relation

**The exponent of the RAR interpolation function depends on galaxy mass.**

This repository presents evidence that the Radial Acceleration Relation (McGaugh+ 2016) is better described by an interpolation function whose exponent varies with the baryonic mass of the system:

```
mu(x, M) = 1 - exp(-x^p(M))

p(M) = 2u / (1 + 3u),   u = (M / M0)^(1/3)
```

where `x = g_bar / a0`, `M` is the baryonic mass, and `M0 ~ 10^10.75 solar masses`.

## Key Finding

| System | Mass (Msun) | p | Effect |
|--------|-------------|---|--------|
| Ultra-dwarf galaxies | 10^7 | ~0.2 | Weak modification |
| Milky Way-class | 10^10 | ~0.5 | McGaugh-like |
| Galaxy clusters | 10^14 | ~0.66 | Strong modification |

The mass dependence of `p` unifies the galaxy-scale RAR with galaxy cluster dynamics using a single formula.

## Results

- **SPARC 175 galaxies + 17 clusters**: 25% improvement over McGaugh+ 2016 (unified RMS)
- **LITTLE THINGS 17 dwarfs**: Independent validation, 27% improvement
- **Ultra-dwarf prediction confirmed**: p < 0.25 for M < 10^8 (observed p = 0.17)
- **Cluster ratio**: 0.95 +/- 0.09 (1.0 = perfect; McGaugh gives 0.65)
- **Bayesian bias test**: Marginalizing over Y_disk and distance with Gaussian priors, p(M) remains suggestive (r=0.18) while G_eff control gives r=0.12. Galaxy-cluster difference is robust.

## Quick Start

```bash
pip install numpy scipy
python run_main_analysis.py        # Main analysis (SPARC + clusters)
python run_little_things.py        # Independent validation (dwarf galaxies)
python run_bayesian_test.py        # Bias test (Li+ 2021 methodology check)
```

## Data

- `data/sparc_data.mrt`: SPARC mass models (Lelli, McGaugh & Schombert 2016, AJ 152, 157)
- `data/little_things/finalrot/`: LITTLE THINGS rotation curves (Iorio+ 2017, MNRAS 466, 4159)
- Cluster data from Vikhlinin+ 2006 (ApJ 640, 691) and X-COP/Ettori+ 2019 (A&A 621, A39)

## Interpretation

The exponent `p(M)` may reflect:
- **CDT dimensional flow**: The effective spatial dimension runs from ~1 (small systems) to ~3 (large systems), with `p = (d_eff - 1) / d_eff`
- **Superfluid dark matter**: BEC phase transition temperature depends on halo mass
- **Scale-dependent holographic encoding**: Information transmission efficiency varies with system mass

We do not claim to identify the physical mechanism. We report the observational fact that `p` depends on `M`.

## Limitations and Caveats

- **Li+ (2021) bias concern**: Fitting a parameter per galaxy risks spurious mass dependence due to Y_disk degeneracy. Our Bayesian test (`run_bayesian_test.py`) shows the galaxy-only signal is suggestive but not definitive (r=0.18 vs control r=0.12). The galaxy-to-cluster difference is robust.
- `M0` is empirically determined (10^10.0 - 10^10.8 range); not yet derived from fundamental constants
- Intrinsic scatter in per-galaxy `p` is ~0.13 (68% explained by M, 32% unexplained)
- Large-scale structure (power spectrum shape) requires cosmological extension
- This work has **not been peer-reviewed**

## Reproducibility

All results can be reproduced by running the Python scripts. The only dependencies are `numpy` and `scipy`. No proprietary data is used — SPARC data is from the public database at Case Western Reserve University.

## License

MIT. See [LICENSE](LICENSE).

## Disclaimer

This is an exploratory, independent research project. It has not been peer-reviewed or published in a scientific journal. The results should be treated as preliminary until verified by the community. Feedback, corrections, and independent verification are welcome.
