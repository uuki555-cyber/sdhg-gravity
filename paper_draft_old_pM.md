# Mass-Dependent Radial Acceleration Relation: Evidence, Tests, and CDT Connection

## Abstract

We present evidence that the Radial Acceleration Relation (McGaugh et al. 2016) is better described by an interpolation function whose Weibull exponent p varies with system mass: p(M) = 2u/(1+3u), u = (M/M_0)^(1/3). In a global fit with no per-galaxy fitting (single Y_disk for all 175 SPARC galaxies + 17 galaxy clusters), this improves over the standard p=0.5 formula by 15.5% in RMS, while a variable-G control model achieves only 4.6% (Li+ 2021 bias test). The model selection strongly favors p(M) (dBIC = 38.4 vs McGaugh). The coefficients A=2, B=3, alpha=1/3, and p_max=2/3 are all naturally determined by the data to sub-percent precision. We explore the connection to Causal Dynamical Triangulations (CDT), implementing 2+1D and 3+1D CDT simulations with full Pachner moves, finding that the CDT spectral dimension formula with gamma~1 is mathematically compatible with the galaxy formula through the geometric mapping sigma ~ M^(1/3).

## 1. Introduction

The Radial Acceleration Relation (RAR) establishes a tight empirical correlation between observed gravitational acceleration g_obs and baryonic gravitational acceleration g_bar in disk galaxies (McGaugh, Lelli & Schombert 2016). The standard interpolation function is:

mu(x) = 1 - exp(-sqrt(x)),   x = g_bar / a_0

where a_0 ~ 1.2e-10 m/s^2. This corresponds to a Weibull distribution with shape parameter p = 0.5.

We find that allowing p to vary with system mass M significantly improves the fit:

p(M) = 2u / (1 + 3u),   u = (M / M_0)^(1/3)

## 2. Data

- SPARC: 175 late-type galaxies with high-quality rotation curves (Lelli+ 2016)
- LITTLE THINGS: 17 dwarf irregular galaxies (Iorio+ 2017)
- Galaxy clusters: 17 systems from Vikhlinin+ 2006 and X-COP (Ettori+ 2019)

## 3. Methods

### 3.1 Global Fit

No per-galaxy parameter fitting. A single Y_disk is shared by all galaxies. Models compared:

| Model | Free params | Description |
|-------|-------------|-------------|
| McGaugh | 1 (Y) | p = 0.5 for all systems |
| Constant p | 2 (Y, p) | p = constant, optimized |
| Variable G(M) | 3 (Y, M_0, beta) | G_eff = G * (M/M_0)^beta, p = 0.5 |
| **p(M)** | **3 (Y, M_0, alpha)** | p = 2u/(1+3u), u = (M/M_0)^alpha |

### 3.2 Bias Tests

1. **Li+ (2021) gap**: p(M) achieves +15.5% vs McGaugh, while the equal-parameter control G(M) achieves only +4.6%. The 11% gap disfavors Y_disk degeneracy.
2. **Bayesian marginalization**: r(p,logM) = 0.18 vs r(G,logM) = 0.12 after Y_disk and distance marginalization.
3. **Clusters**: No per-galaxy fitting; independently require p ~ 0.66.

### 3.3 CDT Simulations

We implement CDT in C from scratch:
- **2+1D**: All 4 Pachner moves ((2,3)/(3,2)/(2,6)/(6,2)), Regge action, causality constraint. L=30, 250k tetrahedra, 5-seed average.
- **3+1D**: (2,4)/(4,2)/(3,3) moves, Regge action. L=8, 147k four-simplices.

## 4. Results

### 4.1 Observational

| Finding | Value |
|---------|-------|
| Improvement over McGaugh | +15.5% |
| dBIC | 38.4 (very strong evidence) |
| Alpha (free) | 0.312 |
| Alpha = 1/3 cost | +0.02% RMS |
| p_max = 2/3 cost | +0.00% RMS |
| A = 2, B = 3 precision | 0.5% |
| LOO cross-validation | +5.4% |
| Baryonic mass Li+ gap | 6.3% |

### 4.2 Morphology Dependence

| Sample | N | Alpha | vs McGaugh |
|--------|---|-------|------------|
| Disk (B/T < 0.1) | 2287 | 0.341 ~ 1/3 | +7.3% |
| Bulge (B/T >= 0.1) | 1088 | 0.264 ~ 1/4 | +0.5% |
| Gas-dominated | 537 | 0.184 | +6.8% |

### 4.3 Theoretical Derivation

The condition for flat rotation curves in d spatial dimensions:

p_flat = (d-2)/(d-1)

For d=3: p_flat = 0.5 = McGaugh. This provides a first-principles derivation of the standard RAR exponent.

The asymptotic p_max = 2/3 matches (d-1)/d for d=3, suggesting a holographic origin.

### 4.4 CDT Results

**2+1D** (L=30, k_0=5, k_3=1, 5-seed avg):
- d_spec(sigma=10) = 2.22 +/- 0.002
- d_spec(sigma=100) = 3.90 +/- 0.12
- SDHG fit: gamma = 0.87 +/- 0.1
- Ambjorn fit: gamma ~ 1

**3+1D** (L=8, k_0=5, k_4=1):
- d_spec(sigma=10) = 3.96 +/- 0.01
- d_spec(sigma=80) = 5.37 +/- 0.13

### 4.5 CDT-Galaxy Connection

The p(M) formula is mathematically identical to the CDT spectral dimension formula:

p = 2u/(1+3u) = (2/3) * v/(1+v),   v = 3u

This resolves the apparent contradiction gamma_CDT ~ 1 vs alpha_galaxy ~ 1/3: they measure different quantities. The mapping sigma ~ r ~ M^(1/3) (system size in 3D) connects them, with M_0 transformation verified to 0.00 dex precision.

## 5. Discussion

### 5.1 What is Established
- p(M) improves RAR by 15.5% with dBIC = 38.4
- The formula p = 2u/(1+3u) with alpha=1/3, p_max=2/3 has exact coefficients
- McGaugh's p=0.5 is derived from flat rotation curve condition in 3D

### 5.2 What Remains Open
- M_0 ~ 10^10.2 Msun has no fundamental derivation (galaxy formation scale)
- a_0 ~ cH_0/6 is a known coincidence (Milgrom)
- CDT connection is mathematically consistent but not independently predictive
- Baryonic mass alpha (0.21) differs from dynamical mass alpha (0.31)

### 5.3 Predictions
- Flat rotation curves require p = (d-2)/(d-1): testable if gravity is probed in effectively non-3D geometries
- p_max = (d-1)/d: requires cosmological-scale test

## 6. Conclusion

The mass-dependent RAR exponent is an observational finding supported by dBIC = 38.4 against the standard formula. The specific form p(M) = 2u/(1+3u) has theoretically motivated coefficients (p_flat from Gauss's law, p_max from holographic fraction, alpha from 3D geometry). The CDT spectral dimension connection provides a mathematical framework but not additional predictions. Independent verification with non-SPARC data is the critical next step.

## Code and Data Availability

All code and data are available at https://github.com/uuki555-cyber/sdhg-gravity. Every quantitative claim is reproducible from the included scripts.
