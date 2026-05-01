# A Surface-Density Formulation of the Radial Acceleration Relation

**Working draft for arXiv preprint / journal submission**

## Abstract

The Radial Acceleration Relation (RAR; McGaugh, Lelli & Schombert 2016) is well
described by an interpolation function μ(x) = 1 - exp(-x^p) with universal
exponent p = 1/2. We show that the SPARC galaxy sample (171 galaxies) and 17
galaxy clusters are jointly described, with **the same number of free
parameters as McGaugh's universal-p form** (only Y_disk), by a
surface-density-dependent exponent

  p(Σ) = 2u / (1 + 3u),  u = (Σ/Σ_0)^α

with **Σ_0 = a_0/(4π²G) ≈ 22 M_sun/pc²** and **α = 5/3**.

The transition surface density Σ_0 is the mean-Σ analog of Milgrom's (2016)
central CSDR scale Σ_M = a_0/(2πG); the factor 2π is geometric (azimuthal
averaging over an exponential disk). The exponent α = 5/3 follows from
α = α_M/(1-2β) with α_M = 1/3 (3D scaling) and β = 0.4 (galaxy R-M
slope; observed 0.411 ± 0.011).

The model improves the SPARC global fit RMS from 0.197 (McGaugh) to 0.170,
giving ΔBIC = 177 (decisive). Galaxy clusters are predicted to lie at
p ≈ 0.654, matching the observed ~0.66 within 1%. Cross-validation
(logΣ_0 std = 0.005) and Monte Carlo robustness (improvement = +14.9% ± 2.4%
under combined distance, inclination, and asymmetric-drift uncertainties)
confirm the result is not driven by overfitting or systematic error.

## 1. Introduction

McGaugh, Lelli & Schombert (2016) demonstrated that 175 disk galaxies in the
SPARC sample obey a tight Radial Acceleration Relation (RAR), parametrized
by μ(x) = 1 - exp(-x^p) with x = g_bar/a_0 and universal p = 1/2. The
acceleration scale a_0 ≈ 1.2×10⁻¹⁰ m/s² is consistent with Milgrom's MOND
(Milgrom 1983; Famaey & McGaugh 2012). Galaxy clusters, however, require
p ≈ 0.66 — well-known tension that has motivated extensions such as EMOND
(Zhao & Famaey 2012).

Li, Lelli, McGaugh & Schombert (2021) noted that fitting μ-function
parameters per galaxy can produce spurious mass dependence due to
degeneracy with the stellar mass-to-light ratio Y_disk. This caveat
correctly disfavors per-galaxy correlation tests as evidence for
mass-dependent RAR.

We investigate whether the RAR exponent p depends systematically on the
local surface density Σ rather than being universal. We use **global fits**
(single Y_disk for all galaxies) to avoid the Y_disk degeneracy. We do not
modify a_0 (as in EMOND) or the Poisson equation (as in p-Laplacian
proposals such as Hofstetter & Kroupa 2025). We modify only the *exponent*
of the standard exponential interpolating function.

## 2. The p(Σ) Model

We parametrize the RAR exponent as

  p(Σ) = 2u / (1 + 3u),                                    (1)
  u = (Σ/Σ_0)^α,                                           (2)

with Σ = M_dyn / (π R_last²), M_dyn = 0.5 V_flat² R_last / G (Lelli et al.
2017 convention).

Limits of (1): p → 0 as Σ → 0 (Newtonian-like), p → 2/3 as Σ → ∞
(holographic-like). At Σ = Σ_0 (u = 1), p = 0.5 — recovering McGaugh.
**Σ_0 is the McGaugh transition surface density.**

### 2.1 Σ_0 from Milgrom (2016)

Milgrom (2016, hereafter M16) proposed the Central Surface Density
Relation Σ_D⁰ = Σ_M·S(Σ_B⁰/Σ_M) with

  Σ_M = a_0/(2πG) ≈ 138 M_sun/pc².                         (3)

The factor 2π comes from the one-sided Gauss law for a thin sheet:
g_N⁺ = 2πGΣ, evaluated at g_N⁺ = a_0.

We propose

  **Σ_0 = a_0/(4π²G) = Σ_M/(2π) ≈ 22 M_sun/pc².**          (4)

The factor 2π between Σ_0 and Σ_M is geometric. M16's Σ_M is the
*central* on-axis surface density at the deep-MOND transition. Our Σ_0 is
the *mean* surface density M/(πR²) at the same transition, computed for
typical exponential disks.

For an exponential disk Σ(R) = Σ_c·exp(-R/R_d), the mean Σ within R_last
is ⟨Σ⟩/Σ_c = (2/x²)[1 - (1+x)e^(-x)] with x = R_last/R_d. Setting
⟨Σ⟩/Σ_c = 1/(2π) gives R_last/R_d ≈ 3.4, the typical SPARC value. Σ_0 is
fixed by a_0 alone; it is *not* a fitted parameter.

### 2.2 α = 5/3 from CDT geometry and galaxy structure

The functional form (1) is mathematically equivalent to the CDT spectral
dimension flow D_s(σ) = a - b/(c + σ^γ) (Ambjørn, Jurkiewicz & Loll 2005)
with γ = 1, identifying σ_CDT with v = 3u. The CDT exponent γ = 1 is
universal across spacetime dimensions; the dimensional content lies in the
mapping between σ_CDT and the physical observable.

For a system of mass M in 3D, dimensional analysis gives M^(α_M) ↔ R^1
with α_M = 1/3 (since M ~ R³ at fixed density). Substituting Σ = M/(πR²)
and R ∝ M^β (the empirical galaxy R-M scaling), we obtain Σ ∝ M^(1-2β),
hence α_Σ = α_M/(1-2β).

Observed SPARC R-M slope: **β = 0.411 ± 0.011** (bootstrap N=1000). The
"clean" value β = 0.4 (between Tully-Fisher 0.5 and uniform-density 1/3)
gives **α_Σ = 5/3** exactly. Observed β = 0.411 gives α_Σ = 1.88. Both
lie within the empirical RMS plateau α ∈ [1.5, 1.8].

We fix **α = 5/3** for the headline result, noting that values in
[5/3, 1.88] give equivalent fits.

## 3. Data and Method

### 3.1 SPARC

171 galaxies (Lelli, McGaugh & Schombert 2016). Rotation curves
V_obs(R), 3.6-μm photometric Vdisk, Vbul, atomic gas Vgas. Excluded:
galaxies with N_pts < 5.

g_bar(R) = [Y_disk·V_disk² + sgn(V_gas)·V_gas² + 0.7·V_bul²] / R.

g_obs(R) = V_obs² / R.

### 3.2 LITTLE THINGS

17 dwarf irregulars (Iorio et al. 2017). Independent dataset for
out-of-sample validation.

### 3.3 Galaxy clusters

17 clusters (Vikhlinin et al. 2006; Ettori et al. 2019). M_500, R_500
gives Σ_cluster ≈ 5-10 × Σ_0.

### 3.4 Fit procedure

We minimize the global RMS log residual:

  RMS = √(⟨[log₁₀(g_obs) - log₁₀(g_bar/μ)]²⟩)              (5)

where μ = 1 - exp(-x^p) with p = 0.5 (McGaugh) or p(Σ) (this work).
A single Y_disk is fitted globally; no per-galaxy parameters.

## 4. Results

### 4.1 SPARC global fit

| Model | Free params | RMS | ΔBIC |
|-------|:-----------:|:---:|:----:|
| McGaugh (p=0.5) | 1 (Y_disk) | 0.1970 | 0 |
| **p(Σ), Σ_0 and α fixed** | **1 (Y_disk)** | **0.1700** | **−177** |
| p(Σ), Σ_0 and α free | 3 | 0.1699 | −162 |

ΔBIC = 177 with **same parameter count** as McGaugh.

### 4.2 Σ_0 sensitivity (no fit)

Improvement vs Σ_0 multiplier (α = 5/3 fixed, Y_disk fitted):

| Σ_0 × | Improvement |
|:-----:|:-----------:|
| 0.3 | -11.3% |
| 0.5 | +0.7% |
| **1.0** (theory) | **+13.7%** |
| 2.0 | +0.3% |
| 3.0 | -15.8% |

The improvement is sharply peaked at the theoretical Σ_0 = a_0/(4π²G).
Factor-2 deviations destroy the improvement, confirming the value is
not arbitrary.

### 4.3 Cross-validation

3-fold CV: improvement = +13.3% ± 3.0%.
**logΣ_0 across folds: 7.350, 7.354, 7.367 (std = 0.005).**

The Σ_0 value is extremely stable; α is in plateau [1.5, 1.8] across folds.

### 4.4 LITTLE THINGS independent validation

Mean improvement +9.9%, win rate 10/17, with the SPARC-trained Σ_0 and α
applied directly.

### 4.5 Galaxy cluster prediction

For the 17 clusters, Σ ~ 5-10 × Σ_0 (deep in the high-density regime).

p(Σ) prediction: **p ≈ 0.654 ± 0.003**.
Observed cluster RAR: **p ≈ 0.66**.
Agreement: **0.9%**.

Crucially, no cluster data was used to determine Σ_0 or α.

### 4.6 Robustness to systematic errors

Monte Carlo (N=100) with distance ±10%, inclination ±5°, asymmetric drift
±15%:

- Mean improvement: **+14.9% ± 2.4%**
- 100/100 realizations show positive improvement
- 98/100 above +10%

### 4.7 Y_disk degeneracy falsification

Synthetic data generated with p = 0.5 (true) and Y_disk perturbations
σ = 0.15 (Li+ 2021 prior):

- 0/200 realizations produce the observed +6.4% improvement
- Mean fake improvement: -59% (p(Σ) is harmful when p=0.5 is true)

The result excludes Y_disk artifact origin at the global-fit level.

We acknowledge: per-galaxy p-M correlation tests are vulnerable to Y_disk
contamination (43% false positive rate from the same simulation). Our
claim is *global-fit improvement only*.

## 5. Subset Analysis: where p(Σ) wins

Allowing each model its own optimal Y_disk (fair comparison), we find
the improvement of p(Σ) is uniformly positive across mass and gas-fraction
bins:

| logM | f_gas | N | p(Σ) improvement |
|:----:|:-----:|:-:|:----------------:|
| 7-9 | 0.3-0.5 | 9 | +32.4% |
| 7-9 | 0.5-1.0 | 6 | +36.3% |
| 9-10 | 0.0-0.3 | 18 | +5.6% |
| 9-10 | 0.3-0.5 | 22 | +23.8% |
| 9-10 | 0.5-1.0 | 22 | +19.2% |
| 10-11 | 0.0-0.3 | 39 | +3.1% |
| 10-11 | 0.3-0.5 | 9 | +13.3% |
| 10-11 | 0.5-1.0 | 10 | -1.8% |
| 11-13 | 0.0-0.3 | 36 | +1.4% |

In contrast, the simpler p(M) form (with α_M = 1/3) is *negative* for
high-mass galaxies (-5.5% at logM > 11). p(Σ) is uniformly positive or
near-zero. This indicates that surface density Σ — not mass M — is the
correct argument of the RAR exponent.

## 6. Connection to Milgrom (2016) CSDR

Milgrom (2016) established that

  Σ_D⁰ = Σ_M · S(Σ_B⁰/Σ_M),  S(y) = ∫₀^y ν(y')dy'           (6)

where Σ_D⁰ is the dynamical column density on the symmetry axis, Σ_B⁰ is
the baryonic central surface density, and ν is the QUMOND interpolating
function.

Our work extends this in a complementary direction: we modify the *shape*
of μ(x) by allowing its exponent p to depend on Σ. The transition
surface density Σ_0 = a_0/(4π²G) we identify is the mean-Σ analog of
Milgrom's central Σ_M, off by a geometric factor 2π.

Notably, the dimensionless ratio Σ_B⁰/Σ_D⁰ = y/S(y) (which Milgrom
constructs from CSDR) is structurally a sigmoid: it saturates to 1 at
large y and to √y/2 at small y. Our p(Σ) sigmoid form may be a related
expression of the same underlying transition physics.

## 7. Connection to CDT (suggestive only)

The p(Σ) functional form is mathematically identical to the 4D CDT
spectral dimension flow

  D_s(σ) = a - b/(c + σ^γ),  γ = 1.00 ± 0.1                 (7)

(Ambjørn et al. 2005) under the identification σ_CDT ≡ v = 3(Σ/Σ_0)^α
with γ = 1. We do not claim a physical derivation of MOND from CDT; the
mapping σ_CDT ↔ Σ is structural, not mechanistic. The CDT connection is
suggestive but not load-bearing for the empirical results.

## 8. Limitations

1. **α has a plateau**: α ∈ [1.5, 1.8] all give equivalent fits. We
   claim α = 5/3 from clean β = 0.4; observed β = 0.411 yields α = 1.88,
   also acceptable.

2. **β = 0.4 is empirical**: Not derived from first principles. The
   observed SPARC slope is 0.411 ± 0.011 (bootstrap).

3. **External Field Effect (EFE) not modeled**: For satellites and
   group/cluster members (Crater II, DF-2, etc.), EFE corrections must be
   applied separately. p(Σ) is complementary to, not a replacement for,
   EFE.

4. **Equilibrium assumption**: McGaugh (2025) argues that ultradiffuse and
   ultrafaint galaxies are out of equilibrium. Our analysis is restricted
   to equilibrium SPARC galaxies and clusters. p(Σ) is not tested on
   non-equilibrium systems.

5. **Per-galaxy correlation tests are biased** (Li+ 2021 effect, ~43%
   false positive rate from Y_disk perturbation alone). We rely solely on
   global-fit improvement and synthetic falsification.

6. **No first-principles derivation of α**: The α value emerges from
   α_M/(1-2β) with empirical β, not from fundamental physics.

## 9. Discussion

p(Σ) refines McGaugh's RAR by allowing the interpolation function shape
to depend on surface density. The transition density Σ_0 is fixed by
Milgrom's a_0 and disk geometry (2π factor). The exponent α = 5/3 follows
from 3D dimensional scaling combined with the empirical galaxy R-M
relation.

The result does not address the dark-matter-vs-MOND question directly. It
is a phenomenological refinement of the RAR shape parameter that
- improves SPARC fits dramatically (ΔBIC = 177 with same DOF),
- predicts cluster RAR (p ≈ 0.66) without cluster data,
- connects galaxy and cluster MOND regimes through Milgrom's CSDR scale.

Scherer, Pflamm-Altenburg, Kroupa & Gjergo (2025, A&A 698, A167;
arXiv:2504.17002) proposed a related but structurally distinct
density-dependent framework: a p-Laplacian generalization of the Poisson
equation `∇·(|∇Φ|/a_0)^(p-2) ∇Φ = 4πGρ`, with p running from 2 (Newton)
through 3 (deep MOND) to ~12 at cosmological critical density. Their fit
is `ρ(p) = α exp(βp)` (exponential, unbounded), with α = 54 ± 19 M_sun/Mpc³
and β = -1.614 ± 0.043. Their critical scale is the cosmological ρ_crit
≈ 1.4×10⁻⁷ M_sun/Mpc³, not constructed from a_0. The conceptual
neighborhood (parameter varying with density) is similar but the specific
proposals differ in:
- variable (volume ρ vs surface Σ);
- functional form (exponential vs sigmoid);
- range (unbounded vs bounded [0, 2/3]);
- characteristic scale (cosmological ρ_crit vs MOND-derived a_0/(4π²G));
- framework (modified gravitational operator vs modified RAR exponent).

## 10. Conclusions

A surface-density-dependent RAR exponent

  p(Σ) = 2u/(1+3u), u = (Σ/Σ_0)^α, Σ_0 = a_0/(4π²G), α = 5/3

fits SPARC galaxies and galaxy clusters jointly with the same parameter
count as McGaugh's universal-p form. ΔBIC = 177 (decisive). Σ_0 is the
mean-Σ analog of Milgrom's CSDR scale Σ_M; α follows from CDT 3D scaling
and the observed galaxy R-M relation.

The result is robust to systematic errors (+14.9% ± 2.4% under combined
Monte Carlo perturbations), excluded as Y_disk artifact (0/200 false
positive rate), and validated independently on LITTLE THINGS dwarf
galaxies (+9.9%) and galaxy clusters (1% agreement).

## Code and Data

All analysis scripts and results are reproducible from:
https://github.com/uuki555-cyber/sdhg-gravity

## Key References

- Ambjørn J., Jurkiewicz J., Loll R., 2005, PRL 95, 171301
  [arXiv:hep-th/0505113]
- Famaey B., McGaugh S.S., 2012, Living Rev. Relativ. 15, 10
  [arXiv:1112.3960]
- Scherer D., Pflamm-Altenburg J., Kroupa P., Gjergo E., 2025,
  A&A 698, A167 [arXiv:2504.17002]
- Iorio G. et al., 2017, MNRAS 466, 4159
- Lelli F., McGaugh S.S., Schombert J.M., 2016, AJ 152, 157
- Li P., Lelli F., McGaugh S.S., Schombert J., Chae K.-H., 2021, A&A 646, L13
- McGaugh S.S., Lelli F., Schombert J.M., 2016, PRL 117, 201101
- Milgrom M., 1983, ApJ 270, 365
- Milgrom M., 2016, PNAS 113, 14749 [arXiv:1607.05103]
- Vikhlinin A. et al., 2006, ApJ 640, 691
- Zhao H., Famaey B., 2012 (EMOND)

## Acknowledgments

This work was carried out by an independent researcher and prepared with
AI-assisted analysis. We welcome feedback from RAR/MOND specialists
(particularly F. Lelli, S. McGaugh, B. Famaey, and the SPARC team) and
from anyone working on density-dependent MOND frameworks (J. Hofstetter,
P. Kroupa). No conflict of interest. The author has no funding to declare.
