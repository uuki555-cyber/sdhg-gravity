# SDHG / p(Σ) Project Status

**Last updated**: 2026-04-24
**Repository**: https://github.com/uuki555-cyber/sdhg-gravity

---

## Bottom line

We propose a surface-density-dependent extension of McGaugh's Radial Acceleration
Relation (RAR):

```
p(Σ) = 2u / (1 + 3u),  u = (Σ/Σ_0)^α
```

with **Σ_0 = a_0/(4π²G) ≈ 22 M_sun/pc²** and **α = 5/3**, where Σ is the
dynamical surface density M_dyn / (πR_last²).

The model uses the **same number of free parameters as McGaugh** (only Y_disk),
yet improves the SPARC global-fit RMS from 0.197 to 0.170 (**ΔBIC = 177** —
decisive). It predicts galaxy clusters at **p ≈ 0.654** without using cluster
data (observed ~0.66, **1% agreement**).

---

## What is verified (high confidence)

### Empirical
- **+13.7% RMS improvement** on SPARC 171 galaxies, same DOF as McGaugh
- **ΔBIC = 177** (>>10 = decisive evidence)
- **Cluster prediction p ≈ 0.654** matches observed 0.66 (1%)
- **3-fold CV stable**: improvement +13.3% ± 3.0%, log Σ_0 std = 0.005
- **LITTLE THINGS independent validation**: +9.9% improvement
- **Σ_0 sensitivity peaked at theoretical value**: factor-2 deviations destroy improvement
- **Monte Carlo robustness**: +14.9% ± 2.4% under combined distance/inclination/AD perturbations
- **Y_disk artifact excluded**: 0/200 false-positive rate in synthetic falsification
- **Quality cuts preserve result**: high-Q SPARC subset gives +8.6%
- **All Y_disk values [0.3, 0.7]** give >+11% improvement
- **Unified corpus β verification** (`run_bigsparc_test.py`): On 403 galaxies
  (SPARC 175 + LITTLE_THINGS 26 + WALLABY 202), β = 0.424 ± 0.010 (bootstrap),
  consistent with SPARC alone β = 0.411 ± 0.011. Validates the α_Σ derivation
  α_Σ = α_M/(1-2β) ≈ 1.9-2.1 across a 2.5× larger galaxy sample.

### Theoretical
- **Σ_0 = a_0/(4π²G) connects to Milgrom (2016) Σ_M = a_0/(2πG)**:
  - Σ_M is central (one-sided thin-disk Gauss law)
  - Σ_0 = Σ_M/(2π) is mean-Σ analog
  - 2π factor is geometric (azimuthal averaging in exponential disks with R_last/R_d ≈ 3.4)
- **α_Σ = α_M/(1-2β) derivation** with α_M = 1/3 (3D scaling) and β = 0.4 (R-M slope) gives α = 5/3
- **Bootstrap β = 0.411 ± 0.011** (SPARC) → α_Σ = 1.88 (within plateau)
- **CDT structural connection**: p(Σ) form ≡ D_s(σ) = a - b/(c + σ^γ) with γ = 1
- **Functional form bounded**: p ∈ [0, 2/3], asymptoting cleanly

### Novelty
- **Distinct from Scherer et al. 2025** (volume density ρ, exponential ρ(p), unbounded p, cosmological scale)
- **Extends Milgrom 2016 CSDR** (function shape Σ-dependent vs. universal)
- **No prior work** found with this exact form (literature search complete)

---

## What is acknowledged as limitation

### Conceptual
1. **α has a plateau**: α ∈ [1.5, 1.8] all give equivalent fits.
   We claim only α = 5/3 from clean β = 0.4; observed β = 0.411 → α = 1.88.

2. **β = 0.4 is empirical**: Not derived from first principles.
   Observed SPARC β = 0.411 ± 0.011.

3. **CDT connection is structural, not physical**: σ_CDT ↔ Σ is mathematical
   identification, not derivation.

### Physical scope
4. **External Field Effect (EFE) regime: p(Σ) under-predicts** (`efe_mond.py`):
   We implemented McGaugh 2016's exact EFE formalism (Eqs. 1-5; ν=1/y in
   deep MOND, σ_iso⁴ = (4/81) GMa_0, σ_efe² = a_0GM/(3 g_ex r_{1/2})).
   This **reproduces McGaugh's Crater II prediction σ = 2.1 km/s exactly**.

   For p(Σ)+EFE, the prediction is much lower:
   | Galaxy | σ_obs | σ_McG_EFE | σ_pΣ_EFE |
   |--------|:-----:|:---------:|:--------:|
   | Crater II | 2.7 | **2.06** ✓ | 0.58 |
   | Sculptor | 9.2 | 8.70 | 4.34 |
   | Antlia II | 5.7 | 1.82 | 0.58 |
   | Draco | 9.1 | 3.92 | 1.74 |

   **Physical interpretation**: In McGaugh's exponential μ(x) = 1 - exp(-x^p),
   the deep-MOND limit is g = g_bar^(1-p) × g_†^p. For p=0.5 this is
   √(g_bar a_0) (standard MOND). For p(Σ)→0 at very low Σ, it becomes
   g ≈ g_bar (Newtonian, no MOND boost). p(Σ) destroys the deep-MOND
   scaling necessary for low-Σ satellites.

   **Conclusion**: p(Σ) is a phenomenological RAR description tuned to
   the transition regime (x ~ 1) where SPARC disk galaxies sit. It does
   **not** extrapolate to the deep-MOND regime where standard MOND
   (p = 0.5) is required. Earlier "p(Σ) explains Crater II" claim was
   overstated. **Retracted.**

5. **The distinguishing prediction is in the TRANSITION regime**: At
   x = g_bar/a_0 ~ 1, μ(x) shape differences matter most. This is exactly
   where typical SPARC disk galaxies sit, which is why ΔBIC = 177 there.
   Clusters are at high Σ (transition complete, both predict p ≈ 2/3).
   Deep-MOND satellites are at low Σ but EFE-dominated (μ → 0, shape
   degenerate). The cleanest test of p(Σ) vs McGaugh would be a truly
   isolated equilibrium galaxy at low Σ — but such systems are rare and
   poorly measured.

6. **Equilibrium assumption**: McGaugh (2025) argues that ultradiffuse and
   ultrafaint galaxies are out of equilibrium. Our analysis is restricted to
   equilibrium SPARC galaxies and clusters.

6. **Dynamical mass M_dyn (not baryonic) for Σ**: Tests with baryonic mass
   give -27% improvement. The "natural" Σ for p(Σ) is the total gravitational
   surface density (dynamical mass proxy), not pure baryonic.
   - **Interpretation**: p depends on the *gravitational regime* of the system,
     not the baryonic distribution per se. This is consistent with MOND's
     transition being a property of the gravitational scale.
   - **Caveat**: This requires careful framing — p(Σ_dyn) is a mixed prediction
     where Σ_dyn comes from observed kinematics (V_flat, R_last) and feeds
     back into the RAR shape.

### Statistical
7. **Per-galaxy correlation tests are biased** (Li+2021 effect, 43% false positive
   rate from Y_disk perturbation). We rely on global-fit improvement only.

8. **Sample size limited**: 17 LITTLE THINGS, 17 clusters. BIG-SPARC (~4000
   galaxies) would be a more powerful test.

### Author/process
9. **Not peer-reviewed**: arXiv preprint planned but not submitted.

10. **Author is not a professional astronomer**: subject to systematic-error
    oversights. Mitigated by Monte Carlo robustness analysis.

---

## Final recommended model: V7 dual-regime

Following further investigation (`run_psigma_advanced.py`, `run_beta_origin.py`),
we recommend the **dual-regime formulation**:

```
p(Σ) = 0.5                        if Σ < Σ_min  (deep MOND regime)
p(Σ) = 2u/(1+3u), u=(Σ/Σ_0)^α     if Σ ≥ Σ_min  (SPARC transition regime)
```

with **Σ_min ≈ 10⁶ M☉/kpc² ≈ 1 M☉/pc²**, **Σ_0 = a_0/(4π²G)**, **α = 5/3**.

**Why dual-regime is cleaner**:
- Below Σ_min: standard MOND deep limit (p=0.5) preserved
- Above Σ_min: SPARC transition physics with p(Σ) variable shape
- Σ_min ≈ minimum Σ in SPARC sample (logΣ ≈ 6.5)
- Acknowledges that SPARC fit is calibrated to its own range; ultra-faint
  dwarfs are extrapolation territory

**Results with V7 dual-regime** (`run_psigma_advanced.py`):
- SPARC global fit: +14.7% (identical to V0; SPARC galaxies all have Σ > Σ_min)
- p(Crater II) = 0.500 ✓ (deep MOND preserved)
- p(cluster) = 0.656 ✓ (observed ~0.66)
- Subset improvements:
  - logM 7-9: +29.4%
  - logM 9-10: +15.9%
  - logM 10-11: +3.8%
  - logM 11-13: -1.3%

This is the recommended formulation for the paper.

## Independent verification with MF96 (2443 galaxies, `run_mf96_test.py`)

We tested the β = R-M relation on the Mathewson & Ford (1996) southern
spirals (Vizier J/ApJS/107/97), 2443 galaxies, completely independent
from SPARC. Result:

| Sample | N | Tracer | β | α_Σ predicted |
|--------|--:|--------|--:|--------------:|
| SPARC | 171 | HI+Hα | 0.411 ± 0.011 | 1.88 |
| LITTLE_THINGS+WALLABY | 228 | HI | 0.424 ± 0.015 | 2.19 |
| **MF96** | **2443** | **Hα optical** | **0.351 ± 0.004** | **1.12** |

The MF96 β is significantly different from the HI-derived samples,
reflecting:

1. **Sample composition**: MF96 is dominated by massive disk galaxies
   (logM 10-12.5), where star-dominated R-M_star slope (~0.31)
   dominates. SPARC has wider mass range with more gas-rich dwarfs.

2. **Methodology**: MF96 uses W20/2 rotation velocity and D0 (optical
   isophotal diameter). SPARC uses V_flat from rotation curve and
   R_last. Different operational definitions of R and V.

3. **Tracer**: Hα probes warm ionized gas, HI probes cold neutral gas.
   The two can give slightly different rotation profiles.

**Honest implication**: β (and thus α_Σ via α = α_M/(1-2β)) is
sample-dependent. There is no universal α_Σ value; it adapts to the
galaxy sample's mass distribution and structural type. This is a
limitation: p(Σ) is a phenomenological fit specific to the calibration
sample.

The headline α = 5/3 corresponds to the SPARC mass distribution.
For a different sample (e.g., MF96 massive spirals), the appropriate
α would be different (~1.1).

## β = 0.4 partial first-principles derivation (`run_beta_origin.py`)

We attempted to derive β = 0.4 from first principles. Results:

| Component | Slope (β) |
|-----------|-----------|
| R-M_star | 0.310 ± 0.016 |
| R-M_dyn | 0.411 ± 0.011 |
| R-M_gas | 0.506 ± 0.014 |

R-M_gas ≈ 0.5 (Tully-Fisher prediction in deep MOND).
R-M_star ≈ 0.31 (size-luminosity relation).
R-M_dyn ≈ 0.4 is the weighted average reflecting galaxy stellar/gas content.

The "**5/12 conjecture**" (β = 5/12 = 0.4167) is consistent with observation
to 1.3%. If R ∝ V^(5/3) and M ∝ V⁴ (Tully-Fisher), then β = 5/12. This is
suggestive but R ∝ V^(5/3) itself remains observational.

By mass range:
- logM 7.5-9 (gas-dom): β = 0.43 (closer to TF)
- logM 9-10: β = 0.42
- logM 10-11: β = 0.36 (star-dominated)
- logM 11-12.5: β = 0.51

**Honest conclusion**: β ≈ 0.4 is **not** derivable from first principles
within standard MOND. It's a weighted average of the stellar (β≈0.31) and
gas (β≈0.51) R-M scaling, reflecting the typical galaxy gas fraction.
For β = 0.4 exactly, the corresponding Σ-M slope is δ ≈ 0.2.

## Modified p(Σ) variants (`run_psigma_modified.py`)

We tested variants that preserve deep MOND (p ≥ 0.5):

| Variant | SPARC RMS | Δ vs McGaugh | Crater II p | Cluster p |
|---------|:---------:|:------------:|:-----------:|:---------:|
| McGaugh (p=0.5) | 0.197 | 0% | 0.5 | 0.5 |
| **V0** (original) | **0.168** | **+14.7%** | **0.01** ✗ | **0.654** ✓ |
| V1: max(0.5, p) | 0.191 | +3.2% | 0.5 ✓ | 0.654 ✓ |
| V2: 0.5 + smooth | 0.203 | -3.0% | 0.5 ✓ | 0.658 ✓ |
| V3: piecewise | 0.191 | +3.0% | 0.5 ✓ | 0.667 ✓ |

**Subset analysis** (logM 7-9, dwarfs):
- V0: +26.6%
- V1: +1.2%

**Critical finding**: V0's SPARC improvement comes primarily from
p < 0.5 prediction at low Σ. This is INCOMPATIBLE with standard MOND
deep-limit g = √(g_bar a_0). V1 preserves standard MOND but loses the
dwarf improvement.

**Cluster prediction is robust across all variants** (p ≈ 0.65-0.67),
confirming the high-Σ tail prediction is independent of the low-Σ
behavior.

**Honest interpretation**: V0 is a phenomenological RAR fit that may
absorb Y_disk or distance systematics in low-mass SPARC galaxies. It
does not preserve standard MOND. V1/V3 are theoretically cleaner
(McGaugh limit preserved) but give only modest SPARC improvement.

**The cleanest claim**: Among p(Σ) variants, only the cluster scale
prediction (Σ >> Σ_0) is unambiguously supported. The dwarf-galaxy
improvement is V0-specific and conflicts with deep MOND at lower
Σ regimes.

## What remains as open questions

1. **Why β ≈ 0.4?** Galaxies are between Tully-Fisher (β=0.5) and 3D
   uniform-density (β=1/3). No first-principles derivation.

2. **Physical mechanism for p(Σ)?** We have only structural connection to CDT.
   No microscopic derivation.

3. **Does p(Σ) work in BIG-SPARC?** Not tested yet (BIG-SPARC unreleased);
   would be a powerful independent test. Unified corpus (438 galaxies)
   confirms β ≈ 0.4 holds across SPARC-style samples.

4. **Why does p(Σ) fail at deep MOND?** Because p(Σ)→0 at very low Σ
   destroys the deep-MOND boost. The improvement on SPARC is in the
   transition regime where this issue doesn't apply. Whether p(Σ) can be
   modified to preserve deep MOND (e.g., p(Σ) = max(0.5, 2u/(1+3u)))
   while maintaining the SPARC improvement is an open question.

5. **Connection to other modified-gravity proposals?** Particularly Scherer
   et al. 2025 (p-Laplacian) — same conceptual neighborhood but distinct
   variables and forms.

6. **What is the proper scope of p(Σ)?** Possibilities:
   - Phenomenological RAR description for SPARC's transition regime (current)
   - Effective description of Y_disk degeneracy in low-mass galaxies
   - Genuine new physics that requires modification at deep MOND limit

---

## Where to be cautious in claims

- **Don't say**: "p(Σ) explains all RAR systems"
  **Do say**: "p(Σ) improves SPARC equilibrium galaxies and clusters"

- **Don't say**: "This proves a_0 is fundamental"
  **Do say**: "This shows the RAR shape is consistent with a_0 controlling a fundamental surface-density scale"

- **Don't say**: "p(Σ) explains Crater II naturally"
  **Do say**: "Crater II requires EFE; p(Σ) without EFE happens to give similar numbers, but this is plausibly coincidence"

- **Don't say**: "This is novel work that supersedes Milgrom"
  **Do say**: "This extends Milgrom's CSDR (2016) by allowing function shape, not just argument, to depend on Σ"

- **Don't say**: "α = 5/3 exactly"
  **Do say**: "α = 5/3 is consistent with observed β = 0.411 and theoretically clean β = 0.4; values 5/3 to 1.88 give equivalent fits"

---

## Files in this repository

### Core science scripts
- `sdhg/`: package (`core.py` for p(M), `data.py` for SPARC/cluster loaders)
- `run_global_fit.py`: McGaugh vs p(M) global comparison
- `run_psigma_test.py`: p(Σ) global fit
- `run_psigma_theory.py`: α candidate exploration, Σ_0 physical scale
- `run_psigma_validation.py`: 2D subsets, LITTLE THINGS, clusters, Σ_0 sensitivity
- `run_alpha_derivation.py`: α_Σ = α_M/(1-2β) derivation
- `run_alpha_theory.py`: alternative α theoretical interpretations
- `run_subset_analysis.py`: fair-comparison subset breakdown
- `run_gas_dominated_test.py`: Y_disk degeneracy tests in gas-pure galaxies
- `run_ydisk_falsification.py`: Monte Carlo synthetic falsification
- `run_systematic_errors.py`: distance/inclination/AD robustness
- `run_robustness_verification.py`: mass definition, quality cuts, β-α
- `run_little_things.py`: LITTLE THINGS independent dwarfs
- `run_loo_cv.py`: leave-one-out cross-validation
- `run_baryonic_mass_fit.py`: baryonic mass variant
- `run_bias_analysis.py`: MOND bias-mass correlation reduction
- `run_slope_test.py`: rotation curve slope vs mass
- `run_main_analysis.py`: per-galaxy detailed analysis
- `run_model_comparison.py`: BIC/AIC formal comparison
- `run_bayesian_test.py`: Bayesian hierarchical Y_disk marginalization
- `analyze_lt_mass.py`: LITTLE THINGS mass-definition checks

### Documentation
- `README.md`: overview, key results, methods summary
- `STATUS.md`: this document — comprehensive status snapshot
- `critique_responses.md`: anticipated critiques with prepared responses
- `paper_draft.md`: working draft for arXiv submission
- `paper_draft_old_pM.md`: archived early p(M) draft
- `make_figures.py`: figure generation

### Data
- `data/sparc_data.mrt`: SPARC 175 galaxies
- `data/little_things/`: LITTLE THINGS dwarf rotation curves
- Cluster data: hardcoded in `sdhg/data.py`

### CDT-related (background, not core to p(Σ) result)
- `cdt_main.c`, `cdt4d.c`: production CDT simulations
- `cdt_pachner.py`, `run_cdt_2plus1d.py`: Python prototypes

---

## Suggested next steps (priority order)

### High priority
1. **Format paper_draft.md as LaTeX, submit to arXiv** — gives proper academic record
2. **Compute EFE-corrected p(Σ) predictions for satellites** (Crater II, Antlia II,
   Milky Way satellites with known EFE)

### Medium priority
3. **Test on BIG-SPARC** (~4000 galaxies; arXiv:2411.13329)
4. **Email Federico Lelli or Tobias Mistele** for technical review (lower-friction
   than McGaugh)
5. **Contact Scherer et al.** to discuss density-dependent MOND framework

### Lower priority
6. **MaNGA / EDGES validation** (different surveys, different methodology)
7. **Tully-Fisher relation prediction from p(Σ)**
8. **Connection to weak-lensing flat rotation (Mistele+ 2024)**

### Investigation
9. **Why β ≈ 0.4?** First-principles derivation attempts
10. **Microscopic mechanism for p(Σ)?** Possible CDT/holographic origin
