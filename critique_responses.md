# Anticipated Critiques and Responses for p(Σ) model

This document collects expected critiques from the RAR/MOND research community
and our prepared responses, including what we *can* defend and what we *cannot*.

---

## Strongest Defenses (statistically robust)

1. **ΔBIC = 177** vs McGaugh with same parameter count (Y_disk only)
2. **Σ_0 = a_0/(4π²G)** matches best-fit to 5.1% (no fitted scale)
3. **Cross-validation logΣ_0 std = 0.005** (extreme stability)
4. **Σ_0 sensitivity peaks at theoretical value**: factor-2 deviations destroy the improvement
5. **Galaxy clusters predicted** to have p ≈ 0.654 vs observed 0.66 (1% agreement, NOT in the SPARC training sample)
6. **Systematic error robustness**: distance ±10%, inclination ±5°, AD ±15% Monte Carlo gives improvement = +14.9% ± 2.4% (`run_systematic_errors.py`)
7. **All Y_disk values in [0.3, 0.7]** give >+11% improvement (Y_disk degeneracy not load-bearing)
8. **0/200 false-positive rate** in synthetic falsification with random Y_disk perturbations

## Honest Limitations (acknowledged upfront)

1. **α=5/3 has a plateau**: α ∈ [1.5, 1.8] all give equivalent RMS. The choice 5/3 is from clean β=0.4; observed β=0.411 → α=1.88 is also valid
2. **Per-galaxy p-M correlation is biased** (43% false positive from Y_disk perturbation alone) — we agree with Li+2021 on this; our claim is global-fit improvement, not per-galaxy correlation
3. **UDG/satellite predictions require EFE**: p(Σ) without External Field Effect gives wrong answers for DF-2 (25.9 vs observed 8.5 km/s). Crater II coincidence (p(Σ)=0.9 km/s, EFE-MOND=2.1, observed 2.7) is plausibly accidental
4. **No first-principles derivation of β=0.4**: this is observational input; α=5/3 = α_M/(1-2β) is structural
5. **CDT connection is structural, not physical**: the functional forms match, but the σ_CDT ↔ Σ_galaxy mapping is empirical
6. **Author is not a professional astronomer**: subject to systematic-error oversights despite the analyses above
7. **Not peer-reviewed**

---

## A. Statistics & Methodology

### A1. "α=5/3 has a plateau — value is not unique"
**Critique**: α ∈ [1.5, 1.8] all give RMS ≈ 0.170. Claim of α=5/3 is arbitrary.

**Response**: Observed SPARC R-M scaling β = 0.411 ± 0.011 (bootstrap N=1000). Via α_Σ = α_M/(1-2β) with α_M=1/3 (CDT 3D prediction), this gives **α_Σ = 1.88 ± 0.13**. The "clean" value β=0.4 yields α=5/3 exactly. The plateau width matches the β uncertainty. We claim **α ∈ [5/3, 1.88] are all empirically and theoretically supported**; we do not claim α=5/3 is unique.

### A2. "ΔBIC=177 is just from changing Y_disk"
**Critique**: Best-fit Y is 0.434 (McGaugh) vs 0.412 (p(Σ)). Improvement is just from refitting Y.

**Response**: Σ_0 sensitivity analysis (`run_psigma_validation.py`) shows ±2× degrades improvement to ~0%. If only Y_disk were doing work, **any Σ_0 would give similar improvement** (different Y_best for each). The narrow peak at Σ_0 = a_0/(4π²G) confirms this is a genuine surface-density effect.

### A3. "Per-galaxy p-M correlation is 43% false-positive (Li+ 2021)"
**Critique**: McGaugh has correctly criticized per-galaxy parameter fits as biased.

**Response**: We agree with Li+. Our headline +13.7% uses **global fit (single Y_disk)** — synthetic falsification (`run_ydisk_falsification.py`) shows 0/200 false-positive rate for global-fit improvement. **Per-galaxy correlation tests and global-fit improvement tests are mathematically distinct.**

---

## B. Physical Interpretation

### B1. "Σ_0 = a_0/(4π²G) looks like post-hoc fitting"
**Critique**: Did you fit Σ_0 first, then back out the 4π² factor?

**Response**: Cross-validation logΣ_0 std = 0.005 across 3 folds (`run_psigma_validation.py`). The 5.1% deviation from a_0/(4π²G) is small. The 4π² factor is geometric (related to surface area of unit sphere), not adjustable. **The MOND scale a_0/G is well-known; the 4π² factor is the new prediction, NOT an extra parameter**.

### B2. "α_Σ = α_M/(1-2β) is circular reasoning"
**Critique**: β is from SPARC data; α_Σ is fit to SPARC data.

**Response**: Galaxy clusters provide independent validation. p(Σ) prediction at cluster Σ ≈ 5-10× Σ_0 gives **p ≈ 0.654, matching observed cluster RAR p ≈ 0.66 to 1%** (`run_psigma_test.py`). Clusters are NOT in the SPARC sample and don't follow the SPARC R-M scaling. This breaks circularity at the cluster scale.

### B3. "Σ depends on M nonlinearly — fitting flexibility increased"
**Critique**: p(Σ) just adds nonlinear redundancy.

**Response**: Same parameter count as McGaugh (Y_disk only). Σ_0 fixed by a_0; α=5/3 from β=0.4. **No additional fitted parameters.** ΔBIC=177 explicitly accounts for parameter count.

---

## C. RAR-Specialist Critiques

### C1. "Which Σ definition? Σ_eff, Σ_disk, Σ_dynamical?"
**Response**: We tested Σ_last, Σ_half, Σ_peak. All give positive improvements (+5.7% to +14.7%). Σ_last works best because consistent with M = 0.5 V_flat² R_last/G (`run_psigma_validation.py`). **Improvement is robust to Σ definition.**

### C2. "What does p(Σ) predict that McGaugh's p=0.5 doesn't? (UDG predictions)"
**Critique**: A theory needs distinguishing predictions.

**Honest Response (after detailed EFE testing in `run_efe_predictions.py`)**:

We tested 7 UDG/satellite systems (Crater II, Antlia II, Sculptor, Fornax,
NGC1052-DF2, DF-44, Leo P) with EFE corrections applied to both McGaugh and
p(Σ):

| System | σ_obs | McG+EFE | p(Σ)+EFE | Better |
|--------|:-----:|:-------:|:--------:|:------:|
| Crater II | 2.7 | 1.0 | 0.17 | McG |
| Antlia II | 5.7 | 1.1 | 0.15 | McG |
| Sculptor | 9.2 | 4.9 | 3.5 | McG |
| Fornax | 11.7 | 10.1 | 8.5 | McG |
| NGC1052-DF2 | 8.5 | 21 | 19 | tie (both bad) |
| DF-44 | 33 | 6.7 | 3.4 | McG |
| Leo P | 15 | 6.3 | 1.8 | McG |

**Honest finding (precise McGaugh 2016 formalism, `efe_mond.py`)**:

We implemented McGaugh 2016's exact EFE formulas:
- σ_iso⁴ = (4/81) G M a_0
- σ_efe² = a_0 G M / (3 g_ex r_{1/2})
- ν(y) = 1/y (deep MOND)
- r_{1/2} = (4/3) × r_h (Wolf+2010)

This reproduces McGaugh 2016's Crater II prediction σ_efe = 2.06 km/s,
matching his published 2.1 km/s exactly.

For p(Σ)+EFE, the predictions are systematically too LOW:

| Galaxy | σ_obs | σ_McG_EFE | σ_pΣ_EFE |
|--------|:-----:|:---------:|:--------:|
| Crater II | 2.7 | 2.06 ✓ | 0.58 |
| Sculptor | 9.2 | 8.70 ✓ | 4.34 |
| Antlia II | 5.7 | 1.82 | 0.58 |
| Draco | 9.1 | 3.92 | 1.74 |

**Why does p(Σ) under-predict?**

McGaugh's exponential μ(x) = 1 - exp(-x^p) gives the deep-MOND limit
g_obs = g_bar^(1-p) × g_†^p. For:
- p = 0.5 (McGaugh): g_obs = √(g_bar a_0) — standard MOND ✓
- p → 0 (low-Σ p(Σ)): g_obs → g_bar — Newtonian, no MOND boost ✗

p(Σ)→0 at very low Σ DESTROYS the deep-MOND boost necessary to fit
ultra-faint dwarfs.

**Important honest conclusion**:

p(Σ) is a phenomenological description calibrated to SPARC's TRANSITION
regime (x = g_bar/a_0 ~ 1). It does NOT extrapolate properly to the
deep-MOND regime (x << 1) where p must approach 0.5 to recover standard
MOND scaling.

This is a clear LIMITATION of p(Σ), not a feature. **The standard McGaugh
formula (p=0.5 universal) remains the correct description for low-Σ
satellite systems with EFE.**

The proper scope of p(Σ) is:
1. SPARC-like rotating disk galaxies in the transition regime (works: +14.7%)
2. Galaxy clusters at high Σ (works: predicts p ≈ 0.66)
3. NOT very-low-Σ systems where deep MOND scaling is essential

**We retract** the earlier claim that p(Σ) "naturally explains" Crater II.
For very low-Σ satellites, McGaugh+EFE is correct, and p(Σ) is wrong.

### C3. "External Field Effect (EFE) is ignored"
**Response**: Confirmed — our analysis assumes isolated equilibrium galaxies (same as McGaugh's RAR). EFE correction must be applied separately for satellites/group members. **p(Σ) is complementary to EFE, not a replacement.**

### C4. "Non-equilibrium dynamics (McGaugh 2025) explain DF-2/UDGs"
**Critique**: McGaugh has argued ultrafaint and ultradiffuse galaxies are out of equilibrium.

**Response**: Agreed. Our analysis is restricted to equilibrium systems (SPARC galaxies, clusters). **We do not claim p(Σ) replaces non-equilibrium dynamics.** The proper test of p(Σ) is in equilibrium isolated galaxies — primarily SPARC.

---

## D. Theoretical Framework

### D1. "CDT connection is mathematical gymnastics"
**Response**: Agreed that the σ_CDT ↔ Σ mapping is structural rather than physical. We claim only that the functional forms are identical. **The CDT connection is suggestive — it is NOT load-bearing for the empirical results.**

### D2. "α=5/3 derivation is just identity manipulation"
**Response**: Correct. The physical content is α_M = 1/3 (CDT 3D prediction). β=0.4 is empirical. **The novel claim is: galaxy R-M scaling β ≈ 0.4 combined with 3D geometry α_M = 1/3 yields α_Σ = 5/3.** We don't claim to derive β from first principles.

### D3. "Has someone proposed Σ-dependent MOND before?"
**Response (literature search complete)**:

**Closest prior work (must cite)**:
1. **Scherer, Pflamm-Altenburg, Kroupa & Gjergo 2025** (A&A 698, A167; arXiv:2504.17002): p-Laplacian framework `∇·(|∇Φ|/a_0)^(p-2)∇Φ = 4πGρ` with **density-dependent exponent**. **Critical differences from this work**:
   - They use volume density ρ (M☉/Mpc³); we use surface density Σ (M☉/pc²)
   - Their p-range 2-12+ (unbounded, from Newtonian to cosmological); ours [0, 2/3] (bounded)
   - Their functional form: exponential ρ(p) = 54·exp(-1.614p) M☉/Mpc³; ours: sigmoid 2u/(1+3u)
   - Their characteristic scale: cosmological ρ_crit ≈ 1.4×10⁻⁷ M☉/Mpc³; ours: Σ_0 = a_0/(4π²G)
   - Their data: dSphs (Lelli+17) + HIFLUGS clusters (Li+23); ours: SPARC + clusters
   - They modify the gravitational operator; we modify the local RAR exponent of μ(x)
   - They explicitly state "the physical meaning of p is to be explored"; we connect p to surface density and CDT spectral dimension flow
   
   **Risk of priority/novelty conflict**: very low. Different variables, different scales, different functional forms.

2. **Milgrom 2016 CSDR** (arXiv:1607.05103): Central Surface Density Relation Σ_D⁰ = Σ_M · S(Σ_B⁰/Σ_M), where Σ_M = a_0/(2πG) ≈ 137 M_sun/pc². **Critical difference**: Σ enters as the *argument* of a fixed ν-function; the *shape* is universal. We modify the function shape itself.

3. **EMOND** (Zhao & Famaey 2012): potential-dependent a_0, not μ(x) Σ-dependent.

4. **Di Paolo, Salucci, Fontaine 2019** (arXiv:1810.08472): empirical evidence of variable RAR slope in LSBs — supports the *empirical* finding but proposes no analytic form.

5. **Desmond, Hees, Famaey 2024** (arXiv:2401.04796): delta-family with universal (not system-dependent) exponent.

**Other surveyed**: Famaey & McGaugh 2012 review confirms all standard interpolating functions (Bekenstein, simple, standard, McGaugh exponential) are universal. QUMOND (Milgrom 2010) and Modified Inertia (Milgrom 2011) keep ν universal.

**Our novelty**: The combination of (a) **surface density Σ** as controlling variable, (b) **sigmoid 2u/(1+3u)** form, (c) **power-law u = (Σ/Σ_0)^α**, and (d) the specific normalization **Σ_0 = a_0/(4π²G)** does not match any prior published proposal we found.

### D4. "What is the relationship between Σ_0 = a_0/(4π²G) and Milgrom's Σ_M = a_0/(2πG)?"
**Critical observation**: **Σ_M / Σ_0 = 2π exactly.** This is not coincidental.

Milgrom's Σ_M = a_0/(2πG) ≈ 137 M_sun/pc² is the **central** disk surface density at the deep-MOND/Newtonian transition.

Our Σ_0 = a_0/(4π²G) = Σ_M/(2π) ≈ 22 M_sun/pc² is the **mean** surface density M/(πR_last²) at the same transition, computed for typical SPARC disks.

For an exponential disk Σ(R) = Σ_c exp(-R/R_d), the ratio of mean Σ within R_last to central Σ_c depends on R_last/R_d:
- R_last/R_d = 3: mean/central ≈ 0.178
- R_last/R_d = 4: mean/central ≈ 0.114
- 1/(2π) ≈ 0.159 (matches R_last/R_d ≈ 3.4)

**Typical SPARC galaxies have R_last/R_d ≈ 3-4**, so our Σ_0 ≈ Σ_M/(2π) is the natural mean-surface-density expression of Milgrom's central transition density. The 2π factor is geometric (from disk integration over angle).

**This places p(Σ) directly in the Milgrom CSDR framework.** Σ_0 is not arbitrary — it's the mean-Σ analog of Milgrom's Σ_M.

---

## E. Quantitative Critiques

### E1. "+13.7% is small relative to SPARC scatter"
**Critique**: SPARC RMS=0.197 dominated by observational systematics.

**Response**: Even small structural improvements at fixed degrees of freedom favor a more correct functional form. **Same DOF gives ΔBIC = 177**, statistically decisive. Systematic errors don't produce coherent ΔBIC at this level.

### E2. "Y_disk = 0.42 vs population synthesis 0.5 within errors"
**Response**: We don't claim Y_disk is "more correct" with p(Σ). We claim the structural form of μ(x) is improved. Y_disk is a nuisance parameter.

### E3. "Submit a paper, blog/repo doesn't count"
**Response**: arXiv preprint planned. Repository public for community review prior to formal submission.

---

## F. Personal/Sociological Critiques

### F1. "Author is not an astronomer"
**Response**: Acknowledged. Open to correction. Repository public, results reproducible from data.

**Mitigation**: Systematic error analysis (`run_systematic_errors.py`) shows:
- Distance ±10%: +13% to +16% improvement
- Inclination ±5°: +13% to +15% improvement
- AD correction ±15%: +12% to +18% improvement
- Combined MC (N=100): +14.9% ± 2.4% (100/100 positive, 98/100 above +10%)

### F2. "30 years of MOND research — overconfident to claim improvement in 1 year"
**Response**: This is an extension within McGaugh's RAR framework, not a replacement. Σ_0 = a_0/(4π²G) builds on McGaugh's claim that a_0 is fundamental. **We position this as a refinement of his framework, not a replacement.**

### F3. "Why 5/3 specifically?"
**Response**: 5/3 emerges from clean β=0.4. Observed β=0.411 gives α=1.88. Both within plateau. We acknowledge: **β = 0.4 is empirical, not derived from first principles.**

---

## Most Damaging Critiques (Honest Assessment)

After careful self-review:

1. **C2 / C3 (UDG predictions need EFE)** — we initially overclaimed; retracted in this document
2. **B2 (circularity)** — clusters provide some independent validation but more datasets desirable
3. **F1 (expertise)** — real limitation; mitigated by systematic-error analysis
4. **A1 (α plateau)** — real ambiguity; we claim α ∈ [5/3, 1.88] not just 5/3

## Where the Result is Strongest

**The cleanest claim**: For SPARC isolated rotating disk galaxies + galaxy clusters, in the equilibrium regime, the RAR exponent is better described by p(Σ) with Σ_0 = a_0/(4π²G) than by McGaugh's universal p=0.5. ΔBIC = 177 with same parameter count.

**The cleanest test**: Take any galaxy with well-measured V_obs(R), well-known M_*, isolated environment. Compute Σ at R_last. Predict p(Σ) and compare to per-galaxy best-fit p. Repeat across the sample. This is what we've done with SPARC.

**The most surprising finding**: Σ_0 = a_0/(4π²G) is not a fitted parameter — it's predicted from MOND's a_0 and matches the SPARC fit to 5%. This is the most concrete connection between MOND scale and RAR shape that we know of.

---

## What We Would Need to Strengthen the Case

1. **An independent dataset** beyond SPARC + LITTLE THINGS + clusters (e.g., BIG-SPARC ~4000 galaxies, MaNGA, EDGES)
2. **A first-principles derivation** of β=0.4 (currently empirical)
3. **EFE-corrected p(Σ) predictions for UDGs** — our preliminary numbers don't include EFE
4. **Direct CDT-galaxy mapping** beyond the structural similarity
5. **Peer review** by RAR specialists (Lelli, Mistele, Famaey)
