# Email Drafts to Specialists

These are prepared drafts for emailing RAR/MOND specialists about p(Σ) findings.
**Not yet sent.** Decide on timing/tone before sending.

---

## Draft 1: Federico Lelli (SPARC creator, observational expert)

**To**: federico.lelli@inaf.it (or current address)
**Cc**: stacy.mcgaugh@case.edu (optional)
**Subject**: A surface-density-dependent extension of the McGaugh RAR — feedback welcome

Dear Dr. Lelli,

I am an independent researcher who has been studying the Radial Acceleration Relation (RAR) using the SPARC sample. I have a result that builds on McGaugh (2016) and Milgrom (2016), and I would value your technical assessment before submitting to arXiv.

**Key finding**: A surface-density-dependent exponent

  p(Σ) = 2u/(1+3u),  u = (Σ/Σ_0)^α

with **Σ_0 = a_0/(4π²G) ≈ 22 M_sun/pc²** and **α = 5/3**, fits SPARC 171 galaxies and 17 galaxy clusters jointly with **the same number of free parameters as McGaugh's universal-p form** (only Y_disk).

**Results**:
- SPARC global-fit RMS: 0.197 → 0.170 (+13.7% improvement, ΔBIC = 177)
- Galaxy cluster prediction: p ≈ 0.654 (observed ~0.66, agreement 1%)
- 3-fold cross-validation: +13.3% ± 3.0%, log Σ_0 stable to 0.005

**Connection to your CSDR work**:
Σ_0 = a_0/(4π²G) is exactly Σ_M/(2π) where Σ_M = a_0/(2πG) is Milgrom's (2016) central CSDR scale. The factor 2π is geometric (mean Σ vs central Σ for typical exponential disks with R_last/R_d ≈ 3.4). p(Σ) extends Milgrom's CSDR by allowing the function shape, not just argument, to depend on Σ.

**Honest limitations** (already documented):
- The dwarf-galaxy improvement comes from p < 0.5 prediction, which conflicts with deep MOND scaling for ultra-faint dwarfs (Crater II, etc. — there McGaugh's standard formula is correct, p(Σ) under-predicts).
- α has a plateau α ∈ [5/3, 1.88] within observational error of β = R-M scaling.
- The CDT structural connection is suggestive but not load-bearing.

Repository: https://github.com/uuki555-cyber/sdhg-gravity
Paper draft: paper_draft.md (under preparation for arXiv)

I make no claim of priority. The work is offered freely for use and critique. If any aspect is useful for the SPARC team's ongoing work or BIG-SPARC release, please use it without attribution constraints.

Specifically, I would value your judgment on:
1. Is the Σ_0 = a_0/(4π²G) connection to your CSDR work physically meaningful, or coincidental?
2. Are there obvious systematic effects in SPARC dwarfs that could explain the +13.7% improvement?
3. Is the data-quality cut N_pts ≥ 5 appropriate, or would you recommend stricter cuts?

I appreciate any time you can spare. The repository is reproducible from the data; all scripts are in Python with comments.

Best regards,
[Your name]

---

## Draft 2: Tobias Mistele (younger theoretical collaborator)

**To**: t.mistele@gmail.com (or current institutional address)
**Subject**: A Σ-dependent RAR exponent — connection to weak-lensing flat curves?

Dear Dr. Mistele,

I am an independent researcher exploring extensions to the Radial Acceleration Relation. I noticed your recent work on "infinitely flat" rotation curves from weak lensing (Mistele et al. 2024, arXiv:2406.09685) and your cluster mass-modeling work, which is potentially relevant to a finding I would like to share.

**Brief**: A surface-density-dependent RAR exponent

  p(Σ) = 2u/(1+3u), u = (Σ/Σ_0)^α
  Σ_0 = a_0/(4π²G), α = 5/3

improves SPARC RMS by 13.7% (ΔBIC = 177) with the same parameter count as McGaugh's universal-p form, and predicts cluster RAR p ≈ 0.654 (observed ~0.66, 1% agreement) without using cluster data.

**Why I think you might be interested**:
1. Your weak-lensing flat curves probe Σ at very large radii, where p(Σ) makes specific predictions
2. The cluster-prediction success of p(Σ) (which you have worked on) is a key result
3. The CDT structural connection (p(Σ) form ≡ CDT spectral dimension flow with γ=1) might appeal to your theoretical interests
4. Σ_0 = a_0/(4π²G) is the mean-Σ analog of Milgrom's central CSDR scale — possibly a clean unification of MOND scales

Repository: https://github.com/uuki555-cyber/sdhg-gravity
Paper draft: paper_draft.md

The work is offered freely for use, critique, or collaboration. I make no priority claims.

Specific questions for you:
1. Does your weak-lensing data at large radii show evidence for varying μ-shape with Σ?
2. Is the Σ_0 = a_0/(4π²G) connection compatible with your cluster mass models (arXiv:2506.13716)?
3. Would the modified gravity community find the structural CDT-RAR mapping interesting, even if non-mechanistic?

I would welcome any feedback. The analysis is fully open-source and reproducible.

Best regards,
[Your name]

---

## Draft 3: Stacy McGaugh (RAR originator)

**To**: stacy.mcgaugh@case.edu
**Subject**: Σ-dependent RAR exponent — refinement within your framework

Dear Prof. McGaugh,

I am an independent researcher who has been working on a phenomenological extension of your Radial Acceleration Relation. I have a result I would like to share, primarily because it builds on your insistence that a_0 is fundamental.

**Key finding**: The RAR interpolating function exponent p in
μ(x) = 1 - exp(-x^p) appears to depend on local surface density Σ:

  p(Σ) = 2u/(1+3u),  u = (Σ/Σ_0)^α

with **Σ_0 = a_0/(4π²G) ≈ 22 M_sun/pc²** (= Σ_M/(2π) where Σ_M is Milgrom's 2016 CSDR scale) and **α = 5/3**.

**Results**:
- SPARC global fit (171 galaxies, single Y_disk): RMS 0.197 → 0.170
- Same parameter count as your universal-p formula
- ΔBIC = 177 (decisive evidence)
- Galaxy clusters: predicted p ≈ 0.654 vs observed 0.66 (1% agreement)
- 3-fold cross-validation stable

**Honest disclosure**:
- p(Σ) calibrated to SPARC's transition regime (x ~ 1)
- For very low-Σ satellites (Crater II, dSphs in deep MOND), your standard formula with EFE remains correct; p(Σ) under-predicts there
- I implemented your 2016 Crater II prediction exactly (σ = 2.06 km/s ✓) and confirmed p(Σ) does NOT supersede it for dSphs

**Position**: This is offered as a refinement WITHIN your framework, not a replacement. The Σ_0 = a_0/(4π²G) finding may be the most concrete connection of your a_0 to RAR shape that has been demonstrated.

Repository: https://github.com/uuki555-cyber/sdhg-gravity

I make no priority claims. The work is offered for use without attribution constraints. If the result advances your research or the SPARC team's work, please use freely.

Best regards,
[Your name]

---

## Sending strategy

**Priority**:
1. **Send to Lelli first** (lower-stakes, observational expert, more likely to read carefully)
2. Wait 2 weeks for response
3. **Send to Mistele** (younger, theoretical, may forward to Famaey)
4. **Send to McGaugh last** (gets many such emails; better with prior endorsements)

**Don't**:
- Send to multiple people simultaneously (looks scattershot)
- Make priority claims ("we discovered first")
- Use language that implies you've solved MOND
- Attach files (they'll skip; provide repo URL only)

**Do**:
- Include 2-3 specific questions they can answer in 5 minutes
- Frame as building on their work, not superseding
- Acknowledge limitations upfront
- Offer the work freely (no attribution constraints reduces friction)

**Realistic expectations**:
- 50% no response (busy researchers, many similar emails)
- 30% polite acknowledgment
- 15% substantive feedback
- 5% leads to collaboration / formal review

The expected value of sending all three is positive, but the most likely outcome is a polite acknowledgment from one. If that one is Lelli (technical) or Mistele (theoretical), it would still be valuable.

**arXiv submission** can happen before, during, or after these emails. Sending the email with a "preprint coming soon" framing is reasonable.
