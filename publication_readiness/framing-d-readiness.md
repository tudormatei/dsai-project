# Publication Readiness — Framing D

**Paper angle**: Concept vocabulary design is the primary bottleneck in sensor-based CBMs.

**Core finding**: 3 human-labeled concepts yield 36.8% activity accuracy; adding one rule-based feature (`movement_variability`) nearly doubles it to 65.8%. Model architecture matters far less than concept choice.

**Target venue**: EXPLAINS (XAI workshop/conference)

---

## The Framing

### Thesis
In sensor-based Concept Bottleneck Models, the choice of concepts matters more than the choice of model. Specifically, human-defined concepts alone are insufficient to bridge the accuracy gap between interpretable and black-box models — hybrid concept vocabularies (human-labeled + algorithmically computed) are necessary.

### Narrative Arc
1. **Setup**: CBMs promise interpretability at a small accuracy cost. But how much cost depends entirely on whether the concepts capture enough discriminative information.
2. **Experiment**: We systematically ablate concept combinations in a sensor-based HAR CBM, testing every subset of 3 human-labeled + 2 rule-based concepts.
3. **Core finding**: With only 3 human-labeled concepts, the ground-truth concept → activity accuracy is just 36.8%. Adding a single rule-based feature (`movement_variability`) nearly doubles it to 65.8%. No architectural change produced anything close to this effect.
4. **Implication**: Concept vocabulary design is the primary bottleneck in CBMs, not model architecture. This is understudied — most CBM papers treat concept selection as a given and focus on model improvements. Our results suggest that effort should be redirected toward concept engineering.
5. **Provocation**: The most impactful concept was rule-based, not human-defined. This raises a tension: if algorithmic features outperform human concepts, what does this mean for the interpretability promise of CBMs?

### Why The Paper Isn't About Performance
The paper does not claim competitive HAR accuracy. It uses the CBM pipeline as an experimental apparatus to study *where interpretability breaks down* — which is a question about concepts, not models.

---

## Why Other Framings Don't Work

### ❌ Framing A: "Our CBM achieves 0.9 AUROC on HAR" (Performance Claim)
- The 0.9 AUROC comes from a single random split with n=38 test samples. CIs are 0.74–1.00.
- LOUO accuracy is 21.7% — the model doesn't generalize to new users.
- Only 6 users / 150 windows. Published HAR benchmarks use 20–30+ users.
- A reviewer would reject this as an underpowered, non-generalizable result. You'd need substantially more data and proper LOUO performance to make this claim.
- **Blocker**: Requires new data collection (10+ more users). Not feasible in 3–4 months as a side project.

### ❌ Framing B: "How labeling methodology affects CBM performance" (Labeling Study)
- The three labeling iterations used different windows, different users (iter 3 added 2 new users), and different labeling methods simultaneously. The comparison is confounded.
- No κ exists for iteration 3 (consensus labeling makes independent agreement unmeasurable).
- A reviewer would ask for a controlled experiment: same windows labeled under both independent and consensus conditions.
- **Blocker**: Requires organizing a new labeling session with the original team. Logistically difficult months later.

### ❌ Framing C: "Lessons learned applying CBMs to sensor data" (Experience Report)
- Feasible but low-impact. Experience reports are the easiest to write but hardest to get accepted — reviewers see them as anecdotal, not scientific.
- The development journey (26% → 70.7%) is compelling narratively but not rigorous enough to stand as a contribution.
- **Not a blocker, just weak**: Likely gets a "borderline reject" with feedback like "interesting but lacks a clear scientific contribution."

### ✅ Framing D: "Concept vocabulary design is the bottleneck" (This One)
- The concept ablation data already exists and is clean.
- The core finding (one rule-based feature nearly doubles accuracy) is a concrete, quantifiable result.
- It sidesteps the weak LOUO numbers because it's not claiming deployment-ready performance — it's studying *where* the CBM loses information.
- The remaining work is scripting, stats, and writing — no new data collection, no teammate coordination.
- It directly addresses the EXPLAINS audience: "Here's where interpretability breaks down in practice."

---

## What's Already Done

| Item | Status | Notes |
|---|---|---|
| Repeated random splits | ✅ Partially done | 10 manual runs exist. Needs scripting to 50–100 runs with formal mean ± std reporting. |
| Confidence intervals | ✅ Done | Bootstrap CIs computed for all concepts (n=38). Already in `CBM.ipynb`. |
| K-fold CV investigation | ✅ Done | Tried 5-fold and 7-fold stratified CV. Failed due to small class sizes (`min(class_count) < n_splits`). Documented as a negative result — this is valid. |
| Concept ablation benchmarks | ✅ Done | All concept combinations tested. Results in `benchmarks.md`. |
| LOUO vs random split comparison | ✅ Done | Both evaluated: LOUO 21.7% (26.1% with augmentation), random split 70.7%. |

---

## Remaining Work

### 1. Script the repeated random splits (~1 day)
- **Current state**: 10 runs done manually.
- **What to do**: Automate 50–100 repeated activity-stratified 75/25 splits. Report mean ± std for each concept combination. This replaces the single-split results with robust estimates.

### 2. Add significance tests (~1 day)
- **Current state**: No statistical tests on concept combination comparisons.
- **What to do**: Apply McNemar's test or paired bootstrap to compare each concept configuration against the 3-concept baseline. Confirm whether the 36.8% → 65.8% jump (with `movement_variability`) is statistically significant.

### 3. Apply multiple comparison correction (~1 hour)
- **Current state**: Many concept subsets tested without correction.
- **What to do**: Apply Bonferroni correction across tested configurations. If `movement_variability`'s effect survives correction (likely given the effect size), this strengthens the core claim.

### 4. Write the rule-based concept discussion (~1 day)
- **Current state**: `movement_variability` (rule-based) outperforms all human-labeled concepts, but this isn't critically examined.
- **What to do**: Frame as a discussion contribution: "The most impactful concept was rule-based, suggesting human-defined concepts alone are insufficient. Hybrid concept vocabularies may be necessary for practical CBMs." Discuss implications for the interpretability promise of CBMs.

### 5. Literature review (~1–2 weeks)
- **Current state**: No survey of CBM or HAR concept-based literature.
- **What to do**: Cover: (a) CBMs (Koh et al. 2020, Yuksekgonul et al. 2023), (b) concept completeness/sufficiency literature, (c) HAR on WISDM, (d) interpretable HAR approaches. Position the paper as addressing concept vocabulary design, which existing CBM papers take for granted.

### 6. Rewrite and reposition the paper (~3–4 weeks)
- **Current state**: The existing report is written as a course project report, not a research paper. It frames the work around the development journey (26% → 70.7%) and the CBM pipeline, not around concept vocabulary design.
- **What to do**: Restructure the paper entirely around Framing D. This means:
  - **New introduction** motivating the concept vocabulary problem specifically
  - **New experiments section** centered on the concept ablation as the primary analysis, not a side result
  - **New discussion** interpreting the rule-based vs. human concept finding and its implications
  - **Reframe the development journey** as background/motivation, not the main contribution
  - **Cut or compress** sections about model architecture details, data augmentation mechanics, and labeling iteration history — these become supporting material, not the story
- This is not editing the existing paper — it's writing a substantially new paper that happens to reuse the same data and results.

---

## Issues to Acknowledge (not fix)

These go in the **Limitations** section as honest disclosures:

| Issue | One-liner for the paper |
|---|---|
| Small dataset (150 windows, 6 users) | "Our findings should be validated on larger datasets with more participants." |
| Labeling used consensus (no κ for final labels) | "Final labels were produced via group consensus; independent reliability metrics were not computed for this iteration." |
| 3-level discretization is coarse | "Finer concept granularity may improve concept-to-activity mapping and warrants future investigation." |
| Concept validity not independently verified | "We did not formally validate that concept labels correspond to measurable signal properties." |
| Augmentation parameters not ablated | "Augmentation hyperparameters were selected empirically; a systematic ablation is left for future work." |
| Hyperparameter search not formalized | "Model architecture was selected through iterative experimentation; the search was not exhaustive." |
| LOUO generalization is weak (21.7%) | "User-independent generalization remains limited, likely due to high inter-user variability and small user pool." |

---

## Suggested Paper Structure

1. **Introduction**: CBMs trade accuracy for interpretability, but how much depends on concept quality — this is understudied.
2. **Background**: CBMs, HAR, WISDM dataset.
3. **Method**: CBM pipeline, concept definitions (3 human + 2 rule-based), evaluation protocol.
4. **Experiments**:
   - Concept ablation: accuracy and AUROC for every concept combination (with repeated splits + significance tests)
   - Evidence that `movement_variability` is the dominant factor
   - CBM vs. black-box comparison (same evaluation protocol)
   - K-fold CV failure analysis as methodological contribution
5. **Discussion**: Why `movement_variability` matters, what this means for concept design in CBMs, hybrid concept vocabularies, limitations.
6. **Conclusion**: Concept vocabulary design is the bottleneck, not model architecture.

---

## Timeline Estimate (3–4 months, side project)

| Month | Focus | Deliverables |
|---|---|---|
| 1 | Script repeated splits, add significance tests, Bonferroni correction | Updated experiment results |
| 2 | Literature review, rewrite introduction + background + method | First draft of repositioned paper |
| 3 | Rewrite experiments + discussion, incorporate supervisor feedback | Complete draft |
| 4 | Final revisions, prepare supplementary materials, submit | Camera-ready paper |

> **Note**: The paper rewrite (months 2–3) is the largest single effort. The existing report cannot be lightly edited into Framing D — it needs to be restructured around the concept vocabulary analysis as the central contribution, with the development journey and pipeline details compressed into supporting context.
