# Publication Readiness: Full Issue List

Comprehensive list of methodological issues that would need to be addressed to bring this project to publication standard (targeting EXPLAINS or similar XAI workshop/conference).

---

## 1. Data & Sampling Issues

### 1.1 Very Small Dataset
- **Problem**: 150 windows total (iteration 3), 6 users. Many HAR papers use thousands of windows across 20–30+ users.
- **Reviewer concern**: "Are these results generalizable or artifacts of a tiny sample?"
- **To fix**: Collect data from at least 10–15 users, target 300+ labeled windows minimum.

### 1.2 Different Windows Sampled Per Iteration
- **Problem**: Each labeling iteration sampled different time windows from the same users. This means accuracy differences (64.94% → 58.33% → 72.93%) conflate label quality changes with window difficulty changes.
- **Reviewer concern**: "How do you know iteration 3 is better because of consensus labeling and not because those windows were easier to classify?"
- **To fix**: Relabel a shared subset of windows under both independent and consensus protocols to create a controlled comparison.

### 1.3 No Control Over Window Difficulty
- **Problem**: Windows are randomly sampled. Some may capture clean mid-activity segments, others may capture transitions or ambiguous moments. There's no stratification by difficulty.
- **Reviewer concern**: "Your sampling strategy could bias results."
- **To fix**: Document the sampling strategy. Ideally, annotate windows with a difficulty/ambiguity score and show results stratified by difficulty.

### 1.4 Class Balance Not Verified Across Iterations
- **Problem**: Each iteration samples 25 windows per user with 4–5 windows per activity, but the exact class distribution across iterations isn't controlled.
- **Reviewer concern**: "Are accuracy differences driven by different class distributions?"
- **To fix**: Report per-class sample counts per iteration and normalize comparisons accordingly.

### 1.5 Two New Users Added in Iteration 3
- **Problem**: Iteration 3 introduced users 8 and 12, who weren't in iterations 1 or 2. This means iteration 3 has both different labels AND different underlying data.
- **Reviewer concern**: "You changed two variables at once — labeling method and user population."
- **To fix**: Report iteration 3 results separately for shared users (3, 5, 6, 7) vs. new users (8, 12) to disentangle the effects.

---

## 2. Labeling Methodology Issues

### 2.1 No Inter-Rater Reliability for Iteration 3
- **Problem**: κ was only measured for iteration 1 (κ = 0.207). Iteration 3 used group consensus, so independent ratings don't exist.
- **Reviewer concern**: "You claim labels improved but can't quantify inter-rater agreement for your final labels."
- **To fix**: Either (a) have labelers independently rate a subset *before* the consensus discussion to compute pre-consensus κ, then compare to post-consensus labels, or (b) reframe consensus labeling as a deliberate methodology choice and cite precedent (Hripcsak & Rothschild 2005, etc.).

### 2.2 Concept Definitions May Be Underspecified
- **Problem**: Periodicity, temporal stability, and coordination are somewhat subjective. The labeling interface may not have included precise operationalized definitions with anchor examples.
- **Reviewer concern**: "How were labelers trained? Were the concepts defined precisely enough for consistent application?"
- **To fix**: Document the exact definitions, training protocol, and anchor examples used. If these didn't exist, this is a gap to acknowledge.

### 2.3 Three-Level Discretization Is Coarse
- **Problem**: Concepts are labeled as 0, 0.5, or 1.0. This limits the expressiveness of the concept bottleneck — only 3^5 = 243 possible concept vectors for 6 activities.
- **Reviewer concern**: "Is the low concept-to-activity accuracy caused by inherent concept insufficiency or artificial discretization?"
- **To fix**: Experiment with finer granularity (e.g., 5-level scale: 0, 0.25, 0.5, 0.75, 1.0) or continuous labels, and compare concept-to-label accuracy across granularities.

### 2.4 No Labeler Qualification or Calibration Protocol
- **Problem**: Labelers were fellow students with no prior experience in sensor data annotation (first XAI project for everyone).
- **Reviewer concern**: "Were labelers qualified? Was there a calibration phase?"
- **To fix**: Document any training/calibration that occurred. Acknowledge labeler expertise level. For future work, include a practice phase with feedback before real annotation begins.

### 2.5 Concept Validity Not Independently Verified
- **Problem**: The chosen concepts (periodicity, temporal stability, coordination) were selected based on domain reasoning, but there's no formal validation that these concepts actually capture meaningful properties of the sensor signal.
- **Reviewer concern**: "Are these concepts measuring what you think they're measuring?"
- **To fix**: Correlate concept labels with computed signal features (e.g., FFT peak strength for periodicity, signal variance for temporal stability) to show concept labels align with measurable signal properties.

---

## 3. Evaluation Methodology Issues

### 3.1 Random Shuffle Split vs. Leave-One-User-Out
- **Problem**: The headline 70.7% accuracy uses random shuffle split, meaning training and test sets likely contain windows from the same users. LOUO accuracy is only 21.7%.
- **Reviewer concern**: "Your model doesn't generalize to unseen users, which is the actual deployment scenario."
- **To fix**: Report LOUO as the primary metric. Use random split only as a development/debugging tool. If LOUO remains very low, this fundamentally limits the paper's claims.

### 3.2 No Cross-Validation
- **Problem**: Results are reported from a single train/test split. With n=38 test samples, a single split is highly sensitive to which windows end up in the test set.
- **Reviewer concern**: "A single split with 38 test samples is unreliable."
- **To fix**: Use k-fold cross-validation (or repeated random splits with reported mean ± std) to demonstrate result stability.

### 3.3 Test Set Too Small for Meaningful Statistical Claims
- **Problem**: Test set is n=38 (or fewer depending on the split). Per-class test counts are as low as 6 samples. Confidence intervals are extremely wide (e.g., 64.1%–90.7%).
- **Reviewer concern**: "These results are statistically inconclusive."
- **To fix**: Increase dataset size, use cross-validation, or at minimum use bootstrap CIs and be transparent about statistical power limitations.

### 3.4 Augmented Data Used in Training, Not Controlled For
- **Problem**: 10x augmentation expands training from 73 to 803 samples. The augmentation parameters (jitter σ=0.05, scale σ=0.1, rotation ±30°) were presumably chosen by experimentation, but there's no ablation over augmentation parameters.
- **Reviewer concern**: "How sensitive are results to augmentation choices? Were these hyperparameters tuned on the test set?"
- **To fix**: Ablate over augmentation type and magnitude. Use a held-out validation set (not the test set) for hyperparameter selection.

### 3.5 Black-Box Baseline Comparison May Not Be Fair
- **Problem**: The black-box model achieving 0.96 AUROC presumably uses the same small dataset and random split. This baseline may be inflated by the same evaluation issues as the CBM.
- **Reviewer concern**: "Both your CBM and black-box results may be overfit to this particular split."
- **To fix**: Evaluate both models under identical, rigorous conditions (same cross-validation folds, same LOUO protocol).

---

## 4. Model & Architecture Issues

### 4.1 Hyperparameter Selection Process Undocumented
- **Problem**: The 3,180-parameter architecture was arrived at through "testing tens of ideas." The search process and selection criteria aren't formally documented.
- **Reviewer concern**: "How do we know this architecture wasn't selected because it happened to perform well on this specific test set?"
- **To fix**: Document the hyperparameter search space, selection criterion, and whether a separate validation set was used (not the test set).

### 4.2 No Comparison to Standard HAR Baselines
- **Problem**: The project compares the CBM to its own internal black-box model, but doesn't benchmark against published HAR methods on the same dataset (or the WISDM dataset it originates from).
- **Reviewer concern**: "How does this compare to existing work on WISDM?"
- **To fix**: Include standard baselines (DeepConvLSTM, InceptionTime, etc.) on the same data and evaluation protocol. Reference published WISDM benchmark numbers.

### 4.3 Rule-Based Concepts Mix Paradigms
- **Problem**: `movement_variability` and `movement_consistency` are computed algorithmically from the signal, not labeled by humans. Mixing human-labeled and rule-based concepts muddles the CBM's interpretability claim.
- **Reviewer concern**: "If the best concept is rule-based, why do you need a CBM at all? Why not just use the computed features directly?"
- **To fix**: Clearly separate results for human-only concepts vs. human+rule-based concepts. Discuss the implications — the rule-based features arguably bypass the concept bottleneck.

### 4.4 Concept-to-Label Classifier Choice Not Justified
- **Problem**: Both Logistic Regression and Random Forest are used for concept → activity prediction. The choice between them and their hyperparameters aren't formally justified.
- **Reviewer concern**: "Why these classifiers? Were they tuned?"
- **To fix**: Report results for multiple classifiers with tuned hyperparameters and justify the final choice.

---

## 5. Experimental Design Issues

### 5.1 No Pre-Registration or Hypothesis
- **Problem**: The project was exploratory (testing ideas until something worked), not hypothesis-driven. This makes it hard to distinguish genuine findings from post-hoc rationalization.
- **Reviewer concern**: "This looks like p-hacking / HARKing (Hypothesizing After Results are Known)."
- **To fix**: Be transparent about the exploratory nature. Frame clearly which results are exploratory findings vs. confirmed hypotheses. For a workshop paper, this is less critical than for a journal.

### 5.2 Labeling Iteration Comparison Is Uncontrolled
- **Problem**: The three iterations differ in (a) windows sampled, (b) labeling method, (c) users included. You can't attribute the accuracy improvement to any single factor.
- **Reviewer concern**: "This is a confounded comparison."
- **To fix**: Design a controlled experiment: same windows, same users, two conditions (independent vs. consensus). Compare directly.

### 5.3 No Ablation Over Number of Training Windows
- **Problem**: You claim the model benefits from data augmentation (73 → 803 samples), but there's no learning curve showing how accuracy scales with training set size.
- **Reviewer concern**: "How do we know 803 augmented samples is enough? What's the data efficiency?"
- **To fix**: Plot accuracy vs. number of training samples (with and without augmentation) to show the learning curve and whether performance has plateaued.

---

## 6. Statistical Rigor Issues

### 6.1 No Significance Tests
- **Problem**: Comparisons between models/iterations/concept combinations don't include statistical significance tests (e.g., McNemar's test, paired t-test on cross-validation folds).
- **Reviewer concern**: "Are these differences statistically significant or within noise?"
- **To fix**: Apply appropriate significance tests to all key comparisons.

### 6.2 Confidence Intervals Are Acknowledged But Very Wide
- **Problem**: The CIs are reported (good), but they're so wide they overlap for most comparisons, making conclusions unreliable.
- **Reviewer concern**: "Your error bars overlap — you can't claim one method is better than another."
- **To fix**: More data, cross-validation, or bootstrap with more iterations. At minimum, be explicit about which claims the CIs actually support.

### 6.3 Multiple Comparisons Not Corrected
- **Problem**: The concept ablation tests many combinations (3 concepts, 4 concepts, 5 concepts, different subsets). No correction for multiple comparisons (Bonferroni, FDR) is applied.
- **Reviewer concern**: "With this many comparisons, some 'best' results are expected by chance."
- **To fix**: Apply multiple comparison correction or clearly state which comparisons were pre-planned vs. exploratory.

---

## 7. Presentation & Reproducibility Issues

### 7.1 Code Is in Notebook Format
- **Problem**: Core pipeline is spread across 6 Jupyter notebooks. This is standard for coursework but not for reproducible research.
- **Reviewer concern**: "Can I reproduce these results?"
- **To fix**: Refactor into modular Python scripts with a clear entry point, configuration file, and a README with exact reproduction steps.

### 7.2 No Random Seed Documentation
- **Problem**: Data augmentation, train/test splitting, and model training all involve randomness. If seeds aren't fixed and documented, results aren't reproducible.
- **Reviewer concern**: "I ran your code and got different numbers."
- **To fix**: Fix and document all random seeds. Report results as mean ± std over multiple seeds.

### 7.3 Dataset Not Publicly Shareable (Potentially)
- **Problem**: The raw sensor data comes from WISDM, which is public, but the concept labels are your team's annotation. It's unclear if the labeled dataset can be shared.
- **Reviewer concern**: "Is the annotated dataset available for replication?"
- **To fix**: Create a clean, shareable version of the annotated dataset with documentation. Host on a repository (Zenodo, GitHub).

### 7.4 No Related Work Section Drafted
- **Problem**: As a course project, there's likely no systematic literature review of CBMs applied to HAR or sensor data.
- **Reviewer concern**: "How does this relate to existing work?"
- **To fix**: Survey CBM literature (Koh et al. 2020, etc.), HAR literature (WISDM benchmarks), and XAI-for-sensors work. Position your contribution clearly.

---

## Summary: Priority Ranking

| Priority | Issues | Effort |
|---|---|---|
| 🔴 **Must fix** | 3.1 (LOUO vs random split), 2.1 (no κ for iter 3), 3.2 (no cross-validation), 5.2 (uncontrolled iteration comparison) | High |
| 🟠 **Should fix** | 1.1 (small dataset), 1.2 (different windows per iteration), 3.3 (small test set), 4.2 (no standard baselines), 4.3 (rule-based concept mixing), 6.1 (no significance tests) | Medium–High |
| 🟡 **Nice to fix** | 2.3 (coarse discretization), 2.5 (concept validity), 3.4 (augmentation ablation), 4.1 (hyperparameter documentation), 5.3 (learning curve), 7.1 (notebook → scripts) | Medium |
| 🟢 **Acknowledge** | 1.5 (new users in iter 3), 2.4 (labeler qualification), 5.1 (exploratory design), 6.2 (wide CIs), 6.3 (multiple comparisons) | Low (just write about it honestly) |
