# BioXplain methodology decisions

Status: written 2026-09-21, **before any model was fitted and before any GSE65194 expression result was examined**.
"Outcome timing" states what had been looked at when each decision was made. Information seen so far: series-matrix metadata, class counts, global value scale and floor statistics (label-free) of both cohorts, probe-ID sets, GPL570 annotation, and the processing-date / title-prefix vs class cross-tabulation in GSE42568 (`docs/data_reconnaissance.md`). **No decision below used a classification result, a gene ranking, a class contrast, or any GSE65194 gene-level value.**
Decisions marked **[SENSITIVITY]** are judgement calls that will be re-examined by a concise sensitivity analysis. Decisions marked **[DEFERRED]** have their principle fixed but the final parameter values will be frozen in `configs/` at the discovery freeze (Phase 8), using GSE42568 only.

Frozen facts (from reconnaissance; must not be changed silently): GSE42568 = 54,675 probes x 121 samples (104 cancer / 17 normal), GPL570, GC-RMA, floor value 2.3128 (28.5% of values); GSE65194 = 178 arrays (130 unique tumours, 23 duplicated tumours, 11 healthy, 14 cell lines), batch-corrected, no floor; 0 shared GSM; GPL570 = 54,675 records, 42,894 usable probes, 20,848 gene symbols, 42,892 usable probes present in both series.

---
## D1. Discovery cohort
- **Decision:** All 121 GSE42568 samples; label 1 = "cancer" (positive class), 0 = "normal". No samples excluded from the primary analysis.
- **Rationale:** Labels are unambiguous (tissue, source_name, title prefix and description agree for all 121). Excluding samples would only shrink an already small normal class (17).
- **Alternatives:** exclude the 3 tumours with unparseable dates; restrict to one processing month (see D19b).
- **Outcome timing:** Before. **Bias:** The known date/prefix-class confounding (9 of 17 normals dated Jan 2005 vs 4 of 104 tumours) may create predictive signal unrelated to tissue biology. **Sensitivity:** Dec-2004-only subset (97 tumours + 8 normals; the 3 undated tumours and 4 Jan-2005 tumours are outside the subset), Phase 11.

## D2. External cohort
- **Decision:** GSE65194, used only after the discovery freeze (Phase 8). Analysis set: 130 unique tumours + 11 healthy (D3, D4).
- **Rationale:** Independent institution, patients, batch and labelling protocol; same platform so probe spaces match. Different chip **technology** is not tested (same GPL570), so "independent" means independent cohort only.
- **Alternatives:** TCGA-BRCA RNA-seq (cross-platform; more work, not planned).
- **Outcome timing:** Before (metadata/scale only). **Bias:** Submitter batch-correction with an unknown design (class term may or may not have been in the model) and TNBC enrichment (41/130) vs ER+ dominance in discovery. Reported as limitations.

## D3. Technical-replicate handling (GSE65194)
- **Decision:** Tumours identified by the `TUMnnn` token in the title; the 23 tumours with two arrays (`_repA/_repB`) are collapsed to one unit by the **arithmetic mean of the two arrays' expression values, per probe**. All other units are single arrays. Result: 130 tumour units + 11 healthy units = 141. Pair IDs, pair Pearson r (0.994-0.997 observed at recon) and the rule are written to a manifest at Phase 9.
- **Rationale:** The submitter states that replicates "were subsequently averaged", but the matrix contains both arrays (0/23 pairs identical), so the matrix is not the averaged data described. Treating both arrays as samples would double-count the tumour (pseudo-replication).
- **Alternatives:** keep one array per tumour (arbitrary choice); median.
- **Outcome timing:** Before. **Bias:** Averaging log-scale values is a mean of logs (geometric-type), slightly shrinks noise for 23 tumours. **Sensitivity:** repeat the external analysis using only repA (Phase 10).

## D4. Cell-line exclusion
- **Decision:** Exclude all 14 `CellLine` arrays.
- **Rationale:** Cell lines are not tissue; a tumour-vs-normal tissue signature is not defined for them. Two lines (MCF-12A, 184B5) are generally described as non-tumorigenic, which would make label "cancer" ambiguous. The rule is metadata-only (`sample_group`).
- **Outcome timing:** Before. **Bias:** none expected; exclusion is class-independent.

## D5. Shared feature universe
- **Decision:** Candidate probes = GPL570 "usable" probes (single gene, non-AFFX, non-empty symbol) **that are present in both series matrices** = 42,892 probes. The two GSE42568-only probes (STAT2, CEP128) are dropped from the universe.
- **Rationale:** Guarantees every frozen gene can be evaluated externally, and gives an a-priori, data-independent enrichment background (D18). Only probe-ID lists are used, not external expression values.
- **Alternatives:** all 54,675 probes; all usable probes regardless of presence.
- **Outcome timing:** Before. **Bias:** Excludes 4% multi-gene probes and 17.5% unannotated probes (predictive signals on unannotated probes cannot be interpreted biologically). **Note:** annotation is dated Aug 2016; symbols may be outdated (mitigation in D18).

## D6. Probe-to-gene mapping **[SENSITIVITY]**
- **Decision (primary):** Inside each training fold, for each gene with more than one probe (after the expression filter D7), keep the probe with the **highest mean expression over the training samples**; ties broken by probe ID (alphabetical). Genes with one probe keep it. The gene symbol is the feature identity. The probe chosen is recorded per fold.
- **Rationale:** Max-mean is a standard, label-free rule that selects the best-detected probe (Miller et al. 2011 compared such rules; not re-read in this session, citation to be verified). Fold-internal computation ensures the validation fold never influences which probe represents a gene. A rule that did use the full data would be a (mild) leak.
- **Alternatives:** highest variance (implemented as the sensitivity alternative); highest IQR; mean of probes; probe-level modelling; a-priori rule with no data dependence.
- **Outcome timing:** Before (no classification result seen). **Bias:** Max-mean prefers high-abundance, often 3'-biased probes and can miss a discriminating probe of a lower-mean transcript isoform. It is label-free so it does not inflate performance. **Sensitivity:** repeat the pilot-scale analysis with max-variance and report whether top-k sets and stability change materially (Phase 6). Variance-based rules are more label-associated in principle, so they are the more cautious alternative to compare.
- **Frozen mapping for external use (D17):** the probe per gene chosen by the same rule on the *full* GSE42568 (discovery only), applied at that probe in GSE65194.

## D7. Expression (floor-detection) filter **[SENSITIVITY]**
- **Decision (primary):** Inside each training fold: estimate the floor as the minimum training value (accepted only if at least 1% of training values equal it; otherwise no filter is applied and a warning is logged). A probe is **detected** in a sample if its value exceeds the floor. Keep probes detected in at least `ceil(0.5 x n_minority_train)` training samples (about 7 of 97 for GSE42568 5-fold training sets).
- **Rationale:** 28.5% of GSE42568 values sit at the floor, so many probes carry no information beyond "not detected", and ties would dominate variance filters, t-tests and permutation importance. The threshold is tied to the **minority-class size** deliberately: a gene whose signal is presence in one class must be detected in about that class's samples, so requiring detection in at least half of the minority class cannot remove genes solely because they are class-specific, whereas a common cut-off such as 20% of all samples (about 19 of 97) would remove every normal-specific gene (only ~13 normals in a training fold). The only label information used is the minority-class **count**, never expression-label relations.
- **Alternatives:** fixed fraction of samples (5%, 10%, 20%); variance or IQR filter; no filter.
- **Outcome timing:** After label-free floor statistics were seen (5,350 probes at floor in >=90% of samples; disclosed), **before** any outcome. Threshold was derived from the argument above, not tuned. **Bias:** Genes detected in fewer than half the minority class are lost, including rare-detection genes. **Sensitivity:** 0.25 x and 1.0 x n_minority, and a 10%-of-samples rule (Phase 6). **Test:** a synthetic gene present only in the minority class must survive the filter (tests/test_preprocessing.py).

## D8. Scaling
- **Decision:** Per-gene z-scoring (mean 0, sd 1) with statistics from the training fold only, after the filter and collapse; applied to LR and SVM. Tree models use unscaled log2 values (scale-invariant).
- **Rationale:** Regularised linear models need comparable feature scales; coefficient magnitudes are then comparable across genes.
- **Alternatives:** no scaling; rank transform; robust scaling.
- **Outcome timing:** Before. **Bias:** z-scoring inflates rarely-detected probes (a probe detected in ~7 samples gets z about 3.5 in those samples); this is an artefact to monitor. It will be checked by looking at the top-ranked genes' detection counts in the pilot.

## D9. Feature selection for predictive modelling
- **Decision:** No supervised feature selection inside the classifier pipeline. All filtered genes enter the model; "selection" for the stability analysis is the **top-k of an explanation ranking** (D13). The t-test baseline (D16b) is a ranking method, not a pre-filter.
- **Rationale:** Pre-filtering with a supervised method would confound the model's explanation stability with the filter's stability.
- **Outcome timing:** Before.

## D10. Model set
- **Decision:** (1) L2-penalised logistic regression (linear), (2) SVM with linear kernel (margin-based), (3) Random Forest (bagged trees), (4) XGBoost (boosted trees). Different inductive biases, not a leaderboard.
- **Status:** (1) implemented in the pilot slice; (2)-(4) parameter values pre-specified in D11b (2026-09-22, before any SVM/RF/XGBoost result); implemented and benchmarked in Phase 3.
- **Alternatives:** elastic-net LR, kernel SVM, kNN, neural networks (not planned).

## D11. Hyperparameter policy **[DEFERRED]**
- **Decision:** No search in the primary analysis. Fixed, pre-specified configurations. Pilot values: LR: L2, `C=1.0`, `class_weight="balanced"`, lbfgs, `max_iter=5000`. Values for the other models are fixed in `configs/` in Phase 3 from library defaults plus class balancing, and not tuned. If a sensitivity check on regularisation strength is run, it is tuned or compared only inside training data and reported as sensitivity.
- **Rationale:** Tuning inside 97-sample training folds with 4-5 normals per validation fold is itself unstable and would add a second source of variability to the stability question. Note (theory): for strongly regularised LR the coefficient vector approaches a mean-difference direction, so LR-coefficient rankings may resemble univariate rankings (hypothesis H3); this is reported, not assumed.
- **Outcome timing:** Before. **Bias:** Default configurations may be suboptimal; accepted because the goal is methodological comparison.

## D12. Explainer set **[DEFERRED]**
- **Decision:** (a) LR coefficients (`|coef|` on z-scored genes; signed value retained); (b) SHAP: TreeSHAP for RF/XGBoost, LinearSHAP for linear models, mean |SHAP| over **training-fold samples**; (c) permutation importance (drop in ROC-AUC). (d) t-test baseline (D16b).
- **They are not the same quantity:** coefficients are the model's linear weights (conditional effect given other genes); SHAP is an additive attribution of the model output, averaged over samples (reflects model behaviour on the reference distribution, affected by correlated features); permutation importance is the loss increase when one gene is broken (depends on the evaluation data and on correlated genes, which can mask importance). None is causal.
- **Open compute issue (Phase 4):** exact permutation importance over ~15-20k genes for tree models is expensive. If it is infeasible a principled reduction will be proposed and documented (not silently applied); a candidate reduction must not be chosen using validation-fold or external information.
- **Note (see D12c):** LinearSHAP on z-scored features is proportional to |coef| times mean |z|, so LR-coefficient and LR-SHAP rankings are expected to be almost redundant; this will be reported and not treated as independent evidence.
- **Outcome timing:** Before.

## D11b. Fixed model configurations and imbalance policy (pre-specified 2026-09-22, before any SVM / RF / XGBoost result)
All four models use class-balanced training so that none is favoured by the 86% cancer prevalence. No parameter is tuned.

| Model | Configuration | Input scale | Score used for ROC/PR | Predicted label |
|---|---|---|---|---|
| Logistic regression (primary C) | L2, `C=1.0`, `class_weight="balanced"`, lbfgs, `max_iter=5000` | z-scored | probability | p > 0.5 (estimator `predict`) |
| Linear SVM | `LinearSVC(C=1.0, loss="squared_hinge", penalty="l2", dual=True, class_weight="balanced", max_iter=20000)` | z-scored | decision function (margin) | margin > 0 |
| Random forest | `RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=4)` | log2 values (scale-invariant) | probability | p > 0.5 (estimator `predict`) |
| XGBoost | `XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, tree_method="hist", n_jobs=4, scale_pos_weight = n_normal_train / n_cancer_train)` (ratio computed inside the training fold) | log2 values | probability | p > 0.5 (estimator `predict`) |

`random_state` = master seed (D22). Rationale for these values: library defaults or standard small-data choices (shallow boosted trees, sqrt features), chosen for balance and not for accuracy. Alternatives (kernel SVM, deeper trees, tuning by nested CV) are not planned. The thread count is part of the recorded configuration; determinism is tested run-to-run. Measured 2026-09-22: RF scores are identical for `n_jobs` 1, 2 and 4, but XGBoost scores differ by up to about 1e-2 between `n_jobs` 1 and 4, so `n_jobs=4` is pinned in the spec and the resolved spec is stored in every manifest. Predicted labels are each estimator's own `predict` (probability > 0.5 for LR/RF/XGBoost, margin > 0 for the SVM), so ties at exactly 0.5 go to the normal class.

**Metric hierarchy (primary vs supporting):** *primary* = ROC-AUC (threshold-free; independent of prevalence) and normal-class average precision `ap_normal` (the informative PR-AUC when the minority class is the one that matters); *supporting* = balanced accuracy, sensitivity (cancer recall), specificity (normal recall), F1 per class and confusion counts (these depend on the fixed decision threshold). Plain accuracy is not a headline metric. Near-ceiling AUC is an observation about this cohort (tumour vs normal is easily separable), not a success criterion, and does not influence any decision below.

## D11c. Regularisation-strength sensitivity for logistic regression (pre-specified 2026-09-22, before any sensitivity result)
- **Question:** is the low stability of coefficient rankings seen in the pilot (Nogueira 0.28-0.32) sensitive to reasonable regularisation choices? This is a robustness check, **not** hyper-parameter optimisation, and the primary configuration stays `C=1` whatever the outcome.
- **Grid (fixed now):** C in {0.01, 0.1, 1, 10, 100} (five decades; 1 is the primary/default). L2, balanced weights, same preprocessing.
- **Design:** GSE42568 only; 5 repeats x 5 folds (25 feature sets per C), seed 20260921, identical splits for every C.
- **Reported per C:** Nogueira / Kuncheva / Jaccard at k = 10, 25, 50; pooled out-of-fold ROC-AUC, `ap_normal` and balanced accuracy with bootstrap CIs; **rank agreement with C=1**: mean over folds of the Spearman correlation of |coef| over all eligible genes, and mean over folds of the Jaccard overlap of the top-25 sets.
- **Pre-specified qualitative criterion:** the band of Phi at k=25 in the paper's own scale (Nogueira et al. 2018, Table 3: < 0.40 poor, 0.40-0.75 intermediate to good, > 0.75 excellent), and the statement "pooled ROC-AUC >= 0.98". Conclusions are called **robust** if every C stays in the same Phi band as C=1 and AUC stays >= 0.98, and **materially changed** otherwise. If they change, the paper reports Phi as a function of C (a finding) and investigates the mechanism; the primary C is not re-chosen.
- **Not used:** GSE65194.

## D12b. Permutation-importance strategy (decided 2026-09-22 after a feasibility benchmark; implementation in Phase 4)
**Problem.** About 20,000 genes per fold. Exact permutation importance (PI) for every gene, every fold and every model is expensive for tree ensembles. Benchmark on one real training fold (`scripts/benchmark_pi_feasibility.py`; 96 training samples, 20,087 genes, R = 10 permutations): RF re-prediction 111 ms, so naive exact PI over all genes would take about 22,000 s per fold; XGBoost 46 ms, about 9,000 s per fold. The same benchmark shows that the fitted models are **sparse**: the forest uses 1,247 of 20,087 genes (6.2%) and XGBoost only 24, whereas the linear models are dense (all 20,087 coefficients non-zero).

**Options evaluated.**

| Option | Scientific validity | Leakage risk | Cost | Interpretability / comparability | Effect on the research question |
|---|---|---|---|---|---|
| **A** exact PI over all genes, naive | exact | none if computed on training data | prohibitive for RF/XGB (hours per fold) | ideal | none |
| **B** staged: cheap filter (e.g. t-test top N), PI only on survivors | approximate; depends on the filter | none if fold-internal | low | **poor**: gives PI a different candidate universe from coefficients/SHAP and imports the t-test ranking into one explainer, which biases the explainer-agreement and stability comparisons and contaminates the t-test baseline comparison | distorts the central comparison |
| **C** exact PI with **exactness-preserving** shortcuts | exact (identical to A up to floating point) | none | seconds per fold | same universe and definition as A | none |
| **D** replace PI by another importance (e.g. impurity importance) | changes the explainer | none | low | different quantity, known biases (Strobl et al.) | changes what is compared |

**Recommendation: Option C.** Each shortcut is exact and will be verified against a brute-force reference in tests:
1. **Linear models (LR, SVM):** permuting gene j changes the score only through coef_j x (permuted - original x_j), so the loss for all 20k genes is computed analytically and vectorised.
2. **Tree ensembles:** a gene never used in any split has exactly zero PI, so PI is computed only for genes the fitted model uses (model structure is a property of the training fold; no external information).
3. **RF only:** permuting gene j changes only the trees that split on j, so only those trees are re-evaluated.

No gene is dropped "because it is probably unimportant", and there is no arbitrary top-N. This is Option A's result, computed efficiently.

**Definition (fixed now).**
- Evaluation data = **the training fold** (the model's reliance on each gene on the data it was fitted to). Rationale: it keeps every explainer computed from training-fold information only (the same guarantee the leakage tests enforce), matches the SHAP choice (D12c), and avoids a 24-sample validation fold with 3-4 normals on which almost every gene would have zero or noisy importance. Limitation (documented): for models that fit the training data perfectly, training-fold PI describes reliance and not generalisation value. A **held-out-fold PI** is a pre-specified, exploratory sensitivity variant.
- Loss = **class-balanced Brier score**, `L = 0.5 * [mean_{y=1}(1-p)^2 + mean_{y=0} p^2]`, with p the predicted probability (RF, XGB, LR) or `sigmoid(margin)` for the SVM. Rationale: ROC-AUC would be 1.0 on training data for every gene (no signal); log-loss is unbounded for saturated forest probabilities; class balancing stops the 86% majority class dominating. Importance_j = mean over R permutations of L(permuted j) - L(original). The SVM margin scale is arbitrary, so PI magnitudes are not comparable across models; only rankings are compared.
- R = 10 permutations of the training rows; the same R permutation vectors are applied to every gene within a fold (each (gene, r) is a uniform random permutation of that gene). Permutation seed derived from the master seed, repeat and fold (D22).
- **Known limitations (reported, not hidden):** PI is diluted by correlated genes (co-expressed modules can mask each other). Zuo et al. 2026 reported near-zero stability for permutation importance on transcriptomic neural networks (as summarised in the literature record; not re-read in full), so low PI stability here may reflect the method's correlation sensitivity and not a pipeline defect. Magnitudes are not comparable to coefficients or SHAP values.
- **Uncertainty flagged for the user:** training-fold evaluation is a judgement call; the alternative (held-out fold) is a legitimate reading of "permutation importance". Switching would affect only the PI explainer, would be reported as a sensitivity analysis, and does not alter the research question.

## D12c. SHAP plan (details fixed now; benchmarks in Phase 4)
Explainer per model: `LinearExplainer` (interventional, training-fold background) for LR and the linear SVM; `TreeExplainer` for RF and XGBoost (model output = probability for RF, log-odds for XGBoost). Importance = mean |SHAP| over **training-fold samples**; signed value = mean SHAP. For standardised features, LinearSHAP of a linear model is proportional to coef_j times the feature's mean absolute deviation, so linear-model SHAP and coefficient rankings are expected to be almost redundant; this will be reported as such and not counted as independent evidence. SHAP is one explainer among three; the research question is explanation consistency, not SHAP. The returned dimensions, reproducibility and a known-answer toy test are prerequisites before any full SHAP run.

## D13. Top-k definitions
- **Decision:** k in {10, 25, 50} genes, taken by rank (rank 1 = largest importance; ties broken by gene symbol). The top-200 of every run is stored with rank and importance; k=25 is the pre-specified headline value, 10 and 50 are reported alongside.
- **Rationale:** Small enough to be a "signature", large enough for overlap statistics. k is not tuned on any outcome.
- **Outcome timing:** Before.

## D13b. Zero-attribution rule for top-k selection (pre-specified 2026-09-22, before any tree-model result)
Sparse models (XGBoost uses about 24 genes and RF about 6% of genes in the benchmark fold) have exactly zero attribution for most genes. Ranking those by gene name would fill the top-k with arbitrary but **identical-across-folds** alphabetical genes and manufacture spurious stability. Therefore a gene is **eligible for a top-k set only if its importance is strictly positive**; the top-k set is the k highest, or all positive genes if fewer than k. Set sizes can therefore be smaller than k for sparse models. Nogueira's Phi supports variable set sizes; Kuncheva's index requires equal sizes and is reported only when sizes are constant. The number of positive-attribution genes per run is recorded. Logistic-regression and SVM coefficients are dense, so nothing changes for the pilot.

## D14. Stability metrics
- **Decision:** Primary: **Nogueira Phi** (Nogueira, Sechidis & Brown, JMLR 18(174), 2018, Def. 4, Eq. 2). Secondary: mean pairwise **Kuncheva** consistency index (Kuncheva 2007) and mean pairwise **Jaccard**. Definitions were checked against the JMLR paper text (Definition 4, Theorem 5, Appendix A Table 6) and the stabm documentation examples (Jaccard 0.7166667, Nogueira 0.7222222 for sets {1:3,1:4,1:5}, p=10).
  - Phi = 1 - [(1/d) sum_f s_f^2] / [(kbar/d)(1 - kbar/d)], s_f^2 = M/(M-1) p_f (1 - p_f); kbar = mean set size; undefined if kbar in {0, d}; max 1; min -1/(M-1); expectation 0 under random selection.
  - Kuncheva = (r - k^2/d) / (k - k^2/d) averaged over pairs; requires equal set size k.
  - Jaccard = |A n B| / |A u B| averaged over pairs; not chance-corrected (grows with k/d).
- **d (universe size):** the a-priori candidate gene universe (D5, after the mapping to genes). Genes removed by a fold's expression filter simply cannot be selected in that fold. The effect of using the smaller data-dependent universe is checked in the pilot.
- **Confidence intervals:** the paper's asymptotic CI (Theorem 7) assumes the M feature sets are **independent** samples; cross-validation training sets overlap, so the CI is **not valid** and will not be reported. Uncertainty is conveyed by per-repeat values and by the permutation null (D15).
- **Dimensions (D14b):** (A) resampling: sets from the same model/explainer across repeats x folds; (B) model: across models at fixed explainer type where comparable; (C) explainer: across explainers at fixed model; (D) joint: sets from all model x explainer configurations. Reported separately; never averaged into one score.
- **Outcome timing:** Before. Tested with toy cases (identical sets = 1, complementary sets = -1 for M=2, random sets ~ 0, Theorem 5 identity, stabm reference values).

## D15. Permutation-label null
- **Decision:** For b = 1..B: draw a random permutation of the 121 class labels of the discovery cohort (class counts preserved, expression matrix fixed), then run the **identical** pipeline (same splits, filter, collapse, scaler, model, explainer, top-k) and compute the same stability metrics. Observed stability is compared with the empirical null (mean/median, percentile, empirical p = (1 + #{null >= observed}) / (B + 1)). B will be chosen after benchmarking (target 200-1000; the value and its justification will be logged).
- **What is permuted / fixed:** permuted: labels (globally, once per null replicate); fixed: expression values, cross-validation split seeds/structure, all pipeline settings.
- **Interpretation:** the null describes stability when labels carry no signal; it is a calibration of chance-level stability under this pipeline, **not** proof of biological significance. A single permutation is never called a null distribution.
- **Caveat:** global label permutation also breaks the date/prefix confounding; it does not isolate that confound.

## D16. Consensus definition **[SENSITIVITY]**
- **Decision:** No composite score. Per gene the following are recorded at each k: selection frequency (fraction of runs with the gene in the top-k), median rank, number of resamples selecting it, number of models and of explainer families for which its within-configuration selection frequency is >= 0.5. **Pre-specified consensus set:** genes with within-configuration selection frequency >= 0.5 (k=25) in at least half of the (model x explainer) configurations. The 0.5 thresholds are conventional majority rules and are frozen before external validation.
- **Sensitivity:** thresholds 0.3 / 0.7 / 0.9 and k=10 / 50 on discovery data only.
- **D16b. t-test baseline:** Welch t-statistic (tumour vs normal) computed on the training fold only, ranked by |t|; the same top-k and stability metrics are applied.
- **Outcome timing:** Before.

## D17. External replication definition (pre-specified before any GSE65194 gene-level result)
Definitions are scale-free because the cohorts are on different numeric scales.
- **Frozen inputs:** the discovery-derived quantities per gene from the frozen pipeline: selection frequency, median rank, consensus membership, stability, and the probe per gene chosen on full GSE42568 (D6).
- **Effect measure (per cohort, per gene):** the Mann-Whitney AUC of the gene for tumour vs normal, A_g in [0,1]. Invariant to monotone rescaling, so comparable across cohorts. Signed effect e_g = A_g - 0.5.
- **A. Gene-level replication (primary unit):** *aligned external effect* r_g = sign(e_g^disc) x e_g^ext (positive = same direction). Also the binary direction concordance and, secondarily, external BH-FDR < 0.05 with the same direction (low power with 11 normals; reported, not primary).
- **B. Signature-level replication:** a **frozen consensus signature** score per sample = mean over up-genes of within-cohort z minus mean over down-genes of within-cohort z (direction from discovery), z-scored within each cohort using unlabeled cohort statistics; reported as external AUC with a bootstrap CI. (Within-cohort standardisation is transductive but label-free and is stated explicitly.)
- **C. Sample-level classification:** frozen model applied to external samples after per-gene within-cohort standardisation; ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity with bootstrap CIs. Secondary because of the scale mismatch.
- **Primary confirmatory questions (only two):**
  1. **C1 (association):** Spearman correlation between internal selection frequency (k=25, joint over configurations) and aligned external effect r_g **across all genes in the universe that were selectable**, not only across selected genes. This avoids conditioning on selection.
  2. **C2 (incremental value):** whether selection frequency still predicts r_g after adjusting for the discovery effect size |e_g^disc| (partial Spearman / rank regression). Stable genes are expected to be strong genes; C2 asks whether stability adds information beyond simple effect size. This is the scientifically important question.
- **Uncertainty:** bootstrap over **external samples** (stratified, recompute A_g and the statistics); genes are non-independent (co-expression), so gene-level p-values are not used as evidence, only effect sizes with sample-bootstrap intervals.
- **Everything else is exploratory** and labelled so. Robustness variants (replicate = repA only; external probe chosen by external-cohort max-mean) are reported as sensitivity, not substitutes.
- **Timing:** written before external data were opened at gene level.

## D18. Enrichment background
- **Decision:** Background = the a-priori candidate gene universe (D5, after mapping to genes: genes with at least one usable probe present in both series), identified by **Entrez ID** (stable) and converted to symbols only for the tool. Foreground = the frozen consensus set (and, separately, the frozen top-k lists). Libraries pre-specified: GO Biological Process, Reactome, MSigDB Hallmark (via gseapy; library release/date, query and background logged). Benjamini-Hochberg within each library. Run once after the freeze; not repeated to look for significance. Web-service drift is recorded by date; local GMT files are preferred if downloadable.
- **Outcome timing:** Before.

## D19. Tissue-composition analysis (hypothesis, not assumption)
- **Decision:** Pre-specified primary marker panel (adipocyte markers named a priori): **ADIPOQ, PLIN1, FABP4**. Extended exploratory panel: LEP, LPL, CFD, ADH1B, CIDEC, PLIN4, CD36, GPD1 (from general adipocyte biology; source to be cited). Checks: (a) are panel genes in top-k lists / the consensus set, and how do their selection frequencies rank among all genes (rank percentile, not a p-value on a tiny panel); (b) an adipose-panel score (within-cohort z-mean) as a one-feature classifier in both cohorts and its correlation with the frozen signature score; (c) composition-ablation: repeat the pilot-scale analysis with the panel genes removed and report change in stability and performance. Absence of these genes is reported equally.
- **D19b. Processing-date sensitivity:** Dec-2004-only subset (97 tumours + 8 normals). With 8 normals, 4-5 fold is possible but tiny per fold; if a statistic is unreliable at this size it is reported as such.
- **Outcome timing:** Before. Limitation: composition markers cannot fully separate composition from tumour biology without cell-type deconvolution (not planned).

## D20. Multiple comparisons
- **Decision:** Confirmatory: two questions (C1, C2), each a single pre-specified statistic, no correction needed beyond that; a Bonferroni note for two tests. Gene-level tests and enrichment: Benjamini-Hochberg FDR. All other analyses are labelled exploratory with no claims of significance.

## D21. Confidence intervals
- **Decision:** Metrics: percentile stratified bootstrap (2,000 resamples) over samples on out-of-fold predictions **pooled within a repeat**; CV folds/repeats are not treated as independent samples (repeats are reported as spread, not as n). Stability: no analytic CI (D14); the permutation null is the reference. External statistics: sample bootstrap (D17). Sample sizes are printed in every figure.
- **Caveat:** pooled out-of-fold predictions share training data across folds, so bootstrap intervals are approximate and probably somewhat narrow.

## D22. Random seed policy
- **Decision:** One master seed (`20260921`) in the config. Split seed for repeat r = master + r. Model `random_state` = master seed. Permutation replicate b uses seeds derived from `numpy.random.SeedSequence(master).spawn(B)`. Every result row stores the seed used. The pipeline is required to be deterministic: two runs of the same config produce identical outputs (tested).

---
## Deviations and open items
- 2026-09-21: Permutation importance compute strategy (D12) and SVM/RF/XGB parameter values (D10, D11) are intentionally undecided; they will be decided before Phase 3/4 code using only discovery data and recorded here.
- Cited sources not re-read in this session are marked as such; they must be verified before entering the paper.
- 2026-09-22 (after the Phase-2 pilot; no decision changed): (i) the pilot showed a flat |coef| profile for LR with C=1 (D11) and low-to-moderate stability (Nogueira about 0.3); a regularisation-strength sensitivity check is a candidate for Phase 6 and must be tuned/compared on discovery data only. (ii) D8 (z-scoring of rare-detection probes) and D7 thresholds remain as pre-specified; the exploratory look at the pilot's top genes was post hoc and is logged in `docs/development_log.md`. (iii) The pooled-metrics bootstrap (D21) costs 40 s for 2 repeats; it will be vectorised before the full experiment (implementation detail, not a design change).
- 2026-09-22 (D11c outcome): the pre-specified regularisation sensitivity returned **materially changed** (Phi(k=25) 0.25 at the primary C=1 vs 0.43-0.48 for the other four C values; performance unchanged). The primary configuration remains C=1 as pre-specified and no C was re-chosen. The result must accompany every reported LR stability value; an optional design extension (extra LR configurations at C=0.1 and C=100 in the model x explainer matrix) awaits the user's decision. See `docs/reports/regularization_sensitivity.md`.

---
## Addendum A. Pre-specification of the remaining pipeline (written 2026-09-22 after the 2x5 pilot, BEFORE the full run, the null run, the freeze and any GSE65194 expression value)

Information available when this was written: the 2x5 pilot (13 pipelines) described in `results/matrix/20260921T204301Z_pilot_matrix_d6a785aa`. Nothing in the pilot changed a design decision; it confirmed that the pipeline is healthy (positive-only rule, deterministic, no leakage). Observations that shape reporting, not design: (i) XGBoost uses about 25 genes, so its two explainers select the same set at k >= 25 (support collapse, not agreement); (ii) linear-model and tree-model top sets almost never overlap; (iii) the t-test top sets overlap tree models (Jaccard 0.12-0.20) but not linear models (about 0).

**A1. Full discovery experiment.** GSE42568 (121 samples), seed 20260921, **20 repeats x 5 folds** (100 folds), the 13 pipelines of `configs/pilot_matrix.yaml` (10 primary model x explainer pipelines, 2 analytic-sensitivity logistic-regression settings C = 0.1 and 100 (coefficients only, labelled sensitivity, never primary), and the Welch-t baseline), k in {10, 25, 50}, R = 10 permutations. Runtime about 60 min sequentially (measured 36 s/fold), run with 2 fold-workers (results are independent of the worker count; tested).

**A2. Permutation-label null (unit of comparison = one 5-fold CV repeat).** Replicate b (b = 0..B-1) = one repeat on labels globally permuted with `SeedSequence([seed, 777, b])` (class counts preserved; expression matrix, preprocessing, feature universe, models, explainers, k and fold structure fixed; stratified splits regenerated on the permuted labels with seed `seed + 100000*(b+1)`). **B = 30** (each replicate costs about 3 min for all 13 pipelines; benchmarked in a 2-replicate pilot first). Observed statistics are computed per repeat by the same function (`repeat_level_stability`), so observed (20 repeats) and null (30 replicates) values are matched in resampling number (5 folds). Report per dimension / group / k: observed mean, median and range across repeats; null mean, median, 2.5-97.5 percentiles and maximum; the difference of medians; and the fraction of observed repeats above the null 97.5th percentile. With B = 30 the smallest attainable empirical exceedance is 1/31, so **no p-value is claimed**; "distinguishable from the null" is a descriptive statement about where the observed distribution sits. Null permutation importance and SHAP use the same settings. Null classification performance is reported as a sanity check (expected ROC-AUC near 0.5).

**A3. Frozen lists** (from the full observed run at freeze time, using only discovery data): `S_cons` (pre-specified consensus rule in `consensus.py`: within-pipeline selection frequency >= 0.5 at k=25 in at least ceil(10/2) = 5 of the 10 primary pipelines), `S_top25` (25 highest mean selection frequency at k=25), `S_top50` (50 highest at k=50); ties by mean rank then gene name. The **continuous internal-stability quantity** for the external analysis is `s_g` = mean selection frequency at k=25 over the 10 primary pipelines for every gene of the eligible universe (0 if never selected). Per gene the frozen probe = max-mean probe on the full GSE42568 (D6). Direction of a gene = sign of its discovery Mann-Whitney effect (cancer vs normal) on the full discovery cohort. All three lists are evaluated externally; none is chosen after seeing external results.

**A4. External evaluation (extends D17; all definitions fixed now).** Analysis units: 130 tumours (23 duplicated tumours = mean of the two arrays) + 11 healthy; cell lines excluded; no cohort-level normalisation across cohorts.
- Gene effect: Mann-Whitney AUC (cancer vs normal) in each cohort, e = AUC - 0.5, at the frozen probe. Aligned external effect `r_g = sign(e_disc) * e_ext`. Direction consistency = fraction of genes with sign(e_ext) = sign(e_disc), with exact binomial (Clopper-Pearson) interval; genes are correlated, so this is descriptive. Standardised effect size: Hedges' g is also reported for the frozen lists (within-cohort, scale-free).
- Rank agreement: Spearman correlation of e_disc and e_ext (a) across all universe genes and (b) within each frozen list.
- **C1:** Spearman(s_g, r_g) over all eligible universe genes; secondary: restricted to s_g > 0. **C2:** rank-regression of r_g on rank(s_g) and rank(|e_disc|); the coefficient of rank(s_g) (and the partial Spearman) is the incremental value of stability beyond discovery effect size. Uncertainty: 1,000 stratified bootstrap resamples of the EXTERNAL samples (discovery quantities fixed); percentile 95% intervals; no gene-level p-values are used as evidence.
- Signature-level: score per sample = mean within-cohort z-score of frozen "up" genes minus mean z of "down" genes (z from each cohort's own unlabeled mean/SD), reported as AUC with a stratified bootstrap CI for each frozen list, in both cohorts (discovery value is in-sample and labelled as such).
- Sample-level: L2 logistic regression (C = 1, balanced) fitted on the FULL discovery cohort restricted to the S_top25 genes (discovery z-scoring), applied to external samples after within-cohort z-scoring; ROC-AUC, average precision (normal), balanced accuracy, sensitivity, specificity with bootstrap CIs. Secondary because of the scale mismatch.
- Sensitivity (descriptive, not optimisation): duplicate handling with replicate array A only; C1/C2 recomputed per subtype (Luminal A+B, HER2, TNBC each vs healthy).
- Access control: the external loader refuses to run unless `docs/freeze/discovery_freeze.json` exists and its recorded commit is an ancestor of HEAD.

**A5. Enrichment.** Foreground = `S_top50` (primary); `S_cons` only if it has >= 10 genes. Libraries (via `gseapy.get_library`, Enrichr): `GO_Biological_Process_2023`, `Reactome_2022`, `MSigDB_Hallmark_2020`; library GMT text hashed and dated. Own hypergeometric test; term size 5-500 after restriction; **background = eligible gene universe intersected with the library's genes** (foreground restricted the same way); Benjamini-Hochberg within each library. Executed once. Reported: n foreground / background, overlap, odds ratio, adjusted p, top 10 terms per library plus the count with adjusted p < 0.05.

**A6. Tissue composition.** Panel (D19): ADIPOQ, PLIN1, FABP4 (primary); LEP, LPL, CFD, ADH1B, CIDEC, PLIN4, CD36, GPD1 (extended, exploratory). Checks: membership and selection-frequency percentile of panel genes among the universe; adipose-panel score AUC in both cohorts and its Spearman correlation with the frozen signature score within each class and cohort; signature AUC after regressing the panel score out (residual score); **composition ablation**: repeat the 2x5 matrix with all panel genes removed from the universe and compare Phi, top-25 overlap with the full run and performance. Panel absence from the lists is reported equally.

**A7. Processing-date sensitivity (D19b).** `dec2004` subset (samples processed 2004-12; expected 97 tumours + 8 normals) and a size-matched random control (8 normals + 97 tumours drawn from the full cohort, seed 1), each 5 repeats x 4 folds (8 normals allow 4 stratified folds), same 13 pipelines. Comparisons against the full-cohort run: performance; per-pipeline resampling Phi; Spearman correlation of universe selection-frequency vectors and Jaccard of top-25 lists between full / dec2004 / control. Interpretation rule: a stability or overlap change of the dec2004 subset counts as date-related only if it exceeds the change seen for the size-matched control; with 8 normals all estimates are reported as unreliable.

**A8. Access and reporting rules.** No number derived from GSE65194 may influence A1-A3, A5-A7. Everything in A4 is reported whether or not it looks favourable. Failure to replicate is a valid result.
