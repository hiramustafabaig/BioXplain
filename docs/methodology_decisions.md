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
- **Status:** (1) implemented in the pilot slice; (2)-(4) [DEFERRED] Phase 3.
- **Alternatives:** elastic-net LR, kernel SVM, kNN, neural networks (not planned).

## D11. Hyperparameter policy **[DEFERRED]**
- **Decision:** No search in the primary analysis. Fixed, pre-specified configurations. Pilot values: LR: L2, `C=1.0`, `class_weight="balanced"`, lbfgs, `max_iter=5000`. Values for the other models are fixed in `configs/` in Phase 3 from library defaults plus class balancing, and not tuned. If a sensitivity check on regularisation strength is run, it is tuned or compared only inside training data and reported as sensitivity.
- **Rationale:** Tuning inside 97-sample training folds with 4-5 normals per validation fold is itself unstable and would add a second source of variability to the stability question. Note (theory): for strongly regularised LR the coefficient vector approaches a mean-difference direction, so LR-coefficient rankings may resemble univariate rankings (hypothesis H3); this is reported, not assumed.
- **Outcome timing:** Before. **Bias:** Default configurations may be suboptimal; accepted because the goal is methodological comparison.

## D12. Explainer set **[DEFERRED]**
- **Decision:** (a) LR coefficients (`|coef|` on z-scored genes; signed value retained); (b) SHAP: TreeSHAP for RF/XGBoost, LinearSHAP for linear models, mean |SHAP| over **training-fold samples**; (c) permutation importance (drop in ROC-AUC). (d) t-test baseline (D16b).
- **They are not the same quantity:** coefficients are the model's linear weights (conditional effect given other genes); SHAP is an additive attribution of the model output, averaged over samples (reflects model behaviour on the reference distribution, affected by correlated features); permutation importance is the loss increase when one gene is broken (depends on the evaluation data and on correlated genes, which can mask importance). None is causal.
- **Open compute issue (Phase 4):** exact permutation importance over ~15-20k genes for tree models is expensive. If it is infeasible a principled reduction will be proposed and documented (not silently applied); a candidate reduction must not be chosen using validation-fold or external information.
- **Note:** LinearSHAP on z-scored features is proportional to |coef| times mean |z|, so LR-coefficient and LR-SHAP rankings are expected to be almost redundant; this will be reported and not treated as independent evidence.
- **Outcome timing:** Before.

## D13. Top-k definitions
- **Decision:** k in {10, 25, 50} genes, taken by rank (rank 1 = largest importance; ties broken by gene symbol). The top-200 of every run is stored with rank and importance; k=25 is the pre-specified headline value, 10 and 50 are reported alongside.
- **Rationale:** Small enough to be a "signature", large enough for overlap statistics. k is not tuned on any outcome.
- **Outcome timing:** Before.

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
