# BioXplain: an empirical study of explanation stability and cross-cohort replication in breast-tissue gene expression

*Research report (foundation for a manuscript). All numbers are taken from saved result files; see `docs/results/results_summary.md` and the manifests.*

## 1. Introduction

**Biological motivation.** Gene-expression profiles separate tumour from normal breast tissue almost perfectly, and machine-learning models are routinely used to nominate "important genes". Whether those genes are reproducible properties of the disease, of the sample composition, or of the analytic choices of the modeller is a separate and less studied question.

**Computational problem.** Expression matrices have tens of thousands of correlated features and, here, only 121 samples (17 normal). In this regime a model's top-ranked genes can change substantially with the training subset, the model class, the explanation method and even a regularisation setting, while predictive accuracy stays at ceiling. Explanation methods measure different quantities (a linear weight, an additive attribution of the model output, a loss increase under permutation) and are known to disagree (Krishna et al. 2022, snippet-level record in `docs/research/literature_update_2026.csv`).

**Research gap (evidence-based, not a novelty claim).** Stability of feature selection is a long-known problem (Ein-Dor et al. 2005; Haury et al. 2011; Nogueira et al. 2018) and stable, leakage-safe selection tools exist (Stabl, OmicSelector, RobustModelMaker; `docs/research/open_source_review.md`). What we did not find in our (limited, mostly abstract-level) search is an empirical study that, on one transcriptomic task, (i) measures chance-corrected stability separately across resampling, model, explainer and their combination with a permutation-label null under an identical pipeline, (ii) includes a simple univariate baseline, and (iii) tests whether internal stability predicts replication in an independent cohort **beyond discovery effect size**, together with an explicit check of tissue-composition confounding. This report is such an empirical evaluation with a tested, reproducible workflow; it does not claim a new algorithm or that stability is a new concept.

**Research question.** How consistently do machine-learning explanations identify gene-expression features across models, explainers and resampling, and does explanation stability predict replication on an independent breast-cancer cohort?

## 2. Methods

### 2.1 Datasets
- **Discovery: GSE42568** (NCBI GEO; Affymetrix HG-U133 Plus 2.0, GPL570; GC-RMA): 121 samples, 104 breast cancers and 17 normal breast tissues, 54,675 probes. Labels are reproducible from four concordant metadata fields.
- **External: GSE65194** (GPL570; GC-RMA followed by submitter batch/hybridisation correction): 178 arrays = 130 unique tumours (23 measured twice), 11 healthy tissues, 14 cell lines. Analysis units: 130 tumours (duplicate arrays averaged per probe) + 11 healthy; cell lines excluded. No shared GSM with the discovery cohort. Because both cohorts use the same chip, this is independent-cohort, not independent-technology, validation.
- **Annotation: GPL570.annot (2016).** 42,894 probes map to exactly one gene; 42,892 of them are present in both series matrices and form the feature universe (20,848 genes).
- Reconnaissance findings that shaped the design (`docs/data_reconnaissance.md`): different numeric scales (a hard floor at 2.3128 in 28.5% of discovery values; none externally), class-blocked GEO order, partial confounding of processing date and title prefix with class in discovery, technical duplicates and cell lines externally.

### 2.2 Leakage prevention
Every learned operation (floor-detection expression filter, probe-to-gene collapse by maximum training mean, z-scoring, model fitting, every explainer, the Welch-t baseline) is fitted on the training fold only. The validation fold enters only `transform`/`predict`. Tests: a fit-index spy, an invariance test that corrupts the validation rows and requires every learned artefact of every pipeline to stay bit-identical, a deliberately leaky runner that must be detected, a demonstration of selection leakage on noise labels, and mutation tests on the real code. Splits are shuffled, stratified repeated 5-fold CV (GEO order is class-blocked).

### 2.3 Models (fixed, pre-specified, class-balanced, no tuning)
L2 logistic regression (C = 1), linear SVM (C = 1), random forest (500 trees), XGBoost (300 trees, depth 3, `n_jobs = 4` pinned). Two logistic-regression settings, C = 0.1 and C = 100, are analytic-sensitivity settings and are never primary. Primary metrics are ROC-AUC and normal-class average precision with stratified bootstrap intervals; balanced accuracy, sensitivity and specificity depend on each model's fixed decision threshold.

### 2.4 Explainers (three different quantities; none is causal)
- **Coefficients** (LR, SVM): the fitted linear weight on z-scored genes, conditional on all other genes.
- **SHAP**: mean absolute additive attribution of the model output over all training-fold samples (LinearExplainer for LR/SVM, TreeExplainer for RF/XGBoost); one common explanation population for every model.
- **Permutation importance**: exact (no top-N filter) loss increase under permutation on the training fold, class-balanced Brier loss, R = 10; computed exactly with three shortcuts (analytic for linear models, unused genes have exactly zero importance for trees, per-tree recomputation for RF) verified against brute force to 1e-12.
- **Welch-t baseline**: |t| on the training fold, ranking only; no p-value is interpreted.
A gene is eligible for a top-k set only if its attribution is strictly positive, so sparse models never receive alphabetical filler genes.

### 2.5 Stability
Sets = top-k genes, k = 10, 25, 50. Nogueira's chance-corrected Φ (JMLR 2018, Definition 4; checked against the paper and stabm reference values, equal to Kuncheva's index for constant size) is the primary measure, with Jaccard and Kuncheva (constant-size sets only) as secondary. Four dimensions are reported separately, never combined: **resampling** (one pipeline across the 5 folds of a repeat), **model** (fixed explainer, different models, within the same fold), **explainer** (fixed model, different explainers, within fold) and **combined** (all ten primary pipelines within fold), plus agreement with the t-test baseline and across the logistic-regression sensitivity settings. The unit of comparison is one CV repeat. Nogueira's analytic confidence interval assumes independent sets and is not valid for overlapping CV training sets; spread across the 20 repeats is reported instead.

### 2.6 Permutation-label null
Each of 30 replicates permutes the class labels globally (class counts preserved) and runs the identical pipeline (same preprocessing, feature universe, 13 pipelines, k values) for one 5-fold repeat, so observed and null statistics are computed by the same function and are matched in resampling number. Results are descriptive (null percentiles versus observed repeats); no p-value is claimed (the smallest attainable exceedance with 30 replicates is 1/31).

### 2.3b Consensus and freeze
No composite score is used. For each gene: selection frequency per pipeline, number of pipelines/models/explainers with frequency ≥ 0.5, median and best rank. Pre-specified consensus rule: frequency ≥ 0.5 at k = 25 in at least 5 of the 10 primary pipelines. Frozen lists: `S_cons`, `S_top25`, `S_top50`. The discovery configuration, code commit, data hashes and frozen gene table are recorded in `docs/freeze/` before the external cohort is opened; the external loader refuses to run otherwise.

### 2.7 External replication (scale-free)
Gene effect = Mann-Whitney AUC − 0.5 in each cohort at the frozen probe (invariant to monotone rescaling). Aligned external effect r_g = sign(discovery effect) × external effect. **C1**: Spearman correlation between internal stability s_g (mean selection frequency, k = 25, primary pipelines; zero for never-selected genes) and r_g across all eligible genes. **C2**: association of s_g with r_g after accounting for the size of the discovery effect |e_disc| (rank regression coefficient and partial Spearman), i.e. whether stability carries information beyond effect size. Uncertainty by stratified bootstrap over external samples (1,000 resamples). Signature level: within-cohort z-scored up-minus-down score with bootstrap AUC; sample level: logistic regression fitted on discovery restricted to `S_top25`, applied after within-cohort standardisation. Sensitivity: replicate array A only; per-subtype analyses.

### 2.8 Enrichment and tissue composition
Own hypergeometric over-representation test; background = eligible gene universe ∩ library genes; terms with 5–500 genes; Benjamini-Hochberg within library; libraries GO Biological Process 2023, Reactome 2022, MSigDB Hallmark 2020 (downloaded once, hashed, dated). Tissue-composition hypothesis: adipocyte marker panel (ADIPOQ, PLIN1, FABP4 primary; extended panel exploratory), its score in both cohorts, correlation with the frozen signature, residual signature AUC, and a composition-ablation rerun with the panel genes removed.

### 2.9 Processing-date sensitivity
Dec-2004-only subset (samples processed 2004-12) and a size-matched random control, 5 repeats × 4 folds, compared with the full cohort; a date-related change is claimed only if it exceeds the change seen in the size-matched control.

### 2.10 Reproducibility
Seeds, environment, data hashes, resolved model configurations and code commit are stored in every manifest (`docs/reproducibility.md`). Two runs are bit-identical, including across hash seeds. 159 tests.
