# Gap analysis (reconnaissance, 2026-09-20)

**Evidence level:** ~15 sources located via web search; ~8 read at abstract/summary level through fetches; none read in full text; no code executed. Statements below are "not found in this search", never "does not exist". Re-check before any novelty claim in the paper.

## 1. Hypotheses from the project draft, tested against the evidence

| Draft assumption | Verdict | Evidence |
|---|---|---|
| "Stability analysis is what differentiates BioXplain" | **Not differentiating.** Already done, often better | Sirohi 2026, Stabl 2024, OmicSelector, RobustModelMaker 2026, Ein-Dor 2005 |
| "Leakage-safe pipeline" is a contribution | **No - table stakes** | OmicSelector, RobustModelMaker, OncoResolve all claim it |
| "Cross-model consensus" is new | **Partly done**, formulas mostly ad hoc | OncoResolve (>=2/3 vote), Shafik 2026 (arbitrary 0.4/0.3/0.3), rank-aggregation literature |
| "SHAP explains the model" is enough | **Weak** - SHAP rankings are unstable under collinearity/seed | Caraker 2026, Claborne 2026, Zuo 2026 |
| "High accuracy on GSE42568" is informative | **No** - near-ceiling accuracy is routine | Kim (IEEE 2025) reports 0.99 on GSE42568 |
| "External validation adds strength" | **Yes but already appears in the closest competitors** | OncoResolve (4 cohorts), Sirohi (CPTAC) |
| "Biological interpretation via pathways" | **Yes, but fragile** | Tissue-composition confounding (Elloumi 2011); Sirohi got 136 GO terms from unstable lists |

## 2. What remains genuinely open (candidate gaps)

**G1 - Stability of the *explanation itself*, jointly over model x explainer x resample, with chance-corrected metrics.** Prior work measures stability of a selector (Sirohi: RF/SVM Gini+permutation; Stabl) or of SHAP for one model family (Claborne: DL; Caraker: boosting simulations; Zuo: NN). We found no study that tabulates, on the same gene-expression task, LR/SVM/RF/XGBoost x (coefficients, TreeSHAP/LinearSHAP, permutation) with Nogueira Phi + CIs and a permuted-label null.

**G2 - Does internal stability predict external replicability?** Papers assume stable = more trustworthy. We found no test of that assumption across two independent breast cohorts: *are genes with high selection frequency in GSE42568 more likely to be differentially expressed / predictive in GSE65194 than low-frequency genes?* Either answer is publishable-as-a-finding; it converts a workflow into an empirical claim.

**G3 - Is any of it better than a plain t-test/limma ranking?** Haury 2011 found simple filters competitive on breast data. Any modern pipeline should be benchmarked against that baseline; many application papers omit it.

**G4 - Reusable Python tool.** Python tools exist for stable selection (RobustModelMaker, Stabl) but not for *explanation-agreement reporting + enrichment + cohort replication*. This is an engineering contribution only; real but modest.

**G5 - Honest reporting of tissue-composition confounding.** Tumour-vs-normal breast: normal tissue is adipocyte-rich. Predicted (not yet measured) consequence: top "predictive" genes include adipose markers (e.g. ADIPOQ, PLIN1, FABP4, ADH1B - plausible, from literature snippets only). These would be *stable, replicable and biologically 'meaningful'* yet reflect sample composition, not tumour biology. Testing this is a cheap, sharp interpretation contribution.

## 3. Proposed contribution (defensible wording)

> "We present an open-source, tested Python workflow and an empirical study of how consistently different model-explainer pairs identify the same genes in breast-tissue expression data, whether that internal consistency predicts replication in an independent cohort, and how far pathway enrichment on such lists can be trusted given tissue-composition confounding."

Not claimed: first, novel algorithm, causal or clinical biomarkers, or superior accuracy.

## 4. Minimum viable research product (MVP)

1. Loader/validator for GEO series matrices + GPL570 annotation (probe -> gene symbol).
2. Leakage-safe repeated stratified CV (fold-contained filtering/scaling/selection).
3. Models: L1/elastic-net LR (interpretable baseline), linear SVM, Random Forest, XGBoost. Dropped/no: deep nets, extra models ("not worth the time").
4. Explainers: coefficients (linear), SHAP (Linear/Tree), permutation importance (all) + **univariate baseline** (Welch t / moderated t).
5. Stability: Nogueira Phi (+CI), Jaccard, Kuncheva at k in {10,25,50}; rank correlation; **permuted-label null**.
6. Consensus: selection frequency + median rank across (model x explainer x resample); documented, no invented weights; compare 2 simple alternatives.
7. Freeze list from GSE42568 -> evaluate on GSE65194 (cell lines excluded; no refitting of anything on it).
8. Enrichment via GSEApy with the *background = all tested genes* (not whole genome), Enrichr library versions logged; plus adipose-marker check for G5.
9. pytest suite incl. deliberate leakage tests; results tables/figures; paper; README.

## 5. Explicitly NOT built (time)
Web UI/frontend; deep learning; multi-omics; raw CEL/RMA re-processing; survival/subtype tasks; TCGA cross-platform validation; hyper-parameter search beyond a small fixed grid; own novel scoring formula; SHAP interaction values; CLI (only if week-2 time remains).

## 6. Known risks
- **Trivial task**: tumour vs normal is near-perfectly separable; accuracy tells us nothing (report it, do not optimise it). Interest = stability/replication.
- **n_normal = 17 (GSE42568), 11 (GSE65194 per literature)**: metric CIs wide; stratified folds hold ~3 normals each; use repeated CV; AUC on external set will have very wide CI.
- **Same platform (GPL570) for both cohorts**: transfer is easy, so "independent" means independent cohort/batch, *not* independent technology. Series-matrix normalisation status is **unverified** - inspect headers first; possible cross-series scale shift (Sirohi found external transfer collapsed without normalisation); check for shared GSM samples between series.
- **Correlated genes** distribute SHAP/permutation credit arbitrarily -> report gene clusters; consider cluster-level stability.
- **Enrichr is a live web service**; results drift by library release. Log date + library name; consider local GMT for reproducibility.
- Sample counts for GSE65194 (130 tumour / 11 normal / 14 cell lines) come from secondary papers, **not** GEO (GEO web pages blocked by CAPTCHA). Verify from the matrix file.
