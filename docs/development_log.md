# Development log

Format: dated entries; problems are recorded as Problem / Expected / Actual / Diagnosis / Fix / Validation / Impact / Lesson; decisions as Decision / Why / Evidence / Impact / Next.

## 2026-09-20 - Reconnaissance

**Decision/result:** Ran project reconnaissance before any implementation.
- **State found:** `D:\Bioxplain` had folder skeleton only (`data/raw`, `data/processed`, `docs`, `notebooks/01_data_exploration.ipynb` [0 bytes], `results/*`, `src/`), a `.venv` (Python 3.11.9) and the draft `BioXplain_Research_Project_Draft (1).docx`. No Git repo, no remote. `gh` CLI not installed.
- **Environment:** venv contains numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.2.0, shap 0.51.0, gseapy 1.3.1, matplotlib 3.11.1, seaborn, scipy 1.17.1, jupyterlab. **Missing:** pytest (needed). System Python 3.14 is the default `python`; project uses the 3.11 venv.
- **Why it matters:** Environment is sufficient for the whole planned pipeline once pytest is added.
- **Evidence:** `docs/research/literature_review.csv`, `open_source_review.md`, `gap_analysis.md`.

**Problem (process):** GEO series web pages (`ncbi.nlm.nih.gov/geo/query/acc.cgi`) returned a CAPTCHA to the fetch tool. Not bypassed. Used GEO FTP directory listings instead (file names and sizes verified: `GSE42568_series_matrix.txt.gz` 22.6 MB; `GSE65194_series_matrix.txt.gz` 76.5 MB; `GPL570.annot.gz` 8.1 MB). Sample counts were then taken from secondary literature; they were **re-verified against the downloaded matrices' own metadata on the same day (see next entry)**.

**Decision:** The draft's headline differentiator ("stability + consensus + leakage-safety") is not novel; reframed as an empirical study (explanation stability across model x explainer, and whether it predicts replication in an independent cohort) plus a tested tool.

## 2026-09-20 - Data download and reconnaissance (checkpoint 1)

**Result:** Downloaded GSE42568 and GSE65194 series matrices and the GPL570 annotation with `curl.exe` (about 34 s total); all pass a full gzip decompress; series-matrix sizes equal the FTP listing sizes. Full findings: `docs/data_reconnaissance.md`; numbers: `results/metrics/data_recon.json`; reproducible via `scripts/data_recon.py`.
- GSE42568: 54,675 x 121, 104 cancer / 17 normal, labels unambiguous (4 fields agree).
- GSE65194: 178 arrays = 153 tumour arrays (130 unique tumours + 23 duplicated) + 11 healthy + 14 cell lines. Counts from secondary literature confirmed.
- No GSM/title/column overlap; probe IDs compatible (54,673 shared).
- **Unexpected findings (no modelling done yet):** the cohorts have different numeric scales (GC-RMA floor at 2.3128 in GSE42568 vs batch-corrected, floor-free, negative-containing values in GSE65194); GSE65194 replicate arrays are not averaged; GSE42568 GSM order is class-blocked; processing-date and title-prefix metadata are partly/perfectly aligned with class in GSE42568; 28.5% of GSE42568 values equal the floor.
- **Impact:** external replication must use scale-free endpoints; replicate averaging and cell-line exclusion are needed; shuffled stratified splits must be tested; probe-to-gene collapse and expression filtering must be fold-internal.
- **Next action:** wait for the user's decision on the a-priori rules listed in the report (section 10), then Steps 3-10.

**Problem (process):** First recon script crashed printing a non-ASCII character (Windows cp1252 console). Fixed with `PYTHONIOENCODING=utf-8`; no effect on data.


## 2026-09-21/22 - Phase 1-2: methodology, package skeleton, first vertical slice

**Decision/result:** Wrote `docs/methodology_decisions.md` (22 decisions, all made before any model was fitted or any GSE65194 gene-level value was seen; the external-replication plan C1/C2 and the permutation-null design are pre-specified there). Built the `bioxplain` package and ran the first real experiment (GSE42568, 2 repeats x 5 folds, L2 logistic regression, |coef| ranking, Nogueira/Kuncheva/Jaccard). Code commit `192f1a7`; run `results/experiments/20260921T190737Z_slice_logreg_coef_1cb7b8e9` (manifest: `dirty=false`).
- Stability metric definitions verified at the primary source (Nogueira et al. JMLR 2018 Def. 4, Theorem 5, Appendix A Table 6) before implementation. On real data Nogueira and Kuncheva agree to all printed digits, as Theorem 5 requires.
- 60 pytest tests pass (about 26 s). Leakage mutation test: moving the preprocessing fit onto all samples inside `run_fold` made 2 tests fail; restored code passes.
- Real-data determinism: two runs with different `PYTHONHASHSEED` are bit-identical (rankings, predictions, stability, pooled metrics).
- Runtime: load 3.9 s; CV loop 13.3 s (about 1.3 s per LR fold); pooled metrics + bootstrap 40 s (dominant, needs optimisation before scaling).

**Problem:** first run of the new test suite: `test_collapser_choice_is_learned_from_training_rows_only` failed. Expected: probe pA chosen on training rows and pB on all rows. Actual: pA both times. Diagnosis: my toy data gave pA and pB identical means over all rows (5.0), so the alphabetical tie-break chose pA (test bug, not code bug). Fix: pA over all rows = 4.5. Lesson: a leakage test needs a case where the leaked and clean outcomes truly differ.

**Problem:** `test_dependence_on_k_for_fixed_relative_overlap` failed for odd k=25. Actual code result (0.5077) was right; my expected value assumed an overlap of k//2 = 12 while the shifted sets overlap in 13. Fix: independent hand-computed expected values. Lesson: do not re-derive the expectation with the same formula as the implementation.

**Problem:** the toy "low-expression" genes were detected in about 20% of samples, above the filter threshold, so the filter test would have been wrong. Fixed the toy generator (mean 0.8 -> about 0.6% detected).

**Problem (real bug, scientific-software relevant):** the real-data determinism check showed Nogueira differing in the 16th decimal between two runs (4 of 9 stability values). Diagnosis: `sum()` over a `Counter` built from a `frozenset` union; string-hash randomisation changes iteration order and float addition is not associative. Fix: `math.fsum`. Validation: my first regression test PASSED on the old code (it could not detect the bug); I then reproduced the bug directly (4 different values over 8 hash seeds), rebuilt the test around subprocesses with different `PYTHONHASHSEED`s, verified it FAILS on the old code and PASSES on the fix, and re-ran the real experiment with different hash seeds (bit-identical). Impact: none on conclusions (differences about 1e-16), but strict reproducibility required it. Lesson: a regression test must be shown to fail on the bug.

**Result (first real numbers, pilot; not for conclusions):**
- Performance is near ceiling, as predicted for tumour vs normal: pooled out-of-fold ROC-AUC 0.999 / 0.997 (95% bootstrap CI 0.997-1.000 / 0.990-1.000); balanced accuracy 0.94 / 0.91 because specificity at the fixed 0.5 threshold is 0.88 / 0.82 (2-3 of 17 normals called cancer) while sensitivity is 1.00. Accuracy would be uninformative (H5 setting).
- Stability of the top-k coefficient rankings is low to moderate: Nogueira 0.275 / 0.322 / 0.317 for k = 10 / 25 / 50 (Kuncheva identical; Jaccard 0.17-0.20). At k=25 there are 130 distinct genes across 10 runs, 1 in 10/10 runs, 6 in at least 8/10, 92 in only one run. So, in this pilot, a near-perfect classifier has a poorly stable top-k explanation (an observation for H5, single model/explainer, no null yet, not a finding).
- |coef| is almost flat across ranks (median 0.0074 at rank 1, 0.0056 at rank 25, 0.0045 at rank 200): with C=1 on 20k z-scored genes the solution is close to a mean-difference direction, so many genes are near-ties (consistent with the D11 note). Regularisation-strength sensitivity is worth a small check later.

**Exploratory diagnostic (post hoc, after seeing results, discovery cohort only; NO design change made):** worried that rare-detection probes inflated by z-scoring (D8) dominate the top lists. Result: rare-detection probes (<20 of 121 samples) are 16.3% of the universe and 21.5% of ever-selected top-25 probes, a mild over-representation, not dominance. Several frequently selected genes are detected in nearly all normals but few cancers (e.g. ABHD1: 16/17 normals vs 8/104 cancers); this is the class-specific detection pattern that the minority-based filter (D7) is designed to keep. Other frequent genes are detected in all samples and carry a negative coefficient (higher in normal, 13 of 14 genes selected in at least 5/10 runs). Some top entries are LOC/pseudogene/antisense identifiers from the 2016 annotation. Prefix/date cannot be separated from class in this cohort (recon F4), so these observations are not evidence about biology or batch; the composition and date analyses stay as pre-specified (D19).
- **Impact on design:** none. Open for later sensitivity: z-scoring of rare-detection probes (D8), the C=1 regularisation strength (D11), 0.5 decision threshold policy for specificity (not a ranking issue).
- **Next action:** review by the user, then Phase 3 (remaining models), after optimising the bootstrap step.

## 2026-09-22 - Optimisation of the metrics/bootstrap step (commit separate from science)

**Result:** Vectorised the stratified bootstrap (`evaluation/bootstrap.py`): resamples are drawn with the same generator calls in the same order as the reference `metrics.bootstrap_ci`, then ROC-AUC, average precision (normal class) and balanced accuracy are evaluated for all 2,000 resamples at once from resample-multiplicity vectors. The reference implementation stays in the code base as the correctness oracle. The statistical definition is unchanged.
- **Before/after (2 repeats x 3 metrics x 2,000 resamples, pilot predictions):** reference 18.4 s (17.7-19.5 s in three standalone timings of the runner function) -> 0.167 s (x110). End-to-end pilot: 57.6 s -> 9.3 s; metrics+stability stage 40.3 s -> 0.26 s. The 40 s recorded in the pilot manifest was inflated by machine load (standalone timing of the same function was 18 s); the CV loop also ran faster (13.3 -> 7.0 s) although unchanged, which is load variance and NOT attributable to this change.
- **Equivalence:** max |reference - vectorised| CI endpoint 1.1e-16; max |committed pilot CI - vectorised| 2.2e-16; documented tolerance 1e-12. Rankings, predictions and stability of a full re-run are bit-identical to the committed pilot; output identical over PYTHONHASHSEED 0-3 (`scripts/benchmark_metrics.py`).
- **Tests:** 8 new tests (weights structure and RNG-order equality with the reference, agreement with scikit-learn on explicit resamples with and without ties, zero-weight thresholds, known values). Mutation check: removing the tie term (+0.5 for tied pairs) made 3 tests fail. Suite: 68 pass.
- **One deliberate improvement:** the bootstrap balanced accuracy now uses the stored predicted labels instead of re-thresholding the score at 0.5; identical for logistic regression (verified for all pilot predictions), and required for SVM whose decision threshold is not a probability.

## 2026-09-22 - Permutation-importance feasibility and strategy decision (docs only)

**Decision/result:** Investigated exact permutation importance (PI) before implementing it. Benchmark on one real GSE42568 training fold (`scripts/benchmark_pi_feasibility.py`; 96 samples, 20,087 genes, R=10): RF (500 trees) re-prediction 111 ms -> naive exact PI over all genes about 22,000 s per fold; XGBoost (300 trees, depth 3) 46 ms -> about 9,000 s per fold; both infeasible for 100+ folds x several repeats. **The fitted tree models are sparse:** the forest splits on 1,247 of 20,087 genes (6.2%, mean 7 nodes/tree) and XGBoost on only 24; LinearSVC fits in 0.2 s with all 20,087 coefficients non-zero.
- **Decision (D12b):** Option C = exact PI with exactness-preserving shortcuts (analytic for linear models; unused genes have exactly zero PI for trees; per-tree recomputation for RF), evaluated on the training fold with a class-balanced Brier loss, R=10. Options B (t-test pre-filter) and D (different importance) rejected because they change the candidate universe or the quantity being compared. No arbitrary top-N. The training-fold-vs-held-out choice is flagged to the user as a judgement call.
- **Unexpected finding with methodological consequence (D13b):** sparse models have exactly zero attribution for most genes; alphabetical tie-breaking would fill top-k with identical arbitrary genes across folds and manufacture stability. Pre-specified before any tree-model result: only genes with strictly positive importance are eligible for top-k sets; set sizes may be < k; Kuncheva only for constant sizes. LR pilot unaffected (dense coefficients).
- **Also pre-specified before any result (D11b/D11c):** fixed parameters for SVM/RF/XGBoost with class balancing; metric hierarchy (ROC-AUC and normal-class AP primary); the logistic-regression regularisation grid C in {0.01, 0.1, 1, 10, 100}, the design (5x5 CV) and a qualitative robustness criterion in the paper's own Phi bands (<0.40 poor / 0.40-0.75 intermediate-to-good / >0.75 excellent, Nogueira et al. Table 3, text checked).
- **Impact on research question:** none. XGBoost's tiny support (24 genes) is itself an observation about boosted-tree attributions that the stability study must report honestly.

## 2026-09-22 - Regularisation sensitivity (D11c): result and investigation

**Result:** Pre-specified run (code `6863ab1`, clean tree; run `results/sensitivity/20260921T194825Z_sensitivity_logreg_C_d8b34927`). Verdict by the pre-specified rule: **materially changed**. Phi(k=25): C=0.01 0.481, 0.1 0.431, **1 0.248**, 10 0.477, 100 0.476; performance unchanged (pooled ROC-AUC 0.996-0.997, every repeat >= 0.990). The non-monotone shape (C=1 lowest) was unexpected.
- **Investigation (post hoc, exploratory):** the solver converged everywhere; C=10 and 100 are effectively the same early-stopped solution; C in {0.01, 0.1} and C in {10, 100} are two regimes with almost disjoint top-25 sets (Jaccard about 0.10), and C=1 is the transition with the flattest coefficient profile. Details in `docs/reports/regularization_sensitivity.md`.
- **Impact:** the pilot's "low-moderate stability" statement is regime-dependent and must be reported with this sensitivity; explanations depend strongly on a within-model analytic choice. No decision changed (primary C=1 stays, as pre-specified). One design extension (additional LR configurations) is offered to the user.
- **Lesson:** the 2-repeat pilot value (0.32) was optimistic compared with 5 repeats (0.25 at C=1; per-repeat 0.10-0.32): stability estimates need enough repeats and their spread must be shown.

**Problem/test decisions:**
1. The criterion k was hard-coded to 25 in the implementation, which crashed on the toy config (KeyError). Category A (implementation lacked a parameter and validation): made `criterion.k` explicit, validated it against `top_k`, and added a test.
2. `(Series == pytest.approx(1.0)).all()` did not compare element-wise as intended (category B, test bug). I first verified that the underlying values were exactly 1.0, then replaced the assertion with an explicit tolerance check that still fails on wrong values.
