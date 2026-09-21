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
