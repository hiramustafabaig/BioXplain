# Phase 2 report: first vertical slice (GSE42568, logistic regression, coefficient ranking)

Code commit `192f1a7` (clean tree). Run: `results/experiments/20260921T190737Z_slice_logreg_coef_1cb7b8e9/` (config sha256 prefix `1cb7b8e9`). Pilot size (2 repeats x 5 folds); **no conclusions are drawn**. No permutation null yet, one model, one explainer, discovery cohort only; GSE65194 was not opened.

## Pipeline exercised
GSE42568 (121 samples, 104 cancer / 17 normal, universe 42,892 probes / 20,848 genes) -> repeated stratified 5-fold CV (seed 20260921 + repeat) -> per training fold: floor-detection filter -> probe-to-gene max-mean collapse -> z-scoring -> L2 logistic regression (C=1, balanced) -> |coef| ranking (top 200 stored) -> Nogueira / Kuncheva / Jaccard on top-k sets.

## Verification
- 60 pytest tests pass (26 s). Leakage: fit-index spy, validation-fold perturbation invariance, deliberately leaky runner detected, noise-label selection-leak demonstration, real-code mutation test (2 tests fail when the fit is moved onto all samples).
- Stability formulas checked against Nogueira et al. (2018) text, hand calculation, stabm documented values, Theorem 5 (Phi = mean pairwise Kuncheva for constant k; holds on the real data to all printed digits), and the theoretical minimum -1/(M-1).
- Real-data determinism: two runs with different `PYTHONHASHSEED` give bit-identical rankings, predictions, stability and metrics.

## Results (pilot)
Per-fold / pooled out-of-fold (each repeat n = 121; 95% stratified bootstrap CI over samples):

| repeat | ROC-AUC (CI) | AP normal | balanced acc. (CI) | sensitivity | specificity |
|---|---|---|---|---|---|
| 0 | 0.999 (0.997-1.000) | 0.997 | 0.941 (0.853-1.000) | 1.000 | 0.882 (15/17) |
| 1 | 0.997 (0.990-1.000) | 0.984 | 0.912 (0.824-1.000) | 1.000 | 0.824 (14/17) |

Stability of the top-k gene sets over the 10 runs (d = 20,848; the alternative universe of 20,335 genes eligible in any fold gives the same values to 3 decimals):

| k | Nogueira Phi | Kuncheva | Jaccard | distinct genes in 10 sets | per-repeat Phi (2 values) |
|---|---|---|---|---|---|
| 10 | 0.275 | 0.275 | 0.173 | 50 | 0.220 / 0.340 |
| 25 | 0.322 | 0.322 | 0.201 | 130 | 0.311 / 0.319 |
| 50 | 0.317 | 0.317 | 0.193 | 251 | 0.276 / 0.322 |

At k=25: 1 gene in 10/10 runs, 6 in at least 8/10, 92 in a single run.

## Interpretation limits
- ROC-AUC near 1 with 3-4 normals per test fold: accuracy-type metrics are uninformative here; balanced accuracy is limited by the fixed 0.5 threshold, not by ranking quality.
- A high-AUC model with only moderately stable top-25 genes is an observation for H5, not yet a finding: it needs the permutation null (Phase 7), other models/explainers and a t-test baseline.
- Analytic Nogueira CIs are not used (they assume independent sets; CV training sets overlap). Per-repeat values are dependent and only show spread.
- Post-hoc exploratory diagnostic (rare-detection probes; class-specific detection; see `docs/development_log.md`) changed no design decision.

## Runtime (seconds)
data load 3.9 | CV loop (10 folds) 13.3 (about 1.3 per fold) | pooled metrics + bootstrap + stability 40.3 | total 57.6. The bootstrap step (2,000 resamples x 3 metrics x 2 repeats, sklearn scoring) dominates and must be vectorised before scaling.
