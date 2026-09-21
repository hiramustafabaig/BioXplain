# Phase 3 report: Linear SVM, Random Forest and XGBoost added; controlled benchmark

Code commit `d2c2dcf` (clean tree). Benchmark run `results/benchmarks/20260921T200249Z_models_1x5_037018c4/` (1 repeat x 5 folds, seed 20260921, GSE42568 only, no explainers, no stability). Configurations are the pre-specified ones in `docs/methodology_decisions.md` D11b; none was tuned, and no result influenced them. **This is not a model leaderboard** (near-ceiling ROC-AUC is an observation about this easy cohort, not a success criterion).

## Verification steps per model
1. **Single controlled fold** (`scripts/benchmark_models.py pilot`): training succeeded; scores finite; probabilities within [0, 1] for LR / RF / XGBoost and unbounded margins for the SVM; predicted labels in {0, 1}; number of predictions equals the fold size; model support <= number of genes.
2. **Tests** (103 in total, all passing): pre-registered defaults guard; parameter carry-through; XGBoost class weight computed from training labels only; fold contract, planted-signal recovery and chance-level behaviour on noise labels for each model; scaling policy per model; fixed-thread determinism; RF thread invariance; exact-zero effect of unused features; the leakage-invariance check for all four models (mutation-tested).
3. **Small benchmark** below.

## Results (pooled out-of-fold, n = 121; 95% stratified bootstrap CI over samples; 1 repeat, so no between-repeat spread)
| Model | ROC-AUC (CI) | AP normal (CI) | balanced acc. | sensitivity | specificity | F1 normal | confusion tn / fp / fn / tp |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.999 (0.997-1.000) | 0.997 (0.982-1.000) | 0.941 | 1.000 | 0.882 | 0.938 | 15 / 2 / 0 / 104 |
| Linear SVM | 0.998 (0.992-1.000) | 0.988 (0.957-1.000) | 0.981 | 0.962 | 1.000 | 0.895 | 17 / 0 / 4 / 100 |
| Random forest | 0.995 (0.985-1.000) | 0.975 (0.926-1.000) | 0.853 | 1.000 | 0.706 | 0.828 | 12 / 5 / 0 / 104 |
| XGBoost | 0.988 (0.969-0.999) | 0.936 (0.854-0.993) | 0.878 | 0.990 | 0.765 | 0.839 | 13 / 4 / 1 / 103 |

Primary metrics: ROC-AUC and normal-class average precision (threshold-free / informative for the minority). The other columns depend on each model's fixed decision rule (probability > 0.5 or margin > 0), which is not calibrated for a 14% minority; they differ mainly through where that fixed threshold falls, not through ranking quality. Confidence intervals overlap for the linear models and RF; XGBoost is lowest on both primary metrics but the intervals overlap, so **no model is declared better**.

## Model support and runtime (per training fold; about 96-97 training samples, about 20,100 genes)
| Model | genes used by the fitted model (mean; range over folds) | prep s | fit s |
|---|---|---|---|
| Logistic regression | 20,128 (100%; dense) | 0.43 | 0.30 |
| Linear SVM | 20,128 (100%; dense) | 0.42 | 0.21 |
| Random forest | 1,172 (5.8%; 1,093-1,238) | 0.33 | 1.59 |
| XGBoost | 24.4 (0.12%; 21-30) | 0.35 | 8.8 |

Projected fit cost of the planned 20 repeats x 5 folds: LR and SVM about 1 min each, RF about 3 min, XGBoost about 15 min (explainers excluded; SHAP and permutation importance costs will be benchmarked in Phase 4 before any full run).

## Observations with methodological consequences
- **Sparse tree explanations.** XGBoost splits on only about 24 genes and RF on about 6% of genes, whereas LR / SVM coefficients are dense. With the zero-attribution rule (D13b) XGBoost top-k sets at k = 25 and 50 contain at most about 24-30 genes, i.e. the whole used set: k-dependence collapses for XGBoost, and its stability is the stability of the entire support. Set sizes differ between models, so only Nogueira's Phi (variable sizes) is comparable across models; Kuncheva is defined only for constant size.
- **Thread count matters for XGBoost.** Scores differ by up to about 1e-2 between `n_jobs` 1 and 4 (RF is invariant); `n_jobs=4` is pinned and recorded. Bit-identical reproduction of XGBoost outputs on another machine additionally assumes the same XGBoost version and thread count (recorded in the manifest).
- **Exact permutation-importance premise confirmed.** Permuting genes outside the fitted model's support never changed any prediction (0 of 40 sampled unused genes in both RF and XGBoost toy tests), whereas used genes did.
- **Explanation interface.** LR and SVM: coefficient explainer works now. RF and XGBoost: no explainer yet (documented plan D12b/D12c; `run_fold` accepts `explainer_name=None`); their models are ready for SHAP (TreeExplainer) and permutation importance in Phase 4.

## Process note (commits)
The three models share one factory / `run_fold` refactor (registry, resolved specs, score types, support, timings), so they were committed together instead of as three artificial intermediate commits; each was benchmarked and tested individually as described above.

## Limitations
One repeat; single cohort; near-ceiling discrimination means model differences here are small and uncertain; fixed decision thresholds; no explainers yet.
