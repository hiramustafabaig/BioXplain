# Regularisation-strength sensitivity of logistic-regression coefficient rankings (D11c)

Run `results/sensitivity/20260921T194825Z_sensitivity_logreg_C_d8b34927/` (code commit `6863ab1`, clean tree, config `configs/sensitivity_logreg_C.yaml` written and committed before the run). GSE42568 only; 5 repeats x 5 folds = 25 feature sets per C, identical splits for every C; universe d = 20,848 genes; runtime 100 s. GSE65194 was not used.

## Pre-specified criterion and verdict
Robust iff every C has the same Nogueira-Phi band at k = 25 as C = 1 (paper's scale: < 0.40 poor, 0.40-0.75 intermediate to good, > 0.75 excellent) and pooled ROC-AUC >= 0.98. **Verdict: materially changed.** Every C other than 1 falls in the "intermediate to good" band, C = 1 in "poor". The performance criterion is satisfied for all C.

| C | Phi k=10 | Phi k=25 | Phi k=50 | Jaccard k=25 | distinct genes k=25 | band (k=25) | pooled ROC-AUC (mean of 5 repeats; min) | balanced acc. | sens. / spec. |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.466 | 0.481 | 0.510 | 0.330 | 123 | intermediate | 0.9966 (0.992) | 0.934 | 0.998 / 0.871 |
| 0.1 | 0.455 | 0.431 | 0.479 | 0.287 | 128 | intermediate | 0.9964 (0.990) | 0.929 | 0.998 / 0.859 |
| **1 (primary)** | **0.207** | **0.248** | **0.271** | **0.151** | **257** | **poor** | 0.9973 (0.994) | 0.934 | 0.998 / 0.871 |
| 10 | 0.456 | 0.477 | 0.505 | 0.333 | 127 | intermediate | 0.9963 (0.995) | 0.961 | 0.921 / 1.000 |
| 100 | 0.445 | 0.476 | 0.497 | 0.332 | 131 | intermediate | 0.9963 (0.995) | 0.961 | 0.921 / 1.000 |

Per-repeat Phi(k=25) at C = 1 ranges 0.10-0.32 (other C: 0.37-0.50); the 2-repeat pilot value 0.32 was a favourable pair. Per-repeat values use 5 dependent sets and only convey spread.

Rank agreement with C = 1 (mean over 25 folds): Spearman of |coef| 0.69 / 0.71 / 1 / 0.63 / 0.62 and top-25 Jaccard 0.31 / 0.32 / 1 / 0.14 / 0.14 for C = 0.01 / 0.1 / 1 / 10 / 100.

## Investigation of the mechanism (exploratory, post hoc; `scripts/diagnose_regularization_regimes.py`, output saved in the run folder)
- **Not a solver artefact:** lbfgs converged in 15-41 iterations for every C (limit 5,000; no convergence warnings) and every model separates the training data (minimum margin 3.7-7.4).
- **C = 10 and C = 100 are one solution.** On separable data the logistic loss has no finite minimiser; the solver stops on its gradient tolerance and the coefficient norm plateaus (0.849 vs 0.868), so the stopping rule and not C determines the fit there. Rank correlation 1.00, top-25 Jaccard 0.92.
- **Two regimes and a transition.** C in {0.01, 0.1}: rank correlation 0.98, top-25 Jaccard 0.68, ranking most similar to a univariate mean-difference ranking (Spearman 0.75 / 0.68). C in {10, 100}: max-margin-like regime (Spearman with mean difference 0.57 / 0.56). The two regimes share almost no top-25 genes (Jaccard about 0.10) although their rank correlation is 0.65. C = 1 lies between them (Jaccard 0.31 to the small-C side, 0.16 to the large-C side, Spearman 0.50 to the mean-difference ranking) and has the flattest coefficient profile (top-25 / median |coef| 4.8 vs 5.0-5.6), which is consistent with more rank noise and lower stability.
- This is an interpretation of associations in six folds, not a proof of mechanism.

## What this means (and does not)
- The pilot's "low to moderate stability of coefficient rankings" is **not a robust property of logistic regression on this cohort**: it depends on the regularisation regime. Phi(k=25) is 0.43-0.48 away from C = 1 and 0.25 at C = 1.
- More importantly for the research question, **the identity of the top genes depends strongly on an analyst's hyper-parameter choice within one model class** (top-25 overlap about 0.1 between regimes), even though predictive performance is the same (ROC-AUC 0.996-0.997). This is an "analytic-choice stability" result in its own right.
- Per the pre-specification, **C = 1 stays the primary configuration**; no C was selected for looking best. Primary Phi values must always be reported together with this sensitivity.
- **Decision for the user (not taken silently):** whether to add a small-C and a large-C logistic-regression configuration (for example C = 0.1 and C = 100) as extra rows of the model x explainer matrix, so that "stable under one choice" versus "stable across analytic choices" is measured directly. It adds configurations but does not change the research question.

## Limitations
Single cohort; 25 dependent feature sets per C; L2 logistic regression only; class-balanced weighting and preprocessing held fixed; diagnostics on 6 folds; no permutation null yet (Phi bands describe agreement beyond chance in the paper's sense, but the chance-level distribution under this exact pipeline is established only by the Phase 7 null).
