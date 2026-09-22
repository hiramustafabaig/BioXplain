# Discovery freeze

Frozen at 2026-09-22T19:29:58.975548+00:00 on commit `bed57e4db41a66c136980050f6299712ab746b8a` (clean tree). Machine-readable: `docs/freeze/discovery_freeze.json`.

After this file was committed no discovery decision may use GSE65194: no tuning, and no gene, k, threshold, preprocessing or endpoint change.

## Discovery run
- run: `results/matrix/20260921T210020Z_full_matrix_5367d04d` (config sha256 `5367d04dda00ec4d`, code commit at run start/manifest `ddb6cf92`, 100 folds, seed 20260921)
- samples: {'excluded_genes': [], 'subset': 'full', 'n_samples': 121, 'n_cancer': 104, 'n_normal': 17, 'labels': 'real'}
- feature universe: 42892 probes / 20848 genes (sha256 `aee3269ffcdd7b1b`); eligible on the full cohort: 20118 genes

## Preprocessing
- expression_filter: floor detection; detected in >= ceil(0.5 * n_minority_train) training samples (D7)
- probe_collapse: max training mean, ties by probe id (D6); frozen probe per gene chosen on the full discovery cohort
- scaling: z-score with training statistics for LR/SVM; log2 values for trees (D8)

## Models (resolved, pinned)
- `logreg`: {'name': 'logreg', 'C': 1.0, 'class_weight': 'balanced', 'max_iter': 5000}
- `svm`: {'name': 'svm', 'C': 1.0, 'loss': 'squared_hinge', 'class_weight': 'balanced', 'max_iter': 20000}
- `rf`: {'name': 'rf', 'n_estimators': 500, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'class_weight': 'balanced_subsample', 'n_jobs': 4}
- `xgb`: {'name': 'xgb', 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.5, 'tree_method': 'hist', 'n_jobs': 4}
- `logreg_C0.1`: {'name': 'logreg', 'C': 0.1, 'class_weight': 'balanced', 'max_iter': 5000}
- `logreg_C100`: {'name': 'logreg', 'C': 100, 'class_weight': 'balanced', 'max_iter': 5000}

## Explainers
- `coef`: |coef| on z-scored genes (LR, SVM)
- `shap`: mean |SHAP| over training-fold samples (Linear/Tree)
- `perm`: exact permutation importance, training fold, class-balanced Brier, R=10 (D12b)
- `ttest`: |Welch t|, training fold only

## Stability
- {'primary': 'Nogueira Phi (JMLR 2018 Def. 4)', 'secondary': ['Kuncheva (constant size only)', 'Jaccard'], 'd': 20848, 'unit': 'one 5-fold CV repeat', 'dimensions': ['resampling', 'model', 'explainer', 'combined', 'baseline', 'sensitivity']}

## Consensus
- rule: within-pipeline selection frequency >= 0.5 at k=25 in >= ceil(n_primary/2) primary pipelines
- frozen list sizes: {'S_cons': 1, 'S_top25': 25, 'S_top50': 50}

## Frozen lists
### S_cons
DKFZp779M0652

### S_top25
DKFZp779M0652, ABHD1, LYVE1, MMP28, DMGDH, FLVCR1, NIPSNAP3B, RAE1, HSPB6, LOC102723493, LOC284825, PQLC2L, ZNRF2P1, ABCD2, CELSR1, SLC7A10, LOC101927420, STX11, DCAF12L2, ALPI, PNMA2, PAFAH1B3, TET3, CCDC85C, ANKRD10-IT1

### S_top50
MMP28, DKFZp779M0652, ABHD1, DMGDH, LYVE1, NIPSNAP3B, HSPB6, LOC284825, LOC102723493, FLVCR1, PQLC2L, ABCD2, RAE1, STX11, SLC7A10, LOC101927420, C11orf70, CCDC178, CELSR1, ANKRD10-IT1, ZNRF2P1, ACADL, INHBA, PFKFB1, DCAF12L2, APOB, TET3, TMOD1, PRKCZ, CCDC85C, DENND2A, PALM, SDC1, PNMA2, PAFAH1B3, ALPI, MYO16, HSPB7, NAALAD2, C22orf39, DNMT1, PAQR4, C8orf34, LRRC15, EPB42, STIL, TREM2, C15orf48, CIDEA, ADH1C

## External endpoints
- docs/methodology_decisions.md Addendum A4 (aligned effect r_g, C1, C2, signature score, frozen LR; bootstrap over external samples)

## Software
- {'python': '3.11.9', 'platform': 'Windows-10-10.0.26200-SP0', 'numpy': '2.4.6', 'pandas': '3.0.5', 'scipy': '1.17.1', 'scikit-learn': '1.9.0', 'pyyaml': '6.0.3', 'shap': '0.51.0', 'xgboost': '3.2.0', 'gseapy': '1.3.1'}

Frozen gene table: `results/freeze/frozen_genes.csv` sha256 `3509ba49148185d9e7a7843d7a8f86f6b8543ed1b45c687d4a2f721685aec00e`.