# Reproducibility guide

Everything in this repository is reproducible from scripts and configuration. Raw GEO data are not stored in Git.

## 1. Environment
- Python 3.11.9; exact package versions in `requirements-lock.txt` (numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.17.1, shap 0.51.0, xgboost 3.2.0, gseapy 1.3.1).
- Windows 11, 4 cores / 8 logical CPUs, 16 GB RAM (timings in the reports are from this machine).
- Install: `python -m venv .venv`, install `requirements-lock.txt`, then `pip install --no-deps -e .`.
- **Thread counts are part of the configuration**: RF and XGBoost use `n_jobs=4`. XGBoost output depends on the thread count (measured: scores differ by about 1e-2 between 1 and 4 threads), so results are bit-identical only for the same XGBoost version and `n_jobs`. RF is thread-invariant. Every manifest stores the resolved model specification.

## 2. Data (download and verification)
| File | URL | Place at | SHA-256 |
|---|---|---|---|
| GSE42568 series matrix (22.6 MB) | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42568/matrix/GSE42568_series_matrix.txt.gz | `data/raw/GSE42568/` | in `configs/data_manifest.json` |
| GPL570 annotation (8.5 MB) | https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz | `data/raw/GPL570/` | same |
| GSE65194 series matrix (76.5 MB) | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65194/matrix/GSE65194_series_matrix.txt.gz | `data/external/GSE65194/` | same |

Every loader verifies the SHA-256 against `configs/data_manifest.json` and raises on mismatch. `python scripts/data_recon.py` regenerates the reconnaissance numbers (`results/metrics/data_recon.json`); `python scripts/build_feature_universe.py` regenerates `configs/feature_universe.tsv`.

## 3. Order of execution
1. `pytest` (all tests; about 3-4 minutes).
2. `python scripts/run_matrix.py configs/full_matrix.yaml` (20 repeats x 5 folds; about 35 min; resumable, per-fold caches in `results/matrix/<run>/folds/`).
3. `python scripts/run_queue.py` (permutation null 30 replicates, Dec-2004 subset, size-matched control, composition ablation).
4. Commit, then `python scripts/create_freeze.py results/matrix/<full_run>` (requires a clean tracked tree) and commit the freeze.
5. `python scripts/run_external.py` (refuses to run unless the freeze is committed and unmodified), `python scripts/run_enrichment.py`.
6. `python scripts/make_figures.py`.

## 4. Randomness
Master seed 20260921. Split seed of repeat r = seed + r; model `random_state` = seed; permutation-importance permutations from `SeedSequence([seed, repeat, fold, config, explainer])`; null replicate b uses labels permuted with `SeedSequence([seed, 777, b])` and split seed `seed + 100000*(b+1)`; bootstraps use the master seed. Two runs of the same configuration are bit-identical, including across `PYTHONHASHSEED` values (tested on real data for a full 13-pipeline fold).

## 5. Provenance
Each result folder contains `manifest.json` (experiment id, git commit and dirty flag, config SHA-256, data hashes, software versions, seeds, resolved model specifications, sample definition, feature universe, runtimes, outputs) and `config.yaml`. The discovery freeze (`docs/freeze/`) additionally records file hashes of the discovery outputs and of the frozen gene table.

## 6. What is not reproducible bit-for-bit
- Logistic-regression coefficient/SHAP importance values differ at about 1e-13 when the BLAS thread count differs (e.g. joblib worker vs main process); gene rankings were identical in every checked fold. Bit-identity holds within the same environment.
- Enrichment libraries are downloaded from Enrichr once and cached with a SHA-256 and download date (`data/external/gene_sets/`, not committed; hashes in `results/enrichment/manifest.json`). Later Enrichr releases may differ.
- Wall-clock timings depend on the machine and load.
