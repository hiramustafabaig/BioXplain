# BioXplain

An empirical study — and a tested, reproducible Python workflow — of how consistently machine-learning explanations identify gene-expression features across models, explainers and resampling, and whether that internal stability predicts replication in an independent breast-tissue cohort.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white) ![Tests](https://img.shields.io/badge/tests-160%20passing-B9D175) ![Status](https://img.shields.io/badge/status-research%20portfolio-450C3F) ![Not peer reviewed](https://img.shields.io/badge/peer%20review-not%20peer%20reviewed-lightgrey)

**Live site: [hiramustafabaig.github.io/BioXplain](https://hiramustafabaig.github.io/BioXplain/)** (findings, pipeline and an in-page literature reader).

**Explore it:** the [showcase page source](docs/index.html) · the written [literature review (PDF)](docs/research/BioXplain_Literature_Review.pdf), also as [Word](docs/research/BioXplain_Literature_Review.docx) · the [full report](docs/reports/final_report.md).

**Full report:** [docs/reports/final_report.md](docs/reports/final_report.md). **Results tables/figures:** [docs/results/results_summary.md](docs/results/results_summary.md), `results/figures/`. **Methodology (22 pre-specified decisions, written before results):** [docs/methodology_decisions.md](docs/methodology_decisions.md). **Discovery freeze:** [docs/freeze/discovery_freeze.md](docs/freeze/discovery_freeze.md). **Limitations:** [docs/limitations.md](docs/limitations.md). **Reproduce it:** [docs/reproducibility.md](docs/reproducibility.md).

## Why this exists

Gene-expression classifiers routinely nominate "important genes" from near-perfectly separable tumour-vs-normal data. Whether those genes are a reproducible property of the disease, of tissue composition, or of the analyst's modelling choices is usually not tested. We found (`docs/research/`) that stable, leakage-safe gene selection already has tools (Stabl, OmicSelector, RobustModelMaker), so BioXplain does not claim a new algorithm. Its contribution is empirical: a chance-corrected decomposition of explanation stability across resampling, model and explainer, referenced against a matched permutation-label null, and a pre-specified test of whether that stability predicts replication in an independent cohort beyond raw effect size — plus an explicit check of tissue-composition confounding.

## What the workflow does

GSE42568 (discovery, 104 cancer / 17 normal) → leakage-safe repeated stratified 5-fold CV → 4 models (logistic regression, linear SVM, random forest, XGBoost) × 3 explainers (coefficients, SHAP, exact permutation importance) + a Welch-t baseline → top-k gene sets (k = 10/25/50, positive attribution only) → Nogueira/Kuncheva/Jaccard stability across four dimensions (resampling, model, explainer, combined) → a 30-replicate permutation-label null → a transparent, un-weighted consensus table → a discovery freeze → external replication on GSE65194 (130 tumours + 11 healthy) with scale-free endpoints → enrichment and a tissue-composition check.

## Datasets

Both from NCBI GEO, GPL570 (Affymetrix HG-U133 Plus 2.0): **GSE42568** (discovery) and **GSE65194** (external). Raw files are not stored in this repository; `configs/data_manifest.json` has the official download URLs and SHA-256 hashes, and every loader verifies them. See `docs/data_reconnaissance.md` for the full structural audit.

## Headline findings (see the full report for context and caveats)

- Predictive performance is near ceiling for every model (ROC-AUC 0.975–0.997) and does not distinguish them.
- Explanation stability is low for most model×explainer pipelines and depends strongly on choices — model, explainer, and for logistic regression, regularisation strength — that do not affect accuracy.
- Against a matched permutation-label null, only some pipelines (SVM, XGBoost, LR away from its primary setting) are clearly more stable than chance; the primary, pre-specified logistic-regression pipeline is not.
- Internal stability shows a small but bootstrap-distinguishable association with independent-cohort gene-level replication, beyond discovery effect size alone (partial Spearman ≈ 0.08).
- A composite signature from the most stable genes replicates strongly at the signature level externally — but a large share of that appears attributable to a tissue-composition-associated signal, not confirmed tumour-specific biology.
- No enrichment term survives correction.

## Repository layout

| Path | What it holds |
|---|---|
| `src/bioxplain/` | The package: data loading, leakage-safe preprocessing, models, explainers, stability metrics, permutation null, consensus, freeze, external replication, enrichment and composition checks |
| `scripts/` | Runnable entry points for each stage (matrix runs, null, freeze, external validation, enrichment, figures) |
| `configs/` | YAML run configurations and `data_manifest.json` (official download URLs and SHA-256 hashes) |
| `tests/` | The pytest suite, including deliberate leakage tests |
| `results/` | Saved result tables, rankings and figures that the report's numbers are read from |
| `docs/` | Report, methodology decisions, discovery freeze, limitations, reproducibility notes and the literature review (`docs/research/`) |
| `docs/index.html` | The standalone showcase web page (served by GitHub Pages) summarising what was done and found |
| `notebooks/` | Data exploration |

## Reproduce it

See [docs/reproducibility.md](docs/reproducibility.md). In short: `pytest`, then `python scripts/run_matrix.py configs/full_matrix.yaml`, the sensitivity/null queues, `scripts/create_freeze.py`, `scripts/run_external.py`, `scripts/run_enrichment.py`, `scripts/make_figures.py`.

## Limitations

Summarised in [docs/limitations.md](docs/limitations.md) and Section 4–5 of the final report. In particular: near-ceiling task difficulty limits what accuracy can show; both cohorts share one microarray platform, so external validation is independent-cohort, not independent-technology; normal-group sizes are small (17 and 11); and no gene identified here should be read as causal, clinically validated, or a discovered biomarker.

## Licence

To be finalised by the project owner before any public release.

## Status

Discovery, sensitivity, null, freeze, external validation, enrichment and figures are complete (see the final report). This is a student research-portfolio project, not a peer-reviewed publication.
