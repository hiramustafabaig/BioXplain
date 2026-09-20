# Open-source / competitor review

Metadata (stars, licence, last push, archived) pulled from the GitHub REST API on 2026-09-20.
Purpose/methods come from README/paper summaries, **not** from running or reading source code. Nothing here has been executed yet.

| Project | Lang | Licence | Stars | Last push | Status | What it does | Overlap with BioXplain | Reuse? |
|---|---|---|---|---|---|---|---|---|
| [gregbellan/Stabl](https://github.com/gregbellan/Stabl) | Python (notebooks) | BSD-3-Clause-Clear | 77 | 2024-11-18 | maintained lightly | Stability selection + noise injection, sklearn-compatible; Nat Biotech 2024 | Stable sparse signatures; no post-hoc explainers, no pathway module | Licence permits reuse with attribution; better used as a *comparator*, not code source |
| [kstawiski/OmicSelector](https://github.com/kstawiski/OmicSelector) | R (mlr3) | MIT per docs (GitHub API: NOASSERTION - verify) | 36 | 2026-07-21 | active | Nested CV, zero-leakage selection, Nogueira index, DALEX interpretability | Closest *feature-set* competitor; R not Python; no external-cohort or pathway module documented | Do not copy; cite |
| [mickaelleclercq/BioDiscML](https://github.com/mickaelleclercq/BioDiscML) | Java | GPL-3.0 | 34 | 2025-07-18 | maintained | Automated biomarker signature search over many selectors/models | Signature discovery; not explanation-centred | GPL-3: cannot incorporate into a permissive package |
| [amaxiom/RobustModelMaker](https://github.com/amaxiom/RobustModelMaker) | Python | MIT | 1 | 2026-09-16 | brand new | Bootstrap stability selection + leakage-safe nested CV (arXiv 2606.01566) | Same leakage+stability idea in Python | Cite; compare; do not claim first |
| [IraSirohi/isirohi-research](https://github.com/IraSirohi/isirohi-research) | Python | MIT | 0 | 2026-09-09 | paper code | Panel-stability analysis (RF/SVM, Kuncheva, GO) on TCGA/CPTAC | Same core premise; no SHAP | Cite |
| [shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics](https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics) | Notebooks + scripts | MIT | 0 | 2026-09-20 | very active (225 commits) | PAM50 subtype classification, tri-method consensus selector (ANOVA/LASSO/RF, >=2/3 vote), SHAP, 4-cohort zero-retraining external validation | Closest *breast + SHAP + consensus + external validation* project; not a package; subtype task; no per-model x explainer stability analysis | Cite; different task |
| [Ekeany/Boruta-Shap](https://github.com/Ekeany/Boruta-Shap) | Python | MIT | 656 | 2024-02-19 | stale, 76 open issues | Boruta selection with SHAP importance | Single-run selector; no stability reporting | Possible comparator |
| [scikit-learn-contrib/stability-selection](https://github.com/scikit-learn-contrib/stability-selection) | Python | BSD-3-Clause | 215 | 2023-06-05 | **archived** | Meinshausen-Buhlmann stability selection | Standard stability selection | Archived - do not depend on it |
| [bommert/stabm](https://github.com/bommert/stabm) | R | LGPL-3.0 | 7 | 2023-04-05 | stable | 20 stability measures | Metric reference | Validate our numbers against it (do not copy) |
| [zqfang/GSEApy](https://github.com/zqfang/GSEApy) | Python | BSD-3-Clause | 710 | 2026-09-02 | active | Enrichr API + GSEA/prerank | Enrichment layer | **Use as dependency** (already in venv, v1.3.1) |
| [shap/shap](https://github.com/shap/shap) | Python | MIT | 25.8k | 2026-09-18 | active | SHAP | Explainer | **Use as dependency** (venv v0.51.0) |
| [bensmailchama-boop/EvoXplain](https://github.com/bensmailchama-boop/EvoXplain) | Python | AGPL-3.0 | 2 | 2026-09-10 | new | Measures explanation multiplicity across training runs | Conceptual support | AGPL: do not reuse |
| [nogueirs/JMLR2018](https://github.com/nogueirs/JMLR2018) | notebooks | none stated | 19 | 2018 | static | Reference code for Nogueira stability | Metric reference | No licence = all rights reserved: re-implement from the paper's formula |
| colombelli/efs-assembler, kuncheva-index | Python | not checked | - | - | - | Ensemble feature selection / Kuncheva index | Small overlap | Not reviewed in depth |

## Take-aways
1. No tool found combines **{several model families} x {several explainers} x {resamples}** with chance-corrected stability *of the explanations themselves*, cross-cohort replication tests, and enrichment in one installable Python package. (Absence of evidence within a limited search - see gap_analysis.md caveats.)
2. Leakage-safe stable selection **already exists** (OmicSelector, RobustModelMaker, OncoResolve, Stabl). We must not sell that as the contribution.
3. GitHub search returned almost nothing for the direct query "SHAP + gene signature stability"; results were dominated by 0-star, days-old repos - the space is active but immature and mostly one-off study code.

## Licence rules adopted
- Depend on MIT/BSD packages via pip; never vendor code.
- Re-implement Nogueira/Kuncheva from the papers; validate against stabm output.
- No GPL/AGPL code copied.
