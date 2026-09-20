# Development log

Format: dated entries; problems are recorded as Problem / Expected / Actual / Diagnosis / Fix / Validation / Impact / Lesson; decisions as Decision / Why / Evidence / Impact / Next.

## 2026-09-20 - Reconnaissance

**Decision/result:** Ran project reconnaissance before any implementation.
- **State found:** `D:\Bioxplain` had folder skeleton only (`data/raw`, `data/processed`, `docs`, `notebooks/01_data_exploration.ipynb` [0 bytes], `results/*`, `src/`), a `.venv` (Python 3.11.9) and the draft `BioXplain_Research_Project_Draft (1).docx`. No Git repo, no remote. `gh` CLI not installed.
- **Environment:** venv contains numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.2.0, shap 0.51.0, gseapy 1.3.1, matplotlib 3.11.1, seaborn, scipy 1.17.1, jupyterlab. **Missing:** pytest (needed). System Python 3.14 is the default `python`; project uses the 3.11 venv.
- **Why it matters:** Environment is sufficient for the whole planned pipeline once pytest is added.
- **Evidence:** `docs/research/literature_review.csv`, `open_source_review.md`, `gap_analysis.md`.

**Decision:** The draft's headline differentiator ("stability + consensus + leakage-safety") is not novel; reframed as an empirical study (explanation stability across model x explainer, and whether it predicts replication in an independent cohort) plus a tested tool.

**Problem (process):** GEO series web pages (`ncbi.nlm.nih.gov/geo/query/acc.cgi`) returned a CAPTCHA to the fetch tool. Not bypassed. Used GEO FTP directory listings instead (file names and sizes verified: `GSE42568_series_matrix.txt.gz` 22.6 MB; `GSE65194_series_matrix.txt.gz` 76.5 MB; `GPL570.annot.gz` 8.1 MB). Sample counts for both series are currently from secondary literature and **must be re-verified from the downloaded matrices.**
