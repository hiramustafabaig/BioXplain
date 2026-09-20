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

