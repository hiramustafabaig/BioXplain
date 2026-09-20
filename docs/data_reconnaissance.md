# BioXplain Data Reconnaissance Report

Date: 2026-09-20. Reproduce with `.venv\Scripts\python scripts\data_recon.py` (about 25 s). Every number below is in `results/metrics/data_recon.json`; per-sample labels are in `results/metrics/sample_manifest_GSE42568.csv` and `..._GSE65194.csv`.
Scope: structure and metadata only. No model, gene selection, class contrast or performance number was computed on either cohort. On GSE65194 only counts, labels, global value scale, ID overlap and technical-replicate similarity were examined.

## 1. Download status
All three downloads succeeded on the first attempt with `curl.exe -L --fail --retry 3` (about 34 s total). Neither `*_RAW.tar` nor `GPL570_family.soft.gz` was downloaded.

## 2-4. Files, sizes, integrity
| File | Exact path | Bytes | Decompressed bytes | SHA-256 |
|---|---|---|---|---|
| GSE42568 series matrix | `D:\Bioxplain\data\raw\GSE42568\GSE42568_series_matrix.txt.gz` | 22,565,943 | 79,859,027 | `43b93f72f1d81838dcd2a5b6ccc3bc1875e46230dad0d7346236773a99e32b76` |
| GPL570 annotation | `D:\Bioxplain\data\raw\GPL570\GPL570.annot.gz` | 8,471,521 | 39,459,873 | `d7cd44352127b1e34f3a720ebea86093ef255a38f1612a85a2962b71bde8f394` |
| GSE65194 series matrix | `D:\Bioxplain\data\external\GSE65194\GSE65194_series_matrix.txt.gz` | 76,464,351 | 165,378,449 | `f92d2109fe0d64ed056d9673e8733b2af25173230f3632e370201b6a4054caba` |

Checks passed for all three: file exists at the exact path; the byte size equals the FTP directory size recorded before download (both series matrices; the annotation was 8.1 MB); a full gzip decompression completes without CRC or length error; the contents were parsed; the accession and platform in the header match (`!Series_geo_accession`, `!Series_platform_id = GPL570`; the annotation header states `GPL570`, HG-U133_Plus_2).

## 5. GSE42568 (discovery cohort)
- Matrix: **54,675 probes x 121 samples**; 0 missing values; 0 duplicate probe IDs; 62 AFFX control probes; the matrix columns match the header GSM order.
- Platform GPL570; Dublin City University (Clarke et al.); processing: GC-RMA; labelling: Affymetrix two-cycle protocol.
- Sample IDs: GSM1045191 to GSM1045311, contiguous (full list in the manifest).
- **Labels: 104 cancer, 17 normal.** Four independent fields agree with no exceptions: `tissue`, `source_name_ch1`, title prefix, description. Metadata is sufficient to reproduce the labels.
- Tumour clinical fields: ER+ 67, ER- 34, ER NA 3; grade 1/2/3 = 11/40/53. Normals have no clinical data.
- Scale: log2-like, min 2.3128, median 2.84, 99th pct 11.43, max 16.05; no negative values.
- Duplicates: 0 identical columns; the maximum sample-sample Pearson r is 0.9875 (no near-duplicate samples).
- Probe-to-gene: see section 7.

## 6. GSE65194 (future external cohort; structure only)
- Matrix: **54,673 probes x 178 arrays**; 0 missing values; 0 duplicate probe IDs; the same GPL570 platform.
- Series title: "Institut Curie (Maire cohort) -- Affy CDF", a SubSeries of GSE65216.
- Sample IDs: GSM1588970 to GSM1589153.
- **Array-level groups (178):** TNBC 55, Her2 39, Luminal B 30, Luminal A 29, CellLine 14, Healthy 11.
- **Unique tumours: 130** (TNBC 41, Her2 30, Luminal B 30, Luminal A 29). 153 tumour arrays = 107 single arrays + **23 tumours measured twice**, identified by `_repA/_repB` in the title. Duplicates never disagree on subtype.
- The 14 cell lines are 184B5, MDA-MB-436, HCC1143, HCC1187, BT20, HCC1937, MCF-12A, HCC38, Hs 578T, MDA-MB-468, BT-549, HCC70, MDA-MB-157, MDA-MB-231. GEO labels all of them "breast cancer derived", but MCF-12A and 184B5 are generally described as non-tumorigenic lines (background knowledge, not from this file).
- Labels are reproducible from metadata (`sample_group`). Frozen a-priori inclusion rule, written to the manifest from metadata alone: exclude cell lines; treat the two arrays of a duplicated tumour as one unit. This yields **130 tumours + 11 normals = 141 analysis units**.
- Scale: min -1.22, median 3.24, 99th pct 12.56, max 20.00; 163 negative values; 6.4% of values < 2. There is **no floor value**.
- Processing (from the submitter): GC-RMA, then **batch and hybridization effects corrected by a linear model with fixed effects**, then "samples with technical replicates were subsequently averaged". One-cycle labelling protocol.
- Replicate arrays are **not** averaged in the matrix: 0 of 23 pairs are identical; pair Pearson r is 0.994 to 0.997 (median 0.996). The two arrays of a tumour sit in different array batches.
- Duplicates: 0 identical columns.
- Array batches B6-B8 (71 arrays) contain only TNBC and Her2 tumours. Healthy samples fall in B1-B5 together with all Luminal tumours and the cell lines.
- Loader: the same `read_series_matrix` / `characteristics` code parses both series without modification.

### Cross-series checks (GSE42568 vs GSE65194)
- **Shared GSM: 0.** Shared sample titles: 0. Identical expression columns on shared probes (4 dp): 0. The institutions differ (Dublin vs Paris). Shared patients cannot be excluded by metadata alone, but nothing suggests any.
- Probe IDs: 54,673 shared, in the **same row order**. Only in GSE42568: `217199_s_at` (STAT2) and `233859_at` (CEP128). Only in GSE65194: none.

## 7. GPL570 annotation
- 54,675 records, one per probe; the IDs are identical to the GSE42568 matrix IDs. Annotation file date: Aug 09 2016.
- Columns: probe **`ID`**; **`Gene symbol`**; **`Gene ID`** (Entrez); plus title, UniGene, GenBank, chromosome, GO terms.
- Missing/ambiguous mappings:
  - 9,557 probes (17.5%) have no symbol; the Gene ID is empty in exactly the same rows.
  - 2,214 probes (4.0%) map to several genes (`///` separator).
  - 62 are AFFX controls (10 of them carry a symbol).
- **Usable probes** (single gene, non-control, non-empty): **42,894 = 78.5%**, covering **20,848 unique gene symbols**. Each symbol maps to exactly one Entrez ID and vice versa.
- Multiple probes per gene: 9,924 genes have 1 probe, 10,924 have two or more, 1,388 have five or more, and the maximum is 20. **76.9% of usable probes belong to multi-probe genes.**
- Usable probes present in both series matrices: 42,892.

## 8. Important methodological findings

None is one of the pre-agreed hard stop conditions: labels are reliable, there is no sample overlap, and the probe spaces are compatible. The findings below are ordered by impact on the design and need your decision.

**F1. The two cohorts are on different numeric scales (design-affecting).** GSE42568 is plain GC-RMA with a hard floor: 28.5% of all values equal 2.3128. GSE65194 is GC-RMA followed by batch correction, so it has no floor, contains negative values, has a wider spread (75th pct 6.92 vs 5.53) and higher per-sample medians. A model or threshold fitted on GSE42568 raw values is not directly transferable. Consequence: external replication must be defined on **scale-free quantities** (direction, standardized effect size, rank) and not on raw-value classification. Transferring a frozen classifier would need per-cohort standardization, which is a transductive but label-free step and must be stated explicitly.

**F2. Technical replicates in GSE65194 (design-affecting).** 23 tumours have two arrays each, with r about 0.996. Treating them as independent samples would be pseudo-replication. Proposed a-priori rule: average the pair, giving 130 + 11 = 141 units.

**F3. The external class balance and power are small.** Discovery has 17 normals (14%); external has 11 normals (8%). External AUC and any per-gene test involving 11 normals will have wide confidence intervals. Replication must be reported as an effect size with a CI, not a p-value alone.

**F4. Class is blocked by GSM order and partly confounded with processing metadata in GSE42568 (leakage-relevant).**
- GSM order is exactly 17 normals first, then 104 tumours. Any unshuffled K-fold gives catastrophic, class-pure folds. Tests must assert shuffled stratified splits.
- Title prefixes (N, P = normals; S, T = tumours) align perfectly with class.
- Date fields in the titles: 9 of 17 normals were processed in Jan 2005 vs 4 of 104 tumours (Dec 2004: 97 tumours, 8 normals). What the date means (extraction, labelling or hybridization) is not stated. A batch or processing effect could therefore create "stable, predictive" genes that are not tissue biology. This affects H4 and cannot be corrected without removing class signal. Proposed handling: record it as a limitation and run a sensitivity analysis on the Dec-2004-only subset (97 tumours, 8 normals) after freeze.
- GSE65194 has the reverse arrangement (healthy samples spread across five batches), which is more favourable.

**F5. Floor censoring in GSE42568.** 5,350 probes sit at the floor in at least 90% of samples, 18,253 in at least 50%, and 96 in all samples (none has zero variance). Variance filters, t-tests and permutation importance will see many ties. A label-free, fold-internal expression filter (for example, keep probes above the floor in at least a stated fraction of the training samples) is needed. The threshold is a design decision.

**F6. Probe-to-gene collapse needs a rule (leakage-relevant).** 78.5% of probes are usable and 20,848 genes remain; 76.9% of usable probes belong to multi-probe genes. Any rule that uses the data (max mean, IQR) is data-dependent and, per the design principle, must be computed inside each training fold. Alternatives: an a-priori rule with no data dependence, or a probe-level analysis with gene-level aggregation afterwards. Recommendation: a fold-internal, label-free max-mean probe per gene, plus a sensitivity check.

**F7. Restrict the feature universe to probes present in both series.** Two GSE42568 probes (STAT2, CEP128) are absent externally. This is a matrix-structure fact only (no external expression is used), so restricting the discovery universe to the 42,892 shared usable probes ensures every frozen gene can be evaluated. It also gives a clean enrichment background: the genes that could have been selected.

**F8. The annotation is from 2016.** Some symbols will be outdated compared with current HGNC names. Enrichment should therefore use Entrez IDs or a documented symbol-update step.

**F9. Populations and protocols differ.** GSE42568 is ER+ dominated (67 of 101 tumours with known ER), with a two-cycle labelling protocol from a Dublin cohort. GSE65194 is TNBC-enriched (41 of 130 unique tumours), with a one-cycle protocol from a Paris cohort. That is a genuinely independent cohort but not a matched one: genes that are subtype-dependent may fail to replicate for reasons unrelated to instability.

**F10. Unverifiable items.** The source of the GSE42568 normals is not stated (GSE65194 healthy tissue is from mammoplasty). The exact model behind the GSE65194 batch correction, in particular whether class was included as a term, is not stated. Both are limitations to report and cannot be checked from these files.

**Suitability of GSE65194 as the external cohort:** yes, with the above rules (exclude cell lines, average duplicates, scale-free replication endpoints). The literature-derived counts used earlier (41/30/30/29 tumours, 11 normals, 14 cell lines) are now confirmed against the GEO file itself.

## 9. Problems / blockers
None blocking. GEO's web pages still show a CAPTCHA; the FTP endpoints work. The first version of the script crashed on Windows console encoding (fixed with `PYTHONIOENCODING=utf-8`, no data effect).

## 10. Recommended next step
After you review F1-F8, confirm the following a-priori decisions (each is label-free and will be written into the frozen configuration file):
1. External replication endpoints are scale-free (direction, standardized effect size, rank correlation, with CIs), plus a within-cohort-standardized signature score as a secondary endpoint.
2. GSE65194 analysis set = average the 23 duplicate pairs, exclude the 14 cell lines (130 tumours + 11 normals).
3. Feature universe = the 42,892 usable probes present in both series, collapsed to genes with a fold-internal max-mean rule, and a fold-internal floor-based expression filter (threshold to be set on GSE42568 only).
4. The Dec-2004-only sensitivity analysis as a planned check for the processing-date confound.
Then proceed to Steps 3-10 (pytest, package skeleton, tests, first vertical slice), unchanged.
