# Limitations

These limitations are known from the design and data reconnaissance; results-dependent limitations are added in the final report.

## Data and design
1. **Two cohorts, one platform.** Discovery (GSE42568, Dublin, two-cycle labelling, GC-RMA) and external (GSE65194, Paris, one-cycle labelling, GC-RMA plus submitter batch correction) are independent cohorts but both GPL570. External replication is therefore **not technology-independent validation**.
2. **Small normal groups.** 17 normals (discovery) and 11 healthy (external). Fold-level performance has 3-4 normals per test fold; external effect estimates and every external interval are wide.
3. **Easy discrimination.** Tumour vs normal is near-perfectly separable, so predictive metrics carry little information; the study's value is in explanation stability and replication, not accuracy.
4. **Class and processing confounding in discovery.** GEO order is class-blocked; title prefix aligns perfectly with class; 9 of 17 normals were processed in Jan 2005 versus 4 of 104 tumours. The Dec-2004 sensitivity (8 normals) is small and does not remove the confounding.
5. **Tissue composition.** Normal breast tissue is adipose- and stroma-rich; stable and replicating genes may reflect cell-type composition and not tumour-specific biology. Without cell-type deconvolution this cannot be separated; the marker-panel checks are indicative only.
6. **Subtype composition differs** (discovery ER+ dominated; external TNBC-enriched), and the submitter's batch-correction design for GSE65194 is not stated (whether class was in the model is unknown).
7. **Technical duplicates.** 23 external tumours were measured twice; they are averaged. The matrix is not the averaged data the submitter describes.
8. **Annotation from 2016.** Some symbols are outdated; 17.5% of probes are unannotated and 4% multi-gene and are excluded; many top entries are LOC/pseudogene/antisense identifiers that enrichment libraries do not cover.
9. **Floor censoring** (28.5% of GSE42568 values at the array floor) affects filters, ties and permutation-based methods.

## Methods
10. **Repeated cross-validation is not independent sampling.** Repeats reuse the same 121 samples; per-repeat values describe spread, not a confidence interval. Nogueira's analytic interval is invalid for overlapping training sets and is not reported.
11. **Null design.** The permutation null uses 30 replicates of one 5-fold repeat each; the smallest attainable exceedance fraction is 1/31, so no p-value is claimed. A global label permutation also destroys the date/prefix confounding.
12. **Sparse tree explanations.** XGBoost uses about 25 genes and RF about 6% of genes; with the strictly-positive rule the top-k of these models is (nearly) the model's whole support, so explainer agreement within XGBoost is support collapse, not independent agreement.
13. **Permutation importance is defined on the training fold** (model reliance, not generalisation value) and is diluted by correlated genes. Magnitudes are not comparable across models or with SHAP or coefficients.
14. **Regularisation matters.** LR coefficient-ranking stability at the primary C=1 is lower than at any other tested C (0.01-100) at identical predictive performance; primary results are reported at C=1 as pre-specified together with this sensitivity.
15. **Fixed model configurations, no tuning.** Different hyper-parameters may give different explanations; this is itself a finding for LR but was not explored for the other models.
16. **Enrichment** is contextual, uses live-service gene-set libraries downloaded once, and is limited by outdated symbols and non-coding identifiers.

## Interpretation limits (always)
Model-supported signals are **not** causal genes, **not** validated biomarkers, and not clinically actionable. High predictive performance does not imply stable, replicable or biologically specific features.
