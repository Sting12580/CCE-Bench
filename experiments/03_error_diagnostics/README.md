# 3. Mean and row error diagnostics

[Technical report](../../paper/technical_report.pdf): Section 6.3, Tables 7–10 and Figures 2–3; Appendix C gives the uncertainty definitions.

## Question and design

A model may predict individual answers more accurately while shifting their average in the wrong direction. We examine that distinction, retain the actual Source-only selection procedure, and then test whether a fixed-mean projection, changed training and selection objectives, or additional Source evidence resolves it.

## Results

### The two metrics can reverse the ranking

In Formal-12, DM has lower nMAE than DANN (**0.146491 vs 0.151106**), while DANN has lower nSAE (**0.052826 vs 0.069394**). In historical Rwanda, Context lowers native row MAE from **0.395944 to 0.303285**, but increases mean per-seed responder SAE from **0.138317 to 0.147831**. These are separate settings, each showing that the metrics answer different questions.

### Source-only selection has Target counterexamples

The four-panel diagnostic first removes the formal Target, then holds out a Source responder. C* selects among F0–F8 and six Ridge/GBR heads using Source validation. Its twelve panel×seed selections choose F4 four times, F6 seven times, and F7 once. Fixed F7 is a later, post-hoc candidate; its lower inspected aggregate cannot replace the selected procedure.

| Diagnostic panel | DM nSAE | C* nSAE | Direction |
|---|---:|---:|---|
| CounselBench | 0.065472 | 0.017963 | Improved |
| PsycSumEval | 0.055881 | 0.019328 | Improved |
| Rwanda alignment | 0.003134 | 0.028520 | Worsened |
| Warmth–Substance | 0.054546 | 0.062965 | Worsened |

C* has Macro-nSAE 0.032194, versus 0.044758 for DM, but improves only two of the four panels. Its conditional 95% difference interval [−0.030581, +0.012425] crosses zero. Source-only selection protects label isolation; it does not guarantee improvement on Target.

For Rwanda, project F7 predictions into the allowed score range while preserving the **DM prediction mean**, using prediction vectors alone:

| Rwanda alignment diagnostic | nSAE | nMAE |
|---|---:|---:|
| DM | 0.003134 | 0.100357 |
| F7 | 0.009802 | 0.077530 |
| F7 projected to DM prediction mean | 0.003134 | 0.082193 |

Equal nSAE follows from the imposed mean constraint. The empirical finding is that projected F7 retains better row accuracy. The projection supplies no new information about the true Target mean and does not validate a deployment rule for choosing when to use it.

### Eleven-panel objective and evidence comparison

A separate Source-held-out diagnostic excludes Well-being and contains 4,872 Target answers per seed. All four variants learn residual corrections on the same Source-trained DM, using Ridge heads, the same alpha candidates, and Source item folds. Original DM has Macro-nSAE **0.056627**.

| Inputs | Row-oriented fitting and selection | Mean-oriented fitting and selection |
|---|---:|---:|
| 50 base/text features | E: 0.064220 | B: 0.073120 |
| The same features plus 59 Source score/text/relation features | C: 0.061652 | D: 0.066796 |

The row configuration fits weighted pair-row squared residuals and selects by clipped responder-equal MSE. The mean configuration fits responder-mean squared residuals plus 0.1 row loss and selects by clipped responder-equal nSAE. Thus each horizontal comparison changes both fitting and selection; each vertical comparison adds scores, text, and relations together. The C cell is `C_SOURCE_ROW`, not Stage1 C*.

The mean configuration worsens Macro-nSAE in both matched comparisons: B−E **+0.008900** and D−C **+0.005145**, with multiplicity-adjusted intervals above zero. Additional evidence improves the point estimates (C−E **−0.002568**, D−B **−0.006323**), but both intervals cross zero. **All four residual variants remain worse than original DM at their point estimates.** A better row configuration within this grid therefore does not establish a benefit from adding the residual model.

## Interpretation

These are retrospective diagnostics on fixed responders. The eleven-panel intervals use shared cluster resampling across methods and seeds, with WMT documents, Warmth scenarios, and other items as clusters. The four matched contrasts use a 17-comparison correction; D−DM has a separate, unadjusted 95% conditional interval and must not be assigned that same correction. Item-fold validation does not supply a second level of unseen-responder validation.

Historical Context tables used distinct seed-aggregation estimands for some point values and intervals; the per-seed SAE values above are not paired with an interval for seed-averaged predictions. F7 compression, SVR, and split mean/shape variants remain negative or local follow-ups: none established the required replacement, and no fixed F7 variant is presented as a universally best final method.

## Result files

- [Formal row and mean metrics](../../results/formal_12_panel.csv).
- [Historical Context row and normalized mean metrics](../../results/context_historical.csv); multiply nSAE by the native scale width of 4 for the displayed historical SAE.
- [C* and F7 aggregates](../../results/context_stage1_summary.csv), [panel metrics](../../results/context_stage1_panel_metrics.csv), and [actual selections](../../results/context_stage1_source_selection.csv), and [paired contrasts](../../results/context_stage1_contrasts.csv).
- [Projection metrics](../../results/context_projection_panel_metrics.csv), and [contrasts](../../results/context_projection_contrasts.csv). The projection table uses `A` for F7 and `P_A` for its projection to the DM prediction mean.
- [Eleven-panel design](../../results/diagnostic_factor_design.csv), [macro results](../../results/diagnostic_factor_macro_metrics.csv), [panel results](../../results/diagnostic_factor_panel_metrics.csv), and [clustered contrasts](../../results/diagnostic_factor_clustered_contrasts.csv).
- [F7 follow-ups](../../results/f7_followups.csv).

## Related methods and code

The package provides [Context feature sets](../../src/ccebench/methods/context_features.py), a [residual predictor](../../src/ccebench/methods/context.py), and [metrics](../../src/ccebench/metrics.py). See [C* selection](../../methods/06_context_calibration/README.md) and [F7 diagnostics](../../methods/07_f7_followups/README.md) for the experimental configurations.
