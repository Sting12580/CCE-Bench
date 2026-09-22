# Results

These tables accompany the [technical report](../paper/technical_report.pdf). The [report guide](../docs/report_guide.md) maps its tables and figures to these files. All results are retrospective. Comparisons require the same responder split, scoring dimension, label budget, and metric.

## Benchmark and formal methods

| File | Contents |
|---|---|
| [formal_panel_contract.csv](formal_panel_contract.csv) | Fixed Targets, score scales, record counts, aggregation, exclusion counts, and clustering. |
| [diagnostic_panel_contract.csv](diagnostic_panel_contract.csv) | Diagnostic responders and excluded formal Targets; `main=True` selects the 11-panel objective study. |
| [formal_method_information.csv](formal_method_information.csv) | Source/Target information access, output type, and validation procedure. |
| [formal_12_panel.csv](formal_12_panel.csv) | Six configurations, with macro mean and row errors. |
| [formal_panel_metrics.csv](formal_panel_metrics.csv) | All 72 panel-method combinations, including DM's two reversals. |
| [formal_matched_contrasts.csv](formal_matched_contrasts.csv) | Paired contrasts under the document/scenario/item bootstrap. |
| [formal_resampling_contrasts.csv](formal_resampling_contrasts.csv) | Shared-Poisson sensitivity intervals from Appendix C. |
| [formal_weight_diagnostics.csv](formal_weight_diagnostics.csv) | Effective sample size, weight variability, and clipping by panel. |

BERT coefficient zero is an auxiliary matched control. Corrected SN-MIPS and SNDR are post-gold retrospective analyses. SN-MIPS learns its representation from Source scores and outputs a system value, so its row-error fields are empty. `Project SNDR` and `SNDR` identify the same configuration.

## Context and LLM information

| File | Contents |
|---|---|
| [context_historical.csv](context_historical.csv) | TextDM, Context, score removal/shuffling, similarity removal, and Agent median under historical LOAO. |
| [context_historical_contrasts.csv](context_historical_contrasts.csv) | Paired question-cluster comparisons for the Context controls. |
| [context_selected_gates.csv](context_selected_gates.csv) | The 21 selected historical gates and their Source selection criteria. |
| [medical_rubric_macro.csv](medical_rubric_macro.csv) | DM, earlier Context, direct LLM scoring, Context plus LLM scores, C*, and semantic features. |
| [medical_rubric_panel_metrics.csv](medical_rubric_panel_metrics.csv) | Results on the four medical-related panels. |
| [medical_rubric_contrasts.csv](medical_rubric_contrasts.csv) | R2 comparisons with unadjusted 95% conditional intervals. |

Historical Rwanda uses `overall11`; the formal and four-panel diagnostic studies use medical-consensus alignment. Historical native SAE is included alongside nSAE; the scale width is four. The selected Context gates are constant. R0 replays C* and appears once with that alias. Earlier `CONTEXT` is a different configuration. `J0_CAL` is a supervised Context-plus-score residual predictor. The semantic-feature study has not passed independent semantic validation.

## Selection and mean-error diagnostics

| File | Contents |
|---|---|
| [context_stage1_summary.csv](context_stage1_summary.csv) | DM, actual C* selection, and fixed F7 macro results. |
| [context_stage1_panel_metrics.csv](context_stage1_panel_metrics.csv) | Stage1 panel results, including C*'s two improvements and two deteriorations. |
| [context_stage1_source_selection.csv](context_stage1_source_selection.csv) | Selected feature/head configurations and Source-validation values. |
| [context_stage1_contrasts.csv](context_stage1_contrasts.csv) | Ordinary and Bonferroni-47 intervals, with scenario sensitivity. |
| [context_projection_panel_metrics.csv](context_projection_panel_metrics.csv) | F7 and Context before and after projection to the DM prediction mean. |
| [context_projection_contrasts.csv](context_projection_contrasts.csv) | Projection contrasts and their multiplicity-adjusted intervals. |
| [diagnostic_factor_design.csv](diagnostic_factor_design.csv) | The four input-evidence and training/selection configurations. |
| [diagnostic_factor_macro_metrics.csv](diagnostic_factor_macro_metrics.csv) | Original DM and the four residual variants. |
| [diagnostic_factor_panel_metrics.csv](diagnostic_factor_panel_metrics.csv) | The five configurations on each of the 11 panels. |
| [diagnostic_factor_clustered_contrasts.csv](diagnostic_factor_clustered_contrasts.csv) | Effects, interval coverage, correction-family sizes, and clustering. |

`CSTAR_SELECTED` is the Source-selected family/head procedure. `F7_SELECTED` identifies a fixed feature family with a Source-selected head; it was highlighted after inspecting results. In the projection tables, `A` means F7 and `P_A` means F7 projected to the predicted DM mean. Equal mean error after projection follows from that constraint. Neither identifier is the factor study's `C_SOURCE_ROW` or `D_SOURCE_SYSTEM`.

The factor study excludes Well-being. Its horizontal comparisons change fitting and selection together. Secondary contrasts retain the 17-comparison correction, even when a smaller subset is displayed; D minus original DM has a separate unadjusted 95% interval.

## Limited Target labels

| File | Contents |
|---|---|
| [target_label_budget_curve.csv](target_label_budget_curve.csv) | Aggregate results at 0%, 5%, 10%, 15%, and 18% label budgets. |
| [target_labels_B18_metrics.csv](target_labels_B18_metrics.csv) | The 18% comparison reported in Table 11. |
| [target_labels_B18_contrasts.csv](target_labels_B18_contrasts.csv) | Update and acquisition effects with paired 95% intervals. |

Model updates are compared on the same unlabelled remainder. Active and Random acquisition are compared using operational MAE: errors on remaining answers divided by the original batch size, with queried answers assigned their human scores. These are different estimands. Intervals condition on the historical responders and are not simultaneous across budgets.

## Supplementary studies

| File | Contents |
|---|---|
| [dm_backbone_targeted.csv](dm_backbone_targeted.csv) | BERT and BGE-M3/projection/GBDT on two targeted historical folds. |
| [judge_calibration.csv](judge_calibration.csv) | Independent grouped calibration using five judge combinations. |
| [f7_followups.csv](f7_followups.csv) | Feature reduction and SVR-head results. |
| [diagnostic_factor_supplementary_macro_metrics.csv](diagnostic_factor_supplementary_macro_metrics.csv) | Nineteen additional objective-study controls and variants. |
| [supplementary_methods.csv](supplementary_methods.csv) | Selected PairDelta, Masked Bridge, reliability/routing, and SSL results from Appendix D. |

## Column conventions

`setting` identifies the [protocol](../docs/settings.md); `analysis_scope` distinguishes fixed-target, exploratory, diagnostic, and post-gold analyses. `role` identifies primary methods and auxiliary controls. Method names otherwise follow the report or its implementation identifiers.

SAE measures error in the responder mean; MAE measures individual-answer error. Prefix `n` denotes division by the native score range. Macro values weight panels equally. Seed standard deviations describe training variability, not paired-comparison uncertainty. Lower error is better. For a contrast A minus B, a negative value favors A unless the table explicitly defines another direction. Confidence intervals condition on fixed fitted models and responders; an interval crossing zero does not establish equivalence.
