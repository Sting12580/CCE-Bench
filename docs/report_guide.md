# Report guide

The [technical report](../paper/technical_report.pdf) includes the main text and appendices. Its [text version](../paper/technical_report.md) provides the same equations, results, and references.

| Report location | Contents | Data and code entry |
|---|---|---|
| Table 1; Appendix A | Panels, score scales, retained records, and Target responders | [Panel definitions](../results/formal_panel_contract.csv), [diagnostic splits](../results/diagnostic_panel_contract.csv) |
| Table 2; Sections 3–4 | Separate protocols and evaluation rules | [Settings](settings.md), [metrics](metrics.md) |
| Section 5; Appendix B | Estimators and Source-only selection | [Method catalog](../methods/README.md), [code guide](code.md) |
| Table 3; Figure 1 | Formal configurations and per-panel DM comparison | [Formal summary](../results/formal_12_panel.csv), [panel results](../results/formal_panel_metrics.csv) |
| Table 4 | Direct LLM scores and supervised Context-plus-score prediction | [Medical scoring results](../results/medical_rubric_macro.csv) |
| Table 5 | Matched adversarial and residual-weighting comparisons | [Paired contrasts](../results/formal_matched_contrasts.csv) |
| Table 6 | Context, score removal/shuffling, and simple same-item aggregation | [Historical Context](../results/context_historical.csv), [contrasts](../results/context_historical_contrasts.csv) |
| Table 7 | Actual C* selection and panel-specific outcomes | [Panel results](../results/context_stage1_panel_metrics.csv), [selected configurations](../results/context_stage1_source_selection.csv) |
| Table 8; Figure 2 | F7 projected to the DM prediction mean | [Projection results](../results/context_projection_panel_metrics.csv) |
| Tables 9–10; Figure 3 | Objective/selection and additional-Source-evidence comparisons | [Design](../results/diagnostic_factor_design.csv), [results](../results/diagnostic_factor_macro_metrics.csv), [intervals](../results/diagnostic_factor_clustered_contrasts.csv) |
| Table 11 | Limited Target labels, model updates, and Active/Random selection | [Budget results](../results/target_labels_B18_metrics.csv), [contrasts](../results/target_labels_B18_contrasts.csv) |
| Appendix D | Judge calibration, F7 follow-ups, and other exploratory branches | [Supplementary studies](discussion.md), [result index](../results/README.md) |

Each experiment guide states the question, permitted inputs, comparison, and conclusion. The CSV files retain method identifiers used in the report; aliases such as R0/C* and the distinction between C* and C_SOURCE_ROW are explained in the [result index](../results/README.md).
