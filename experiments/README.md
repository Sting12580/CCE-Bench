# Experiments

CCE-Bench asks whether existing human scores can estimate the mean score of a responder whose answers are visible but whose scores are withheld. The experiments establish baselines, test additional information and correction methods, examine mean-estimation errors, and then allow a limited number of Target labels.

| Research question | Experiment | Technical report |
|---|---|---|
| Which basic scoring routes estimate the Target mean accurately? | [Baselines and direct scoring](01_baselines/README.md) | Section 6.1; Tables 3–4; Figure 1 |
| Can extensions reduce error beyond their corresponding baselines? | [DM extensions and parallel approaches](02_dm_extensions/README.md) | Section 6.2; Tables 5–6 |
| Why can row accuracy improve while mean accuracy worsens? | [Mean and row error diagnostics](03_error_diagnostics/README.md) | Section 6.3; Tables 7–10; Figures 2–3 |
| Do a few Target labels improve prediction of the remaining answers? | [Limited Target labels](04_target_labels/README.md) | Section 7; Table 11 |

The [technical report](../paper/technical_report.pdf) gives the full study. Its Section 4 and Table 2 separate Formal-12, historical LOAO, Source-held-out diagnostics, judge calibration, and limited-label experiments. Results from these settings do not form one combined ranking.

[nSAE measures responder-mean error; nMAE measures answer-level error](../docs/metrics.md). Metrics are calculated within each panel and seed before aggregation. Repeated seeds use the same responders. All findings are retrospective, and corrected SN-MIPS and SNDR retain their post-gold design-correction status.

See the [result tables](../results/README.md), [method descriptions](../methods/README.md), and [code guide](../docs/code.md). The package contains small reference implementations; the report describes the complete experimental configurations.
