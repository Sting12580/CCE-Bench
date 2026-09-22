# Context residual prediction

[Technical report](../../paper/technical_report.pdf): Section 5.4, Section 6.2, Table 6, and Appendix B.2.

## Same-question support

Context uses other scored answers to the same question to predict a correction to DM. Features summarize Source scores, answer similarities, support availability, and related quantities. With an optional gate, the normalized prediction is:

```text
prediction = clip(DM_prediction + gate * predicted_residual, 0, 1)
```

The historical model uses 37 features. Its ablations remove context-score or similarity features; a shuffled-score control breaks the pairing between Source answers and scores while retaining each Source responder's score distribution. Removing context-score features still leaves the Source-trained DM and supervised residual targets.

## Findings

In historical Rwanda, Context's native row MAE is **0.303285**, versus **0.392949** without context-score features and **0.396953** with shuffled scores. Correct pairing helps this pipeline, but the simple same-question Agent median is stronger at **0.288358**. The median uses available peer Agent scores for each question, excluding Human and Target responders.

CounselBench does not establish the same score-channel benefit: Context is **0.483443**, versus **0.479736** without scores and **0.484827** with shuffled scores. Removing similarity worsens its MAE to **0.520336**. All 21 selected historical Context models use constant gates, so their performance does not demonstrate response-dependent gating.

These are historical LOAO comparisons. Rwanda uses `overall11`, rather than the formal benchmark's medical-consensus alignment score. The [Source-held-out selection study](../06_context_calibration/README.md) is a separate experiment.

Results: [historical controls](../../results/context_historical.csv), [paired contrasts](../../results/context_historical_contrasts.csv), and [extension experiments](../../experiments/02_dm_extensions/README.md).

## Reference code

[ContextResidualBridge](../../src/ccebench/methods/context.py) fits and applies a scaled residual using supplied Context features and base predictions. It implements the prediction operation without the historical feature extraction and gate-selection search.
