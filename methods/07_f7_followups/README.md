# F7 and follow-ups

[Technical report](../../paper/technical_report.pdf): Section 6.3, Table 8 and Figure 2, and Appendix D.

## Fixed-feature and fixed-mean diagnostics

F7 retains 19 of the 37 Context features, excluding explicit human/Agent role features while retaining the DM prediction, support availability, Source score summaries, similarity-based quantities, and length differences. It predicts a residual around DM. F7 is a fixed family examined retrospectively; [C*](../06_context_calibration/README.md) is the actual overall Source-only selection procedure.

In the Rwanda alignment diagnostic, F7 improves nMAE from DM's **0.100357 to 0.077530**, but worsens nSAE from **0.003134 to 0.009802**. Projecting F7 to the **DM predicted mean**, with clipping included in the constraint, retains nMAE **0.082193**. The identical nSAE **0.003134** is imposed by that constraint. This demonstrates useful answer-level structure at a fixed prediction mean, without establishing a better estimate of the true mean.

## Alternative features and heads

Follow-ups test 1-, 3-, 5-, and 9-feature subsets, linear and RBF SVR residual heads, and separate mean/shape optimization. None meets its required replacement criterion across the relevant comparisons.

CounselBench's 9-feature subset has nSAE **0.013916**, versus **0.022718** for F7-19, but does not establish stable improvement under the study's comparison correction. The SVR results show another mean-versus-row tradeoff:

| Panel | F7 nSAE | Selected SVR nSAE |
|---|---:|---:|
| CounselBench | 0.022718 | 0.040079 |
| PsycSumEval | 0.010382 | 0.026357 |
| Rwanda alignment | 0.009802 | 0.015336 |
| Warmth–Substance | 0.054953 | 0.060958 |

SVR improves row MAE in the first three panels but worsens mean error in all four. These local findings neither establish a universal F7 replacement nor reject SVR in general. Appendix D also describes the TabNet and self-supervised alternatives.

Results: [projection metrics](../../results/context_projection_panel_metrics.csv), [projection contrasts](../../results/context_projection_contrasts.csv), [feature/head follow-ups](../../results/f7_followups.csv), and [error diagnostics](../../experiments/03_error_diagnostics/README.md). In the projection table, `A` denotes F7 and `P_A` its projection to the DM prediction mean.

## Reference code

[F7_FEATURES and F7_REDUCED_FEATURE_SETS](../../src/ccebench/methods/context_features.py) define the feature subsets. The package provides those definitions and the generic Context residual model; the projection and model-selection experiments are described in the report.
