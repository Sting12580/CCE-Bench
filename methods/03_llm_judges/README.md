# LLM scoring and judge combinations

[Technical report](../../paper/technical_report.pdf): Section 5.5, Sections 6.1–6.2, Table 4, and Appendix D.

## Direct scores and Context extensions

The four-panel diagnostic uses the same `gpt-4.1-mini` judge for CounselBench, PsycSumEval, Rwanda, and Warmth–Substance. The formal Target is removed before a Source responder becomes the diagnostic Target.

| Configuration | Information flow | Macro-nSAE |
|---|---|---:|
| `J0_RAW` | Rubric, question, Target answer, and anonymous Source answer texts produce a direct LLM score; Source numerical scores are omitted | 0.194542 |
| `J0_CAL` | C* features plus the LLM score and missingness indicator feed a Source-supervised residual regressor; its correction is added to DM and clipped | 0.059694 |
| `R0` / C* | Selected Context prediction without the added LLM score | 0.032194 |
| `R2` | C* with additional LLM-extracted rubric content and relation features | 0.063003 |

`J0_CAL` is a combined supervised predictor, not a one-dimensional score calibration or LLM fine-tuning. Invalid judge outputs use a DM fallback and a missing indicator. R0 reproduces C* predictions exactly.

The combined predictor improves on raw scoring at these point estimates, but neither added LLM scores nor rubric features improve on Context alone. R2 worsens 11 of 12 panel-by-seed cells. Independent semantic acceptance remains incomplete. These findings do not establish that LLM scoring or semantic information is generally ineffective, and J0_RAW does not test in-context examples that include numeric scores.

Results: [four-panel aggregates](../../results/medical_rubric_macro.csv), [panel metrics](../../results/medical_rubric_panel_metrics.csv), and [baseline experiments](../../experiments/01_baselines/README.md).

## Separate judge calibration

The Weighted Judge study combines five donor scores with nonnegative weights summing to one, an intercept, and clipping. Its grouped-test RMSE is **0.190414**, compared with **0.192252** for an unweighted mean. This uses development human labels and different data and score dimensions; it does not enter the Formal-12 or four-panel rankings.

[Judge calibration results](../../results/judge_calibration.csv) preserve that separate comparison. The reference [SimplexJudgeCombiner](../../src/ccebench/methods/weighted_judge.py) implements the constrained weights on supplied scores, without the study's intercept and clipping. LLM scoring and feature extraction are described in the report rather than provided as API workflows.
