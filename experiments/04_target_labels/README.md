# 4. Limited Target labels

[Technical report](../../paper/technical_report.pdf): Section 7 and Table 11.

## Question and design

Zero-Target-label estimates remain imperfect. If a small number of Target answers receive human scores, can the current update method reliably improve prediction of the remaining answers?

This experiment returns to historical LOAO CounselBench `overall` and Rwanda `overall11`, and releases simulated Target-label budgets of 5%, 10%, 15%, and 18%. A frozen TextDM supplies initial predictions. Acquired Target scores fit a Source-selected Ridge residual adapter. Active selection uses predicted error risk learned from Source pseudo-Targets together with coverage; Random uses the same label budget.

This changes both label access and the evaluation target relative to the zero-label experiments. It evaluates remaining-answer prediction, rather than claiming to solve the earlier Target-mean estimation problem.

## Results at 18%

CounselBench acquires 18 of 100 scores and evaluates 82 remaining answers. Rwanda acquires 91 of 506 scores and evaluates 415 remaining answers.

| Dataset | Unchanged DM on Active's remainder | Updated model on that same remainder | Update delta and 95% interval |
|---|---:|---:|---|
| CounselBench | 0.502456 | 0.519406 | +0.016951 [−0.009441, +0.044957] |
| Rwanda `overall11` | 0.368224 | 0.353120 | −0.015104 [−0.025732, −0.004678] |

These values are native-scale row MAE; delta means updated minus unchanged, so negative is better. Rwanda shows a local update benefit at this budget. CounselBench does not establish one.

The selection comparison needs a different denominator because Active and Random leave different answers unqueried. **Operational MAE** is the sum of errors on the remaining answers divided by the original total number of answers. Queried answers contribute zero error against the supplied gold, modeling them as manually resolved.

| Dataset | Active − Random operational MAE | 95% interval |
|---|---:|---|
| CounselBench | +0.023218 | [−0.016366, +0.059693] |
| Rwanda `overall11` | +0.017684 | [+0.000465, +0.033421] |

Active does not outperform Random. In Rwanda it is worse on the operational comparison, even though its adapter improves prediction on its own remainder. Updating the predictor and choosing which labels to acquire are therefore distinct questions.

## Interpretation

The update comparison uses exactly the same remainder for both predictors. Operational error can also decline merely by removing difficult answers, so it must not be interpreted as an isolated model-update effect. Historical Rwanda's `overall11` differs from medical-consensus alignment in the formal and diagnostic studies.

The intervals condition on fixed historical responders and offline replays; they are not simultaneous intervals over all budgets. Historical update p-values used different aggregation weights from the displayed deltas and are not attached to these results. These findings concern the current adapter and acquisition rule, not every few-label method, measured annotation time, or stable improvement of the Target mean.

## Result files

- [18% prediction and operational metrics](../../results/target_labels_B18_metrics.csv).
- [Update and acquisition contrasts](../../results/target_labels_B18_contrasts.csv), including recomputed intervals and explicit estimands.

## Related methods and code

The [residual predictor](../../src/ccebench/methods/context.py) illustrates residual fitting on supplied features. The label acquisition policy and budget experiment are described in the report; the package does not provide their experiment runner.
