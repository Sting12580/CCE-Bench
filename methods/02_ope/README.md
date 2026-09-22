# Weighting and residual correction

[Technical report](../../paper/technical_report.pdf): Section 5.2, Section 6.2, Tables 3 and 5, and Appendix B.1.

## SN-MIPS

SN-MIPS estimates the Target mean by reweighting Source scores:

```text
value = sum(weight * Source score) / sum(weight)
```

The corrected formal estimator uses Source-score-informed BGE-M3 `learned576` features for logistic density estimation, with nested Platt calibration. Strict item cross-fitting excludes held-out items throughout representation, PCA, density, and calibration fitting. Source scores inform the representation; the Target batch is score-free. Probabilities and odds are clipped before self-normalization. Its native output is a system value, so answer-level nMAE and nRMSE are not defined.

## SNDR

SNDR begins with the same-seed frozen DM predictions. It calculates a weighted mean of Source residuals using out-of-fold predictions, adds that correction to each Target prediction, clips each corrected prediction to the score range, and then averages:

```text
correction = weighted_mean(Source score - out_of_fold_DM_prediction)
prediction = clip(Target_DM_prediction + correction)
value = mean(prediction)
```

Clipping each answer matters: the resulting value need not equal an unclipped DM mean plus the correction. This is an OffCEM-inspired project analogue, not an official OffCEM reproduction or a causal-value guarantee.

## Findings

SN-MIPS achieved Macro-nSAE **0.075532**, below Source Mean's **0.099236** but above DM's **0.069394**. SNDR reached **0.066565**; its difference from DM was **−0.002829**, with conditional 95% interval **[−0.011218, +0.008446]**. The residual correction therefore did not establish an incremental benefit. Both corrected estimators retain **post-gold retrospective** status.

Results: [formal aggregate](../../results/formal_12_panel.csv), [matched contrasts](../../results/formal_matched_contrasts.csv), and [extension experiments](../../experiments/02_dm_extensions/README.md).

## Research implementation

The corrected estimator is [`crossfit_reward_informed_sn_mips_v2`](../../research/src/cce_data/estimators/cce_primary_zero_label_m13_m14_v2.py). The [corrected runner](../../research/scripts/run_cce_benchmark_primary_m13_m14_corrected_v2.py) combines its weights with frozen DM and out-of-fold Source predictions for SNDR. [`sndr_predictions`](../../research/src/cce_data/estimators/cce_primary_zero_label.py) performs the row-level correction and clipping.

## Small example

[fit_density_ratio, mips, and offcem_style](../../src/ccebench/methods/ope.py) implement compact weighting and residual-estimation examples. `offcem_style` uses in-sample residuals and an unclipped value correction; it does not reproduce the formal SNDR cross-fitting and per-answer clipping procedure.
