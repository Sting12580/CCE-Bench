# BERT and domain adaptation

[Technical report](../../paper/technical_report.pdf): Sections 5.1 and 5.3, Section 6.2, Tables 3 and 5, and Appendix B.1.

## Matched regression models

BERT is the encoder; DANN adds domain-adversarial training. The formal configuration uses BERT-base-uncased on question–answer pairs, CLS pooling, and reward and domain heads, with the final two encoder layers trainable. Source scores supervise the reward head. A domain classifier distinguishes Source from score-free Target inputs, while gradient reversal discourages that distinction in the shared encoder.

DANN uses an adversarial coefficient of `lambda=0.1`. The matched `lambda=0` auxiliary control executes the same domain batches but blocks their adversarial gradient into the encoder. Initialization, batch schedules, validation indices, and eligible checkpoints match across the 36 panel-by-seed pairs. Source validation can select different epochs for the two models.

## Findings

DANN's Formal-12 Macro-nSAE is **0.052826**, compared with **0.052713** for BERT `lambda=0`. The difference is **+0.000113**, with conditional 95% interval **[−0.002084, +0.002163]**. This does not establish an incremental adversarial benefit; it is not an equivalence result.

BERT `lambda=0` has a lower point estimate than the BGE-M3 DM, but remains an auxiliary control. That cross-configuration comparison changes representation, regression, training, and validation together, so it cannot identify a backbone or domain-adaptation effect.

Results: [formal aggregate](../../results/formal_12_panel.csv), [matched contrasts](../../results/formal_matched_contrasts.csv), and [extension experiments](../../experiments/02_dm_extensions/README.md).

## Research implementation

[`fit_fixed_fold_bert_dann_pair`](../../research/src/cce_data/estimators/cce_primary_zero_label.py) builds the matched pair with helpers from [dann_finetuned_encoder.py](../../research/src/cce_data/estimators/dann_finetuned_encoder.py). The [formal runner](../../research/scripts/run_cce_benchmark_primary_zero_label_methods_v1.py) supplies the data, folds, and method configuration.

## Small example

[DANNRegressor and fit_dann](../../src/ccebench/methods/domain_adaptation.py) demonstrate gradient reversal in a small network operating on supplied features. They do not implement the report's BERT fine-tuning pipeline. PyTorch is optional: install with `pip install -e ".[domain]"`.
