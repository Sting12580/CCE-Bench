# Code guide

The repository includes the formal-method research code and a smaller set of components for learning and reuse.

## Formal benchmark implementations

[research/](../research/README.md) contains the implementations used by the formal benchmark. Its original module and function names are preserved so that the method descriptions can be followed directly into the code.

| Report method | Implementation | Experiment entry |
|---|---|---|
| Source Mean; BGE-M3 DM (Sections 5.1, 6.1) | [Core utilities](../research/src/cce_data/estimators/cce_primary_zero_label.py), [learned representation](../research/src/cce_data/estimators/learned_embedding.py), [score regression](../research/src/cce_data/estimators/split_score_estimator.py) | [Formal-method runner](../research/scripts/run_cce_benchmark_primary_zero_label_methods_v1.py) |
| BERT regression and DANN (Sections 5.3, 6.2) | [BERT fine-tuning and domain adaptation](../research/src/cce_data/estimators/dann_finetuned_encoder.py) | The formal runner pairs coefficient 0.1 with its coefficient-zero control. |
| Corrected SN-MIPS and SNDR (Sections 5.2, 6.2) | [Corrected estimators](../research/src/cce_data/estimators/cce_primary_zero_label_m13_m14_v2.py), [density-ratio models](../research/src/cce_data/estimators/density_ratio.py) | [Corrected-weighting runner](../research/scripts/run_cce_benchmark_primary_m13_m14_corrected_v2.py) |

The corrected SN-MIPS representation is trained with Source scores. Its item cross-fitting and nested calibration differ from a single classifier fitted to frozen embeddings. SNDR uses same-seed DM predictions, out-of-fold Source residuals, and per-answer clipping before averaging.

The research code needs the corresponding prepared data, configuration files, frozen inputs, and model snapshots. These assets and the evaluator are not bundled. [Research setup](../research/README.md) documents the dependencies and expected inputs; the repository does not provide a complete raw-data-to-results replay.

## Small reusable components

| Module | Operation |
|---|---|
| [dm.py](../src/ccebench/methods/dm.py) | Fit an outcome regressor on numeric features and estimate the Target mean. |
| [representations.py](../src/ccebench/methods/representations.py) | Form prompt/answer interaction features and learn a small supervised projection. |
| [ope.py](../src/ccebench/methods/ope.py) | Fit logistic density ratios, compute weighted means and effective sample size, and illustrate a residual correction. |
| [domain_adaptation.py](../src/ccebench/methods/domain_adaptation.py) | Demonstrate gradient reversal on precomputed features. |
| [context.py](../src/ccebench/methods/context.py) | Learn a clipped residual correction from supplied Context features. |
| [context_features.py](../src/ccebench/methods/context_features.py) | Define F0–F8, F7, and reduced feature subsets. |
| [weighted_judge.py](../src/ccebench/methods/weighted_judge.py) | Fit a nonnegative judge combination with weights summing to one. |
| [metrics.py](../src/ccebench/metrics.py) | Compute row errors and responder-mean errors separately. |

These components power the [synthetic example](../examples/quickstart.py). The projection is a small MLP rather than the research two-tower representation, and the DANN example operates on arrays rather than fine-tuning BERT. The `offcem_style` example uses in-sample residuals and a scalar correction; the reported SNDR uses the corrected implementation above. The simplex judge example does not reproduce the complete historical calibration pipeline.

## Context and subsequent studies

The complete Context feature extraction and selection pipelines, LLM scoring and rubric extraction, mean-projection experiment, 11-panel objective study, and Target-label query policy are not included as executable pipelines. Their inputs, model choices, comparisons, and results are described in the [experiment guides](../experiments/README.md) and report Appendices B–D. The feature definitions, selected configurations, and numerical tables are included where applicable.

The tests exercise the reusable components. They do not fit the research models or download datasets.
