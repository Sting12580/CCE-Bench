# Direct Method

[Technical report](../../paper/technical_report.pdf): Section 5.1, Section 6.1, Table 3, and Appendix B.1.

## Prediction

Direct Method (DM) fits a score predictor on Source answers and their human labels, applies it to the Target answers, and averages those predictions. Source Mean is the simpler reference: the equally weighted mean of all retained Source score records, used as a constant prediction.

The formal DM encodes questions and answers separately with BGE-M3. A 64-dimensional Source-score-informed projection and a 128-dimensional Source-fitted PCA produce 192 features per text. Concatenating question, answer, and their elementwise product gives 576 inputs to a GradientBoostingRegressor with 200 trees, depth 3, and learning rate 0.05. The learned projection is not full BGE-M3 fine-tuning. Its representation validation uses a seeded 20% Source-row split.

## Findings

DM reduced Formal-12 Macro-nSAE from **0.099236** for Source Mean to **0.069394**, improving 10 of 12 panels. PsycSumEval and Rwanda reversed that result: Source Mean/DM nSAE was **0.087087/0.132005** and **0.027050/0.078953**. Both estimators use historical scores, so the comparison measures the added value of supervised content modeling.

BERT regression is another complete supervised configuration, described under [BERT and domain adaptation](../04_domain_adaptation/README.md). Differences from DM do not isolate the backbone because training, regression, and validation also change.

Results: [formal aggregate](../../results/formal_12_panel.csv), [panel results](../../results/formal_panel_metrics.csv), and [baseline experiments](../../experiments/01_baselines/README.md).

## Research implementation

The formal [`run_m3`](../../research/scripts/run_cce_benchmark_primary_zero_label_methods_v1.py) assembles the learned representation and gradient-boosted regressor. The representation is implemented in [learned_embedding.py](../../research/src/cce_data/estimators/learned_embedding.py), with feature construction in [split_score_estimator.py](../../research/src/cce_data/estimators/split_score_estimator.py). See [research setup](../../research/README.md) for required inputs.

## Small example

[DirectMethod](../../src/ccebench/methods/dm.py) fits a regressor to supplied features. [pair_features and RewardInformedProjection](../../src/ccebench/methods/representations.py) illustrate feature construction and a compact learned transformation. The projection implementation is smaller than the formal representation model; the [code guide](../../docs/code.md) explains these differences.
