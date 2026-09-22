# Benchmark

CCE-Bench estimates a held-out responder's average human score on questions already answered by other responders. Its 12 formal panels span 11 dataset families. Each seed uses 37,842 Source response-score records and 4,922 Target answers.

| Element | Definition |
|---|---|
| Source | Eligible answers and human scores from the other responders. |
| Target | One fixed responder per panel. Its answers are visible; its scores are withheld during fitting and selection. |
| Task | Fixed score dimension, scale, rating aggregation, retained questions, and responder identities. |
| Output | A predicted score for each Target answer, averaged into a mean, or a directly estimated mean. |
| Evaluation | Predictions or values are frozen before evaluation against Target scores. Panel metrics are computed per seed and then averaged with equal panel weights. |

[Panel definitions](panels.md) list the formal Targets and scoring rules. [Protocol definitions](settings.md) distinguish the formal benchmark from historical LOAO, Source-held-out diagnostics, judge calibration, and Target-label experiments.

## Source validation

All formal methods share the outer Target holdout, but their internal validation differs. DM uses a historical 20% Source-row split for representation learning. BERT and DANN use one fixed item fold per seed for validation and four for training. Corrected SN-MIPS uses item cross-fitting and nested calibration; SNDR reuses same-seed out-of-fold Source predictions. Details appear in [method information](../results/formal_method_information.csv) and Appendix B of the [report](../paper/technical_report.pdf).

Freezing predictions protects the execution's label boundary. It does not remove the effects of earlier result inspection during development. The reported experiments are retrospective; the corrected weighting methods are explicitly marked as post-gold analyses.
