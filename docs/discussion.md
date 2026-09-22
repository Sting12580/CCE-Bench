# Discussion and supplementary branches

Historical human scores can be reused conditionally. On the fixed benchmark, DM improves Macro-nSAE over Source Mean on 10 of 12 panels. This does not establish a universally best method, reliable unseen-responder selection, or new-question generalization.

The matched BERT comparison does not support an adversarial increment. SNDR's additional gain is unconfirmed. Same-item scores help some Context configurations, but a simple Agent median remains a necessary comparator. Source-only selection preserves label isolation; it does not guarantee better Target predictions. Row accuracy and mean accuracy require separate evaluation. A small Target-label budget improves the current update on one historical dataset and does not establish a generally superior query policy.

## Supplementary work

| Branch | Placement and evidence status |
|---|---|
| Weighted LLM Judge | Separate [Judge-Cal table](../results/judge_calibration.csv); grouped-test RMSE, not formal Macro-nSAE. The result uses a separate calibration dataset and does not establish a zero-label scoring method. |
| Targeted BERT/DM comparison | [Historical backbone table](../results/dm_backbone_targeted.csv); two selected folds, no general ordering. |
| Reduced F7 and SVR | [Follow-up summaries](../results/f7_followups.csv); no supported universal replacement. Split/mean, TabNet, and SSL remain negative/exploratory branches; complete replay is not supplied. |
| PairDelta, Masked Bridge, human-only, SSA, routers and gates | Historical support and pairing exploration; selected 11-panel variants may appear in supplementary tables under that diagnostic protocol. |
| DeBERTa, KMM, additional direct/ICL judges, LoRA, Skywork | Recorded exploration with incomplete comparable prediction/configuration/freeze chains. They are not a verified formal ranking. |

## Remaining evidence gaps

A fresh responder unused in design is needed to assess a frozen selection rule prospectively. Current intervals condition on fitted models, fixed panels, and responders. Three training seeds do not supply three new responders. Medical-rubric extraction has not passed independent semantic acceptance. Raw-source reconstruction and full research replay remain incomplete; see [code guide](code.md).

Annotation cost motivates the work, but measured time savings and clinical benefits are not established.
