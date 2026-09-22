# Methods

The methods address different parts of the same task: predicting an unscored responder's mean human score from scored Source answers and visible Target text. They include direct prediction, weighting, residual correction, domain adaptation, and LLM-based scoring.

| Method | Main role | Technical report |
|---|---|---|
| [Direct Method](01_direct_method/README.md) | Supervised answer-level prediction, followed by averaging | Section 5.1; Table 3 |
| [Weighting and residual correction](02_ope/README.md) | SN-MIPS estimates a weighted Source mean; SNDR corrects DM residuals | Section 5.2; Tables 3 and 5 |
| [LLM scoring and judge combinations](03_llm_judges/README.md) | Direct rubric scores, Context plus an LLM score, rubric features, and separate judge calibration | Section 5.5; Table 4; Appendix D |
| [BERT and domain adaptation](04_domain_adaptation/README.md) | Matched supervised and adversarial regression | Sections 5.1 and 5.3; Tables 3 and 5 |
| [Context residual prediction](05_context_bridge/README.md) | Same-question Source scores and answer relationships | Section 5.4; Table 6 |
| [Context feature selection](06_context_calibration/README.md) | C* selects a feature family and regression head using Source labels | Sections 5.4 and 6.3; Table 7 |
| [F7 and follow-ups](07_f7_followups/README.md) | Fixed-feature diagnostics, mean projection, and alternative residual heads | Section 6.3; Table 8; Appendix D |

Read the [technical report](../paper/technical_report.pdf) for the complete configurations and [experiments](../experiments/README.md) for their comparisons. Directory numbers are navigation identifiers, not a sequence of replacements.

The [code guide](../docs/code.md) describes the reference implementations and their scope. Formal training details, selection procedures, and label-access rules are specified in Sections 4–5 and Appendix B of the report.

The [method overview](../docs/methods.md) shows the relationships among these approaches.
