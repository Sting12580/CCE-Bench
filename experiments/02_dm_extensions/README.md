# 2. DM extensions and parallel approaches

[Technical report](../../paper/technical_report.pdf): Section 6.2 and Tables 5–6. Methods are described in Sections 5.2–5.5.

## Question and design

DM supplies an initial prediction, but its Target mean remains imperfect. Can residual corrections, same-question support, or extra semantic information reduce that error? SNDR and Context extend the DM prediction in their respective settings. DANN explores adversarial training in the BERT branch; SN-MIPS independently estimates a weighted Source mean. These are different design hypotheses, not stages of one mandatory upgrade sequence.

We first compare complete methods, then use matched controls or ablations to examine the added component. The comparisons below stay within their own settings.

## Results

| Formal-12 comparison | Macro-nSAE | What it supports |
|---|---|---|
| DANN vs matched BERT `lambda=0` | 0.052826 vs 0.052713 | No established incremental gain from adversarial training; the auxiliary control remains visible. |
| SN-MIPS vs Source Mean | 0.075532 vs 0.099236 | Weighting improves on the constant prior, while DM is lower at 0.069394. |
| Project SNDR vs DM | 0.066565 vs 0.069394 | A lower point estimate; the paired difference interval crosses zero. |

Project SNDR adds a weighted out-of-fold Source residual correction to frozen DM Target predictions, then clips individual predictions before averaging. It is an OffCEM-inspired project analogue, not an official OffCEM reproduction. Corrected SN-MIPS and SNDR results are explicitly **post-gold retrospective**. SN-MIPS produces a system value, so it has no row-level nMAE or nRMSE. Rwanda's effective sample size is about 5.78% of Source records; this indicates concentrated weights, not a demonstrated cause of error.

Historical LOAO Context uses same-question Source answers and scores to predict a residual. Its key controls use native row MAE:

| Historical comparison | CounselBench | Rwanda `overall11` |
|---|---:|---:|
| TextDM | 0.547252 | 0.395944 |
| Context selected | 0.483443 | 0.303285 |
| Remove context-score features | 0.479736 | 0.392949 |
| Shuffle support scores | 0.484827 | 0.396953 |
| Remove similarity features | 0.520336 | 0.309852 |
| Same-question Agent median | 0.977333 | 0.288358 |

Correct score–answer pairing helps the Rwanda Context pipeline, yet the simple Agent median is better. This median uses the visible peer Agent scores for each question, excluding Human and Target responders; it is not a median of responder-wide means. CounselBench does not establish a score-channel gain, although removing similarity worsens its error. Across the 21 historical Context selections, every chosen gate was constant, so these results do not support response-dependent gating.

In Diagnostic-4P, adding fixed LLM rubric features to C* gives `R2` Macro-nSAE **0.063003**, versus **0.032194** for `R0` / C*. Eleven of twelve panel×seed cells worsen. The current semantic feature pipeline has not shown an incremental benefit, and its independent semantic acceptance remains incomplete.

## Interpretation

DANN's matched control tests the adversarial term; a comparison to the differently trained BGE-M3 DM cannot do so. Historical Context ablations have their own Source-only selection, so they compare pipelines rather than identical fitted parameters with one input erased. Removing context-score features also leaves the score-trained DM and supervised residual labels in place.

Historical Rwanda uses an eleven-rubric overall score. Formal-12 and the medical diagnostic use medical-consensus alignment. Their numeric results cannot be merged. Neither failure of the current semantic extension nor unconfirmed component gains establishes that all semantic information, residual correction, or domain adaptation is ineffective.

## Result files

- [Formal-12 results](../../results/formal_12_panel.csv) and [matched contrasts](../../results/formal_matched_contrasts.csv).
- [Historical Context controls](../../results/context_historical.csv) and [paired contrasts](../../results/context_historical_contrasts.csv).
- [Semantic-stage aggregate](../../results/medical_rubric_macro.csv) and [panel results](../../results/medical_rubric_panel_metrics.csv), and [semantic contrasts](../../results/medical_rubric_contrasts.csv).

## Related methods and code

See [weighting and residual correction](../../methods/02_ope/README.md), [DANN](../../methods/04_domain_adaptation/README.md), [Context](../../methods/05_context_bridge/README.md), and [LLM features](../../methods/03_llm_judges/README.md) for method details and reference code.
