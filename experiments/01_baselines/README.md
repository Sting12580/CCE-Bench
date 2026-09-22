# 1. Baselines and direct scoring

[Technical report](../../paper/technical_report.pdf): Section 6.1, Tables 3–4 and Figure 1. Methods are described in Sections 5.1 and 5.5.

## Question and design

Can historical Source scores estimate the mean score of a Target responder on the same questions? Source answers and scores are available; Target answers are visible, while Target scores are withheld from fitting and selection.

The Formal-12 comparison starts with Source Mean, the equally weighted mean of all retained Source response-score records. It is a global constant, not a same-question average. DM predicts each Target answer with BGE-M3 features, a Source-score-trained representation projection, and a GradientBoostingRegressor, then averages its predictions. BERT `lambda=0` supplies a supervised regression reference from the matched DANN training framework and retains its status as an auxiliary control.

## Results

| Formal-12 configuration | Macro-nSAE |
|---|---:|
| Source Mean | 0.099236 |
| DM | 0.069394 |
| BERT `lambda=0` auxiliary control | 0.052713 |

DM improves on Source Mean in 10 of 12 panels, reducing Macro-nSAE by about 30.1%. The exceptions are PsycSumEval (Source Mean **0.087087**, DM **0.132005**) and Rwanda (**0.027050**, **0.078953**). Both comparators use historical scores, so this is an incremental text-modeling comparison, not an experiment with versus without historical labels.

A separate Diagnostic-4P study compares three scoring paths on CounselBench, PsycSumEval, Rwanda, and Warmth–Substance. These diagnostic responders are held out from Source after the original formal Target is removed.

| Diagnostic path | What it uses | Macro-nSAE |
|---|---|---:|
| `J0_RAW` | One `gpt-4.1-mini` rubric judge, with the question, candidate answer, and anonymous same-question Source answer texts; no Source numerical scores in its prompt | 0.194542 |
| `J0_CAL` | C* Context features plus the LLM score and a missing-output indicator; a Source-supervised residual predictor added to DM | 0.059694 |
| `R0` / C* | The selected Context predictor without those LLM-score features | 0.032194 |

The joint supervised predictor improves on direct LLM scoring at these point estimates, but does not beat Context alone. `J0_CAL` fits a residual from Context features and the LLM score; it is neither one-dimensional score calibration nor LLM fine-tuning. Invalid judge outputs fall back to the DM prediction and receive a missing indicator.

## Interpretation

The Formal-12 DM and BERT configurations differ in representation, regression, training, and validation. DM used historical 20% Source-row validation for its representation; BERT used one fixed item fold per seed for validation. Their difference cannot isolate a backbone effect or establish a final method for new responders. The matched adversarial comparison appears on [the next page](../02_dm_extensions/README.md).

The four-panel judge comparison changes both inputs and prediction procedure, so its improvement cannot be attributed only to calibration. Its fixed responders differ from Formal-12, the semantic stage has not passed independent acceptance, and these point estimates are not presented as significant pairwise differences. This experiment also does not test an LLM given same-question numerical scores as in-context examples. Other direct and ICL judges were explored, but their incomplete evidence chains prevent a common formal ranking.

## Result files

- [Formal-12 aggregate](../../results/formal_12_panel.csv) and [panel results](../../results/formal_panel_metrics.csv).
- [Four-panel judge and semantic-stage aggregates](../../results/medical_rubric_macro.csv) and [panel results](../../results/medical_rubric_panel_metrics.csv).
- [Separate multi-judge calibration](../../results/judge_calibration.csv), which uses a different protocol and RMSE.

## Related methods and code

See [DM](../../methods/01_direct_method/README.md), [BERT regression](../../methods/04_domain_adaptation/README.md), and [LLM scoring](../../methods/03_llm_judges/README.md) for method details and reference code.
