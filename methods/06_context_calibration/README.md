# Context feature selection

[Technical report](../../paper/technical_report.pdf): Sections 5.4 and 6.3, Table 7, and Appendix B.2.

## The C* selection procedure

The four-panel diagnostic removes the formal Target, then holds out a Source responder. C* selects among F0–F8 feature families and six Ridge/GBR heads using Source validation. The families add or remove blocks of the 37 Context features, including score summaries, similarity, length, availability, and human/Agent role information.

F7 is one fixed 19-feature family with a head selected within that family. Its later use for structural diagnostics does not make it the overall C* selection rule. The twelve C* panel-by-seed choices are F4 in four cells, F6 in seven, and F7 in one.

## Findings

C* reaches Macro-nSAE **0.032194**, versus **0.044758** for DM. The conditional 95% difference interval **[−0.030581, +0.012425]** crosses zero. C* improves CounselBench and PsycSumEval but worsens Rwanda and Warmth–Substance.

Rwanda illustrates the distinction between selection and mean accuracy: C* lowers nMAE from **0.100357 to 0.073765**, while increasing nSAE from **0.003134 to 0.028520**. Source-only selection respects label isolation but does not guarantee better Target performance. The result does not separately identify whether the candidates, criterion, or responder differences caused the transfer failure.

Results: [aggregate](../../results/context_stage1_summary.csv), [panel metrics](../../results/context_stage1_panel_metrics.csv), [selected configurations](../../results/context_stage1_source_selection.csv), [contrasts](../../results/context_stage1_contrasts.csv), and [error diagnostics](../../experiments/03_error_diagnostics/README.md).

## Reference code

[CONTEXT_FEATURE_SETS](../../src/ccebench/methods/context_features.py) defines F0–F8 and supports feature selection from a supplied matrix. The complete C* validation search is specified in the report and is not implemented by this feature-definition module.
