# Result settings

A shared protocol tag alone is insufficient for comparison: panel, responder, rubric, label budget, evaluation subset, and metric must also match.

| Tag | Experiment and permitted comparison |
|---|---|
| `Formal-12` | Twelve fixed Target panels, zero Target fitting labels. Compare complete configurations on common retained answers; retain auxiliary-control and post-gold-correction roles. |
| `Historical-LOAO` | Historical CounselBench (three held-out agents) and Rwanda (four), or explicitly narrower folds. Historical Rwanda uses `overall11`, unlike the formal medical-consensus dimension. |
| `Diagnostic-4P` | Remove the original formal Target, then hold out a Source responder on CounselBench, PsycSumEval, Rwanda, and Warmth–Substance. Includes C*/F7 and medical-rubric features. |
| `Diagnostic-4P-Medical-Rubric` | Explicit subtag for the four-panel J0_RAW/J0_CAL/R0/R2 study; the same Diagnostic-4P responder definition, with distinct extraction and acceptance status. |
| `Source-held-out-Diagnostics` | Metadata-only umbrella used by the diagnostic panel contract; `main=True` selects the 11-panel factor set. This is not a result leaderboard. |
| `Diagnostic-11P` | Source-held-out objective/evidence diagnostic, excluding Well-being; 4,872 Target answers per seed. `C_SOURCE_ROW` is distinct from Stage-1 `C*`. |
| `Judge-Cal` | Separate grouped judge calibration and development labels; compare RMSE only within that setting. |
| `Historical-LOAO-TargetLabels` | Historical zero-label constraint relaxed at a specified budget. Compare updated/unchanged DM on the same remaining set; compare Active/Random using operational MAE. |

All are retrospective. Corrected SN-MIPS and SNDR remain post-gold retrospective. Diagnostic acceptance, code tests, and freeze checks do not create prospective evidence or verify clinical meaning.

The [result index](../results/README.md) identifies the protocol and evidence status for each table.
