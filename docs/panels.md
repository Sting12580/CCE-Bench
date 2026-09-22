# Formal evaluation panels

The formal benchmark contains **11 dataset families and 12 evaluation panels**. OpenMEVA contributes two distinct panels, so “12 datasets” would be inaccurate.

In each panel, a fixed Target responder's text is visible, but its gold score is unavailable until predictions are frozen. Source and Target answer the same item set; this is hidden-responder score transfer, not unseen-question generalization.

| Panel | Input → response | Primary score | Native scale | Source / Target rows per seed |
|---|---|---|---:|---:|
| `counselbench` | mental-health question → counseling answer | overall counseling quality | 1–5 | 300 / 100 |
| `gbb_jme` | Greek legal case/question → legal answer | official mean of facts, articles, analysis | 1–10 | 240 / 60 |
| `hanna` | creative-writing prompt → story | coherence | 1–5 | 960 / 96 |
| `openmeva_roc` | ROCStories context → story continuation | overall story quality | 1–5 | 800 / 200 |
| `openmeva_wp` | WritingPrompts topic → story | overall story quality | 1–5 | 800 / 200 |
| `psycsumeval` | Cochrane abstract → clinical summary | main findings coverage | 0–2 | 333 / 111 |
| `rwanda` | Rwanda health/care scenario → advice | alignment with medical consensus | 1–5 | 2,024 / 506 |
| `summeval` | news article → summary | expert consistency | 1–5 | 1,500 / 100 |
| `warmth_substance` | multilingual health question → answer | clinical accuracy | 1–5 | 378 / 126 |
| `wellbeing_advice_study1` | advice-seeking post → advice comment | likely effectiveness | 1–7 | 150 / 50 |
| `wmt20_ende` | English segment → German translation | professional pSQM | 0–6 | 12,483 / 1,387 |
| `wmt20_zhen` | Chinese segment → English translation | professional pSQM | 0–6 | 17,874 / 1,986 |

The fixed primary score prevents choosing a favorable dimension after inspecting Target results. Secondary dimensions are diagnostic only and are not averaged into new composites.


## Fixed Target responders

These identities define the formal evaluation. Diagnostic responders differ and are listed separately in [diagnostic_panel_contract.csv](../results/diagnostic_panel_contract.csv).

| Panel | Formal Target | Task ID |
|---|---|---|
| `counselbench` | `llama3` | `overall` |
| `gbb_jme` | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | `avg` |
| `hanna` | `GPT-2 (tag)` | `coherence` |
| `openmeva_roc` | `gpt` | `overall` |
| `openmeva_wp` | `gpt` | `overall` |
| `psycsumeval` | `Claude Opus` | `main_findings` |
| `rwanda` | `deepseek-r1` | `alignment_with_medical_consensus` |
| `summeval` | `M14` | `consistency` |
| `warmth_substance` | `deepseek` | `clinical_accuracy` |
| `wellbeing_advice_study1` | `gpt-4o` | `effectiveness` |
| `wmt20_ende` | `Huoshan_Translate.832` | `psqm` |
| `wmt20_zhen` | `Tencent_Translation.1249` | `psqm` |

Exact aggregation, exclusions, scales, and cluster units are in [formal_panel_contract.csv](../results/formal_panel_contract.csv). Source and Target counts total 37,842 and 4,922 per seed. Dataset references are listed in Appendix A of the [report](../paper/technical_report.pdf).
