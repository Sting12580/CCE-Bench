# Formal experiment source

This directory contains the original estimator and runner code for the formal configurations discussed in the technical report. The Python files retain their research implementations. They complement the smaller illustrative components in `src/ccebench/` at the repository root.

The source is provided for inspecting the implemented methods. Dataset construction, fitted models, embeddings, frozen registries, predictions, and evaluator inputs are not distributed here, so this directory is not a complete data-to-results reproduction package. Context, C*, the semantic-feature studies, and the Target-label experiments are described in the report and result tables; their research pipelines are outside this source subset.

## Methods and entry points

| Report configuration | Implementation | Inputs and behavior |
|---|---|---|
| M1: Source Mean | [`source_mean_predictions`](src/cce_data/estimators/cce_primary_zero_label.py) | Normalized Source scores and Target row count; returns the constant Source mean. |
| M3: Direct Method | `run_m3` in the [formal runner](scripts/run_cce_benchmark_primary_zero_label_methods_v1.py), using [`learned_embedding.py`](src/cce_data/estimators/learned_embedding.py) and `build_features` in [`split_score_estimator.py`](src/cce_data/estimators/split_score_estimator.py) | Separate question/answer BGE-M3 embeddings and Source scores; learns the Source-informed representation, constructs 576-dimensional features, and fits the specified gradient-boosted regressor. |
| M6: BERT DANN and matched BERT control | `fit_fixed_fold_bert_dann_pair` in [`cce_primary_zero_label.py`](src/cce_data/estimators/cce_primary_zero_label.py), with helpers in [`dann_finetuned_encoder.py`](src/cce_data/estimators/dann_finetuned_encoder.py) | Source question/answer text and scores, score-free Target text, fixed Source training/validation indices, and a local BERT snapshot; fits matched coefficient-0.1 and coefficient-0 arms. |
| M13: corrected SN-MIPS | `crossfit_reward_informed_sn_mips_v2` in [`cce_primary_zero_label_m13_m14_v2.py`](src/cce_data/estimators/cce_primary_zero_label_m13_m14_v2.py) | 1024-dimensional BGE-M3 question/answer embeddings, Source scores, and fixed item-fold IDs; learns a Source-reward-informed 576-dimensional representation, estimates cross-fitted calibrated density weights, and returns a system value. |
| M14: corrected project SNDR | `sndr_predictions` in [`cce_primary_zero_label.py`](src/cce_data/estimators/cce_primary_zero_label.py), assembled by the [corrected runner](scripts/run_cce_benchmark_primary_m13_m14_corrected_v2.py) | Frozen same-seed DM Target predictions, Source scores, out-of-fold Source predictions, and corrected M13 weights; adds the weighted residual and clips each row before averaging. |

The corrected M13/M14 configurations retain their post-gold retrospective status. The older M13/M14 routines remain in the shared formal module because the original runner uses them; they are superseded for the report's corrected results. Source scores train the corrected M13 representation. Only its underlying BGE text-embedding cache is score-independent. M14 is a project adaptation inspired by OffCEM, not an official reproduction.

## Runners

- [`run_cce_benchmark_primary_zero_label_methods_v1.py`](scripts/run_cce_benchmark_primary_zero_label_methods_v1.py) assembles the 12-panel formal experiment, including M1, M3, paired M6, and the earlier M13/M14 arms.
- [`run_cce_benchmark_primary_m13_m14_corrected_v2.py`](scripts/run_cce_benchmark_primary_m13_m14_corrected_v2.py) computes corrected M13/M14 using the frozen upstream run. It requires the exact upstream DM predictions, Source out-of-fold outcomes, base embeddings, registry bindings, and freeze records.
- [`run_cce_benchmark_primary_dm_bge_m3_learned_gbr.py`](scripts/run_cce_benchmark_primary_dm_bge_m3_learned_gbr.py) supplies BGE encoding, configuration, and I/O helpers imported by both runners. Its standalone command describes the earlier 11-task DM experiment; it is not the 12-panel entry point.

The runner paths resolve relative to this `research/` directory. Their defaults refer to the original `configs/`, `data/processed/`, and `outputs/` layouts, which are not included. Several paths and hashes are fixed by the original experimental contracts; supplying an arbitrary dataset directory does not reproduce the reported experiment. The corrected runner also retains its original `--device mps` provenance requirement, while its nuisance fits execute on CPU.

Target gold is not an input to these fitting runners. Evaluation occurs separately after predictions have been frozen. The evaluator and data-preparation pipeline are not part of this subset.

## Environment and inspection

Use Python 3.10 or newer with `numpy`, `scikit-learn`, and `joblib`. Representation fitting requires `torch`; text encoding and the BERT branch also require `transformers` and compatible local pretrained model snapshots. `psutil` is optional for process-memory diagnostics. These are additional research dependencies; installing the root reference package does not install all of them. Exact historical dependency pins are not supplied here.

The following commands, run from the repository root, only display argument help:

```bash
python research/scripts/run_cce_benchmark_primary_zero_label_methods_v1.py --help
python research/scripts/run_cce_benchmark_primary_m13_m14_corrected_v2.py --help
python research/scripts/run_cce_benchmark_primary_dm_bge_m3_learned_gbr.py --help
```

To inspect modules directly, put `research/src` on `PYTHONPATH`:

```bash
PYTHONPATH=research/src python -c "from cce_data.estimators.cce_primary_zero_label import source_mean_predictions, sndr_predictions; print(source_mean_predictions.__doc__); print(sndr_predictions.__doc__)"
```

Help and import checks do not fit models or verify reproduction of the reported scores. Running a fitting entry point requires the experiment-specific inputs listed above.
