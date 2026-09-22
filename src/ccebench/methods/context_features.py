"""Named feature families used in the Context/F7 diagnostic line."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_array


FULL_CONTEXT_FEATURES = (
    "m0_target",
    "human_score",
    "m0_human_gap",
    "aux_score_mean",
    "aux_score_median",
    "aux_score_std",
    "aux_score_min",
    "aux_score_max",
    "aux_score_range",
    "all_support_score_mean",
    "all_support_score_median",
    "all_support_score_std",
    "all_support_score_range",
    "human_agent_mean_gap",
    "human_agent_mean_gap_abs",
    "m0_all_support_mean_gap",
    "target_human_cosine",
    "target_agent_cosine_mean",
    "target_agent_cosine_max",
    "target_agent_cosine_min",
    "closest_agent_score",
    "similarity_weighted_support_score_t005",
    "similarity_weighted_support_score_t010",
    "similarity_weighted_support_score_t020",
    "m0_similarity_weighted_gap_t005",
    "m0_similarity_weighted_gap_t010",
    "m0_similarity_weighted_gap_t020",
    "top1_top2_similarity_margin",
    "target_support_length_diff",
    "target_support_length_abs_diff",
    "target_length_outlier",
    "support_count",
    "agent_support_count",
    "missing_human",
    "missing_agent",
    "missing_top2_similarity",
    "missing_context_reward",
)

MISSINGNESS_FEATURES = {
    "support_count",
    "agent_support_count",
    "missing_human",
    "missing_agent",
    "missing_top2_similarity",
    "missing_context_reward",
}
SUPPORT_SCORE_FEATURES = {
    "all_support_score_mean",
    "all_support_score_median",
    "all_support_score_std",
    "all_support_score_range",
    "m0_all_support_mean_gap",
}
ROLE_SCORE_FEATURES = {
    "human_score",
    "m0_human_gap",
    "aux_score_mean",
    "aux_score_median",
    "aux_score_std",
    "aux_score_min",
    "aux_score_max",
    "aux_score_range",
    "human_agent_mean_gap",
    "human_agent_mean_gap_abs",
}
SIMILARITY_FEATURES = {
    "target_human_cosine",
    "target_agent_cosine_mean",
    "target_agent_cosine_max",
    "target_agent_cosine_min",
    "closest_agent_score",
    "similarity_weighted_support_score_t005",
    "similarity_weighted_support_score_t010",
    "similarity_weighted_support_score_t020",
    "m0_similarity_weighted_gap_t005",
    "m0_similarity_weighted_gap_t010",
    "m0_similarity_weighted_gap_t020",
    "top1_top2_similarity_margin",
}
LENGTH_FEATURES = {
    "target_support_length_diff",
    "target_support_length_abs_diff",
    "target_length_outlier",
}
EXPLICIT_ROLE_FEATURES = ROLE_SCORE_FEATURES | {
    "target_human_cosine",
    "target_agent_cosine_mean",
    "target_agent_cosine_max",
    "target_agent_cosine_min",
    "closest_agent_score",
    "agent_support_count",
    "missing_human",
    "missing_agent",
}
TARGET_RELATED_FEATURES = SIMILARITY_FEATURES | LENGTH_FEATURES | {
    "m0_target",
    "m0_human_gap",
    "m0_all_support_mean_gap",
}
CONTEXT_REWARD_FEATURES = {
    "human_score",
    "m0_human_gap",
    "aux_score_mean",
    "aux_score_median",
    "aux_score_std",
    "aux_score_min",
    "aux_score_max",
    "aux_score_range",
    "all_support_score_mean",
    "all_support_score_median",
    "all_support_score_std",
    "all_support_score_range",
    "human_agent_mean_gap",
    "human_agent_mean_gap_abs",
    "m0_all_support_mean_gap",
    "closest_agent_score",
    "similarity_weighted_support_score_t005",
    "similarity_weighted_support_score_t010",
    "similarity_weighted_support_score_t020",
    "m0_similarity_weighted_gap_t005",
    "m0_similarity_weighted_gap_t010",
    "m0_similarity_weighted_gap_t020",
}


def _ordered(names) -> tuple[str, ...]:
    wanted = set(names)
    return tuple(name for name in FULL_CONTEXT_FEATURES if name in wanted)


CONTEXT_FEATURE_SETS = {
    "F0": _ordered({"m0_target"} | MISSINGNESS_FEATURES),
    "F1": _ordered({"m0_target"} | MISSINGNESS_FEATURES | SUPPORT_SCORE_FEATURES),
    "F2": _ordered(
        {"m0_target"} | MISSINGNESS_FEATURES | SUPPORT_SCORE_FEATURES | ROLE_SCORE_FEATURES
    ),
    "F3": _ordered(set(FULL_CONTEXT_FEATURES) - LENGTH_FEATURES),
    "F4": FULL_CONTEXT_FEATURES,
    "F5": _ordered(set(FULL_CONTEXT_FEATURES) - SIMILARITY_FEATURES),
    "F6": _ordered(set(FULL_CONTEXT_FEATURES) - CONTEXT_REWARD_FEATURES),
    "F7": _ordered(set(FULL_CONTEXT_FEATURES) - EXPLICIT_ROLE_FEATURES),
    "F8": _ordered(set(FULL_CONTEXT_FEATURES) - TARGET_RELATED_FEATURES),
}

F7_FEATURES = CONTEXT_FEATURE_SETS["F7"]
F7_REDUCED_FEATURE_SETS = {
    "LITE_1": ("m0_target",),
    "LITE_3": _ordered(
        {"m0_target", "all_support_score_mean", "similarity_weighted_support_score_t010"}
    ),
    "LITE_5": _ordered(
        {
            "m0_target",
            "all_support_score_mean",
            "similarity_weighted_support_score_t010",
            "all_support_score_std",
            "top1_top2_similarity_margin",
        }
    ),
    "LITE_9": _ordered(
        {
            "m0_target",
            "all_support_score_mean",
            "similarity_weighted_support_score_t010",
            "all_support_score_std",
            "top1_top2_similarity_margin",
            "all_support_score_median",
            "all_support_score_range",
            "target_support_length_diff",
            "target_support_length_abs_diff",
        }
    ),
    "F7_19": F7_FEATURES,
}


def select_features(
    matrix: ArrayLike,
    input_names: tuple[str, ...] | list[str],
    selected_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Select named columns while preserving ``selected_names`` order."""

    try:
        x = check_array(matrix, dtype=float, ensure_all_finite="allow-nan")
    except TypeError:  # scikit-learn < 1.6
        x = check_array(matrix, dtype=float, force_all_finite="allow-nan")
    if x.shape[1] != len(input_names):
        raise ValueError("matrix width does not match input_names")
    positions = {name: index for index, name in enumerate(input_names)}
    missing = [name for name in selected_names if name not in positions]
    if missing:
        raise ValueError(f"selected features are missing: {missing}")
    return x[:, [positions[name] for name in selected_names]]


assert len(FULL_CONTEXT_FEATURES) == 37
assert len(F7_FEATURES) == 19
