"""Small synthetic example for the public CCE-Bench interfaces."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from ccebench.methods import (
    DirectMethod,
    SimplexJudgeCombiner,
    fit_density_ratio,
    mips,
    offcem_style,
)


def expected_score(features: np.ndarray) -> np.ndarray:
    return np.clip(0.50 + 0.15 * features[:, 0] - 0.10 * features[:, 1], 0.0, 1.0)


def main() -> None:
    rng = np.random.default_rng(7)
    source_x = rng.normal(0.0, 1.0, size=(800, 4))
    target_x = rng.normal(0.35, 1.0, size=(400, 4))
    source_y = np.clip(expected_score(source_x) + rng.normal(0, 0.03, len(source_x)), 0, 1)

    dm = DirectMethod(LinearRegression()).fit(source_x, source_y)
    ratio = fit_density_ratio(source_x, target_x)
    weights = ratio.weights(source_x)
    offcem = offcem_style(
        source_x,
        source_y,
        target_x,
        outcome_model=LinearRegression(),
    )

    # Three noisy judges, calibrated only on a separate development sample.
    judge_dev = np.column_stack(
        [source_y + rng.normal(0, noise, len(source_y)) for noise in (0.03, 0.09, 0.18)]
    )
    combiner = SimplexJudgeCombiner().fit(judge_dev, source_y)

    print(f"Synthetic Target truth : {expected_score(target_x).mean():.4f}")
    print(f"Direct Method estimate: {dm.estimate(target_x):.4f}")
    print(f"SN-MIPS estimate      : {mips(source_y, weights):.4f}")
    print(f"OffCEM-style estimate : {offcem.value:.4f}")
    print(f"Density-ratio ESS/n   : {offcem.ess_fraction:.3f}")
    print(f"Judge weights         : {np.round(combiner.weights_, 3)}")


if __name__ == "__main__":
    main()

