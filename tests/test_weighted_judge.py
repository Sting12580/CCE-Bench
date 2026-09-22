import numpy as np

from ccebench.methods.weighted_judge import SimplexJudgeCombiner


def test_simplex_combiner_is_interpretable_and_prefers_accurate_judge():
    rng = np.random.default_rng(4)
    gold = rng.uniform(0, 1, size=200)
    judges = np.column_stack(
        [gold + rng.normal(0, 0.01, len(gold)), rng.uniform(0, 1, len(gold))]
    )
    model = SimplexJudgeCombiner().fit(judges, gold)
    assert np.all(model.weights_ >= 0)
    assert np.isclose(model.weights_.sum(), 1.0)
    assert model.weights_[0] > model.weights_[1]

