import numpy as np
from sklearn.linear_model import LinearRegression

from ccebench.methods.context import ContextResidualBridge
from ccebench.methods.context_features import (
    CONTEXT_FEATURE_SETS,
    F7_FEATURES,
    F7_REDUCED_FEATURE_SETS,
    FULL_CONTEXT_FEATURES,
    select_features,
)


def test_feature_family_dimensions_and_selection():
    assert len(FULL_CONTEXT_FEATURES) == 37
    assert len(F7_FEATURES) == 19
    assert {len(F7_REDUCED_FEATURE_SETS[name]) for name in ("LITE_1", "LITE_3", "LITE_5", "LITE_9")} == {1, 3, 5, 9}
    assert tuple(CONTEXT_FEATURE_SETS) == tuple(f"F{i}" for i in range(9))
    matrix = np.arange(74, dtype=float).reshape(2, 37)
    selected = select_features(matrix, FULL_CONTEXT_FEATURES, F7_FEATURES)
    assert selected.shape == (2, 19)


def test_context_bridge_learns_residual_not_base_model():
    rng = np.random.default_rng(5)
    context = rng.normal(size=(100, 2))
    base = np.full(100, 0.4)
    truth = base + context @ np.array([0.10, -0.05])
    bridge = ContextResidualBridge(LinearRegression(), clip=None).fit(context, truth, base)
    np.testing.assert_allclose(bridge.predict(context, base), truth, atol=1e-10)

