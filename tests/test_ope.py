import numpy as np
from sklearn.linear_model import LinearRegression

from ccebench.methods.ope import effective_sample_size, fit_density_ratio, mips, offcem_style


def test_density_ratio_is_one_for_duplicated_pools():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(120, 4))
    ratio = fit_density_ratio(x, x)
    weights = ratio.weights(x)
    np.testing.assert_allclose(weights, np.ones(len(x)), atol=1e-8)
    assert np.isclose(effective_sample_size(weights), 1.0)


def test_mips_and_offcem_style_on_identical_linear_population():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 3))
    y = 0.5 + x @ np.array([0.1, -0.05, 0.02])
    weights = np.ones(len(x))
    assert np.isclose(mips(y, weights), y.mean())
    result = offcem_style(x, y, x, outcome_model=LinearRegression())
    assert np.isclose(result.value, y.mean(), atol=1e-10)
    assert np.isclose(result.residual_correction, 0.0, atol=1e-10)

