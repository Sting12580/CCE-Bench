import numpy as np
from sklearn.linear_model import LinearRegression

from ccebench.methods.dm import DirectMethod
from ccebench.methods.representations import RewardInformedProjection, pair_features


def test_direct_method_fits_linear_outcome():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 3))
    y = 0.4 + x @ np.array([0.1, -0.2, 0.05])
    dm = DirectMethod(LinearRegression(), clip=None).fit(x, y)
    np.testing.assert_allclose(dm.predict(x), y, atol=1e-10)


def test_pair_features_and_learned_projection_shapes():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(40, 5))
    a = rng.normal(size=(40, 5))
    pair = pair_features(q, a)
    assert pair.shape == (40, 15)
    y = np.clip(0.5 + 0.1 * pair[:, 0], 0, 1)
    projection = RewardInformedProjection(latent_dim=4, max_iter=500).fit(pair, y)
    assert projection.transform(pair).shape == (40, 4)
