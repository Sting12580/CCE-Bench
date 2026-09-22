import numpy as np
import pytest

pytest.importorskip("torch")

from ccebench.methods.domain_adaptation import fit_dann  # noqa: E402


def test_dann_smoke_uses_source_scores_and_unlabeled_target_features():
    rng = np.random.default_rng(6)
    source = rng.normal(size=(20, 3))
    target = rng.normal(0.2, size=(15, 3))
    scores = 0.5 + 0.1 * source[:, 0]
    fitted = fit_dann(source, scores, target, epochs=2, hidden_dim=4)
    assert fitted.model.predict(target).shape == (15,)
    assert len(fitted.reward_loss) == 2

