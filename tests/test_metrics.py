import numpy as np

from ccebench.metrics import nmae, nrmse, nsae


def test_metrics_separate_mean_and_row_error():
    truth = np.array([0.0, 1.0])
    prediction = np.array([0.25, 0.75])
    assert nsae(truth, prediction) == 0.0
    assert nmae(truth, prediction) == 0.25
    assert nrmse(truth, prediction) == 0.25

