import numpy as np


def test_clip_grad():
    from vlgp import fast
    np.random.seed(0)
    n = 100
    x = np.random.randn(n)
    fast.clip_grad(x, bound=1.0)

    assert np.all(np.logical_and(x >= -1.0, x <= 1.0))


def test_cut_trial():
    from vlgp import fast
    y = np.random.randn(100, 10)
    x = np.random.randn(100, 5)
    trial = {'y': y, 'x': x}
    fast_trials = fast.cut_trial(trial, 10)
    for each in fast_trials:
        assert each['y'].shape == (10, 10)
