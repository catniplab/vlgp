import pytest


@pytest.fixture()
def fitted():
    import numpy as np
    from vlgp import preprocess

    ydim = 5
    zdim = 2

    length = 100
    ntrial = 5

    a = np.random.randn(zdim, ydim)
    b = np.array([[-2.]])

    trials = []
    for i in range(ntrial):
        z = np.column_stack(
            (
                np.sin(np.linspace(0, 8 * np.pi, length)),
                np.cos(np.linspace(0, 8 * np.pi, length)),
            )
        )
        x = np.ones((length, 1), dtype=np.float)
        y = np.random.poisson(np.exp(z @ a + x @ b))
        trials.append({"y": y, "id": i, "x": x, "mu": z, "v": np.ones_like(z)})

    return {"trials": trials, "params": {"a": a, "b": b}, "config": preprocess.get_config()}


def test_loglike(fitted):
    from vlgp.evaluation import loglik
    loglik(fitted)
