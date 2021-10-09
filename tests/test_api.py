import pytest


@pytest.fixture()
def data():
    import numpy as np

    ydim = 5
    zdim = 2

    length = 100
    ntrial = 5

    a = np.random.randn(zdim, ydim)
    b = -2

    trials = []
    for i in range(ntrial):
        z = np.column_stack(
            (
                np.sin(np.linspace(0, 8 * np.pi, length)),
                np.cos(np.linspace(0, 8 * np.pi, length)),
            )
        )
        y = np.random.poisson(np.exp(z @ a + b))
        trials.append({"y": y, "id": i})

    return trials


def test_fit(data):
    from vlgp.api import fit, transform
    result = fit(data, n_factors=2)
    trials = result['trials']
    params = result['params']
    config = result['config']

    transform(trials, params, config)
