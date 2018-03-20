def get_default_model():
    import numpy as np
    from vlgp import initialization
    from vlgp.callback import Saver
    from vlgp.preprocess import build_model

    np.random.seed(0)
    ydim = 10
    xdim = 2
    length = 50
    ntrial = 5

    loading = np.random.randn(xdim, ydim)
    bias = -np.random.rand(ydim)

    x = np.random.randn(ntrial, length, xdim)
    rate = np.exp(np.minimum(x @ loading + bias, 10))
    y = np.random.poisson(rate)

    config = {'y': y, 'lik': ['poisson'] * ydim, 'lat_dim': xdim, 'callbacks': []}
    model = build_model(**config)
    callbacks = model['callbacks']

    initialize = initialization.factanal

    if not model.get('initialized', False):
        initialize(model)
        model['initialized'] = True

    path = model.get('path')
    if path is not None:
        saver = Saver()
        callbacks.extend([saver.save])

    return model


def test_save():
    import pathlib
    from vlgp.util import save
    model = get_default_model()
    model['path'] = "test_save.npz"
    save(model)
    path = pathlib.Path(model['path'])
    path.unlink()
