import numpy as np
from numpy import empty

from .callback import Saver, Printer
from .core import vem
from .initialization import factanal
from .preprocess import build_model
from .util import add_constant, lagmat

__all__ = ['fit', 'predict']


def fit(**kwargs):
    """
    vLGP API

    Parameters
    ----------
    y : ndarray
        obserbation
    lik : ndarray
        types of observation dimensions, 'spike' or 'lfp'
    z_dim : int
        number of latent dimensions
    x : ndarray, optional
        external factors
    a : ndarray, optional
        initial value of loading
    b : ndarray, optional
        initial value of regression
    mu : ndarray, optional
        initial value of posterior mean
    z : ndarray, optional
        true value of latent
    alpha : ndarray, optional
        true value of loading
    beta : ndarray, optional
        true value of regression
    history : int, optional
        history filter length
    rank : int, optional
        rank of incomplete Cholesky
    eps : double, optional
        a small positive number
    tol : double, optional
        numerical tolerance
    path : string, optional
        path to the save file
    callbacks : list, optional
        callbacks

    Returns
    -------
    dict
        fit
    """

    callbacks = kwargs.pop('callbacks', [])

    model = kwargs.pop('model', None)

    if model is None:
        model = build_model(**kwargs)

    if model['initialize'] == 'fa':
        initialize = factanal
    else:
        raise NotImplementedError(model['initialize'])

    if not model.get('initialized', False):
        initialize(model)

    printer = Printer()
    callbacks.extend([printer.print])

    saver = None
    path = model.get('path')
    if path is not None:
        saver = Saver()
        callbacks.extend([saver.save])

    try:
        vem(model, callbacks)
    finally:
        printer.print(model)
        if saver is not None:
            saver.save(model, force=True)

    return model


def predict(**kwargs):
    """
    Predict firing rate

    Parameters
    ----------
    model : dict
        fitted model
    z : ndarray
        latent
    a : ndarray
        loading
    b : ndarray
        regression
    v : ndarray
        posterior variance
    maxrate : float
        maximum predicted firing rate
    y : ndarray
        spike trains for history filter

    Returns
    -------
    ndarray
        predicted firing rate
    """

    model = kwargs.get('model')
    if model is not None:
        z = model['mu']
        y = model['y']
        a = model['a']
        b = model['b']
        v = model['v']
    else:
        z = kwargs.get('z')
        y = kwargs.get('y')
        a = kwargs.get('a')
        b = kwargs.get('b')
        v = kwargs.get('v')

    ntrial, nbin, z_dim = z.shape
    y_dim = a.shape[1]
    history = b.shape[0] - 1

    shape_out = (ntrial, nbin, y_dim)
    # regression (h dot b) part
    if y is None:
        y = np.zeros(shape_out)

    hb = empty(shape_out)
    for n in range(y_dim):
        for trial in range(ntrial):
            h = add_constant(lagmat(y[trial, :, n], lag=history))
            hb[trial, :, n] = h @ b[:, n]
    eta = z.reshape((-1, z_dim)) @ a + hb.reshape((-1, y_dim))
    if v is not None:
        r = np.exp(eta + 0.5 * v.reshape((-1, z_dim)) @ (a ** 2))
    else:
        r = np.exp(eta)
    maxrate = kwargs.get('maxrate', np.exp(20))
    np.clip(r, 0, maxrate, out=r)
    return np.reshape(r, shape_out)
