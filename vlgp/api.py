import numpy as np
from numpy import asarray, newaxis, empty

from .preprocess import build_model
from .callback import Saver, Printer
from .core import vem
from vlgp.initialization import factanal
from .util import add_constant, lagmat


__all__ = ['fit', 'predict']


def fit(**kwargs):
    """
    vLGP API

    Parameters
    ----------
    y : ndarray
        obserbation
    y_type : ndarray
        types of observation dimensions, 'spike' or 'lfp'
    dyn_ndim : int
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

    model = build_model(**kwargs)

    if model['initialize'] == 'fa':
        initialize = factanal
    else:
        raise NotImplementedError(model['initialize'])

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


def predict(z, a, b, v=None, maxrate=None, y=None):
    """
    Predict firing rate

    Parameters
    ----------
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
    ntrial, nbin, z_ndim = z.shape
    y_ndim = a.shape[1]
    history = b.shape[0] - 1

    shape_out = (ntrial, nbin, y_ndim)
    # regression (h dot b) part
    if y is None:
        y = np.zeros(shape_out)

    hb = empty(shape_out)
    for y_dim in range(y_ndim):
        for trial in range(ntrial):
            h = add_constant(lagmat(y[trial, :, y_dim], lag=history))
            hb[trial, :, y_dim] = h @ b[:, y_dim]
    eta = z.reshape((-1, z_ndim)) @ a + hb.reshape((-1, y_ndim))
    if v is not None:
        r = np.exp(eta + 0.5 * v.reshape((-1, z_ndim)) @ (a ** 2))
    else:
        r = np.exp(eta)
    np.clip(r, 0, maxrate, out=r)
    return np.reshape(r, shape_out)
