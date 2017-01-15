import numpy as np
from numpy import asarray, newaxis, empty

from .callback import Saver, Printer
from .core import initialize, vem
from .math import sexp
from .util import add_constant, lagmat


def fit(y,
        dyn_ndim,
        x=None,
        obs_types=None,
        a=None,
        b=None,
        mu=None,
        history_filter=0,
        z=None,
        alpha=None,
        beta=None,
        sigma=None,
        omega=None,
        rank=None,
        path=None,
        callbacks=None,
        **kwargs):
    """
    vlgp main function

    Parameters
    ----------
    y : ndarray
        obserbation
    obs_types : ndarray
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
    history_filter : int, optional
        history_filter length
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
    kwargs : dict, optional
        algorithm options. See fill_options()

    Returns
    -------
    dict
        fit
    """

    y = asarray(y)
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]

    # obs_types = check_y_type(obs_types)
    ntrial, nbin, obs_ndim = y.shape

    obs_types = obs_types if np.size(obs_types) > 0 else ['spike'] * obs_ndim  # all are spike trains by default

    # make design matrix of regression
    h = empty((obs_ndim, ntrial, nbin, 1 + history_filter), dtype=float)
    for obs_dim in range(obs_ndim):
        for trial in range(ntrial):
            h[obs_dim, trial, :] = add_constant(lagmat(y[trial, :, obs_dim], lag=history_filter))

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent

    model = dict(y=y,
                 channel=obs_types,
                 dyn_ndim=dyn_ndim,
                 history_filter=history_filter,
                 h=h,
                 mu=mu,
                 a=a,
                 b=b,
                 sigma=sigma,
                 omega=omega,
                 rank=rank,
                 x=z,
                 alpha=alpha,
                 beta=beta,
                 path=path,
                 options=kwargs)

    initialize(model)

    callbacks = callbacks or []

    printer = Printer()
    callbacks.extend([printer.print])

    if path is not None:
        saver = Saver()
        callbacks.extend([saver.save])

    try:
        vem(model, callbacks)
    finally:
        # print('\nExiting...\n')
        printer.print(model)
        saver.save(model, force=True)

    return model


def predict(z, a, b, y=None, v=None):
    """
    Predict firing rate

    Parameters
    ----------
    z : ndarray
        latent
    a : ndarray
        loading
    b : ndarray
        history filter
    y : ndarray
        spike trains for history filter
    v : ndarray
        posterior variance

    Returns
    -------
    ndarray
        predicted firing rate
    """
    ntrial, nbin, z_ndim = z.shape
    y_ndim = a.shape[1]
    history_filter = b.shape[0] - 1

    shape_out = (ntrial, nbin, y_ndim)
    # regression (h dot b) part
    if y is None:
        y = np.zeros(shape_out)

    hb = empty(shape_out)
    for y_dim in range(y_ndim):
        for trial in range(ntrial):
            h = add_constant(lagmat(y[trial, :, y_dim], lag=history_filter))
            hb[trial, :, y_dim] = h @ b[:, y_dim]
    eta = z.reshape((-1, z_ndim)) @ a + hb.reshape((-1, y_ndim))
    r = sexp(eta + 0.5 * v.reshape((-1, z_ndim)) @ (a ** 2)) if v is not None else sexp(eta)
    return np.reshape(r, shape_out)
