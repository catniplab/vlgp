from numpy import asarray, newaxis, empty

from .callback import Saver, Progressor, Printer
from .core import initialize, vem, check_y_type
from .util import add_constant, lagmat


def fit(y,
        obs_types,
        dyn_ndim,
        exog=None,
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
    vLGP main function

    Parameters
    ----------
    y : ndarray
        obserbation
    obs_types : ndarray
        types of observation dimensions, 'spike' or 'lfp'
    dyn_ndim : int
        number of latent dimensions
    exog : ndarray, optional
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

    # print(model['options'])

    saver = Saver()
    printer = Printer()
    pbar = Progressor(model['options']['niter'])

    callbacks = callbacks or []
    callbacks.extend([pbar.update, saver.save, printer.print])
    try:
        vem(model, callbacks)
    finally:
        print('\nExiting...\n')
        printer.print(model)
        saver.save(model, force=True)

    return model
