from .callback import Saver
from .core import vem
from .preprocess import check_data
from .util import get_default_config

__all__ = ["fit"]


def fit(trials, n_factors, **kwargs):
    """
    vLGP API

    Parameters
    ----------
    y : ndarray
        obserbation
    lik : ndarray
        types of observation dimensions, 'spike' or 'lfp'
    lat_dim : int
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
    config = get_default_config()
    # TODO: update config
    # check_config(config, **kwarg)

    # initialization
    trials, params = check_data(trials, config)

    callbacks = config['callbacks']
    saver = None
    path = config.get('path', None)
    if path is not None:
        saver = Saver()
        callbacks.extend([saver.save])

    try:
        vem(trials, params, config)
    finally:
        # printer.print(model)
        if saver is not None:
            saver.save(trials, params, config, force=True)

    return


# def predict(**kwargs):
#     """
#     Predict firing rate
#
#     Parameters
#     ----------
#     model : dict
#         fitted model
#     z : ndarray
#         latent
#     a : ndarray
#         loading
#     b : ndarray
#         regression
#     v : ndarray
#         posterior variance
#     maxrate : float
#         maximum predicted firing rate
#     y : ndarray
#         spike trains for history filter
#
#     Returns
#     -------
#     ndarray
#         predicted firing rate
#     """
#
#     model = kwargs.get('model')
#     if model is not None:
#         z = model['mu']
#         y = model['y']
#         a = model['a']
#         b = model['b']
#         v = model['v']
#     else:
#         z = kwargs.get('z')
#         y = kwargs.get('y')
#         a = kwargs.get('a')
#         b = kwargs.get('b')
#         v = kwargs.get('v')
#
#     ntrial, nbin, z_dim = z.shape
#     y_dim = a.shape[1]
#     history = b.shape[0] - 1
#
#     shape_out = (ntrial, nbin, y_dim)
#     # regression (h dot b) part
#     if y is None:
#         y = np.zeros(shape_out)
#
#     hb = empty(shape_out)
#     for n in range(y_dim):
#         for trial in range(ntrial):
#             h = add_constant(lagmat(y[trial, :, n], lag=history))
#             hb[trial, :, n] = h @ b[:, n]
#     eta = z.reshape((-1, z_dim)) @ a + hb.reshape((-1, y_dim))
#     if v is not None:
#         r = np.exp(eta + 0.5 * v.reshape((-1, z_dim)) @ (a ** 2))
#     else:
#         r = np.exp(eta)
#     maxrate = kwargs.get('maxrate', np.exp(20))
#     np.clip(r, 0, maxrate, out=r)
#     return np.reshape(r, shape_out)
