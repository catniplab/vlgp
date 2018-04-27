import copy

from .preprocess import get_params, get_config, fill_trials, fill_params
from .callback import Saver, show
from .core import vem, estep, update_w, update_v
from .util import cut_trials
from .gp import make_cholesky
from .preprocess import initialize

__all__ = ["fit"]


def fit(trials, n_factors, **kwargs):
    """
    :param trials: list of trials
    :param n_factors: number of latent factors
    :param history: length of history filter
    :param x: external regressors
    :param lik: likelihood
    :param params: initial parameters
    :param kwargs: options
    :return:
    """
    print("\nvLGP")
    config = get_config(**kwargs)

    # add built-in callbacks
    callbacks = config['callbacks']
    if config.get('path', None) is not None:
        saver = Saver()
        callbacks.extend([show, saver.save])
    config['callbacks'] = callbacks

    # prepare parameters
    params = get_params(trials, n_factors, **kwargs)

    # initialization
    print("Initializing...")
    initialize(trials, params, config)

    # fill arrays
    fill_params(params)

    fill_trials(trials)
    make_cholesky(trials, params, config)
    update_w(trials, params, config)
    update_v(trials, params, config)

    subtrials = cut_trials(trials, params, config)
    make_cholesky(subtrials, params, config)

    fill_trials(subtrials)

    # params['initial'] = copy.deepcopy(params)
    # VEM
    print("Fitting")
    vem(subtrials, params, config)
    # E step only for inference given above estimated parameters and hyperparameters
    make_cholesky(trials, params, config)
    update_w(trials, params, config)
    update_v(trials, params, config)
    print("Inferring")
    estep(trials, params, config)

    return trials, params, config

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
