import copy
import logging

import click

from . import gpfa
from .preprocess import get_params, get_config, fill_trials, fill_params, initialize
# from .callback import Saver, show
from .core import vem, update_w, update_v, infer
from .util import cut_trials
from .gp import make_cholesky
import numpy as np

__all__ = ["fit", "sample_posterior", "transform"]

logger = logging.getLogger(__name__)

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
    config = get_config(**kwargs)
    logger.info("\n".join(["{} : {}".format(k, v) for k, v in config.items()]))

    # add built-in callbacks
    # callbacks = config["callbacks"]
    # if "path" in config:
    #     saver = Saver()
    #     callbacks.extend([show, saver.save])
    # config["callbacks"] = callbacks

    # prepare parameters
    kwargs["omega_bound"] = config["omega_bound"]
    params = get_params(trials, n_factors, **kwargs)

    # initialization
    click.echo("Initializing")
    initialize(trials, params, config)
    click.secho("Initialized", fg="green")

    # fill arrays
    fill_params(params)

    fill_trials(trials)
    make_cholesky(trials, params, config)
    update_w(trials, params, config)
    update_v(trials, params, config)

    splits = cut_trials(trials, params, config)
    make_cholesky(splits, params, config)
    fill_trials(splits)

    params["initial"] = copy.deepcopy(params)

    # VEM
    click.echo("Fitting")
    vem(splits, params, config)
    # E step only for inference given above estimated parameters and hyperparameters
    make_cholesky(trials, params, config)
    update_w(trials, params, config)
    update_v(trials, params, config)

    click.echo("Inferring")
    infer(trials, params, config)
    click.secho("Done", fg="green")

    result = {"trials": trials, "params": params, "config": config}

    return result


def map2vi(trials, C, d, **kwargs):
    import numpy as np
    n_factors = trials[0]['mu'].shape[-1]
    config = get_config(**kwargs)
    logger.info("\n".join(["{} : {}".format(k, v) for k, v in config.items()]))

    # add built-in callbacks
    callbacks = config["callbacks"]
    if "path" in config:
        saver = Saver()
        callbacks.extend([show, saver.save])
    config["callbacks"] = callbacks

    # prepare parameters
    kwargs["omega_bound"] = config["omega_bound"]
    params = get_params(trials, n_factors, **kwargs)

    params['a'] = C
    params['b'] = np.log(d)

    make_cholesky(trials, params, config)
    update_w(trials, params, config)
    update_v(trials, params, config)
    config['max_iter'] = 5
    result = resume(trials, params, config)

    return result


def fastfit(trials, n_factors, dt, var, scale, max_iter=20, **kwargs):
    import numpy as np
    omega = np.full(n_factors, 0.5 / ((scale/dt) ** 2))

    # MAP
    y, C, d, R, K = gpfa.prepare(trials, n_factors, dt=dt, var=var, scale=scale)
    z, C, d, R = gpfa.em(y, C, d, R, K, max_iter)

    # vLGP
    result = map2vi(trials, C, d, omega=omega, **kwargs)

    return result


def resume(trials, params, config):
    click.echo("Inferring")
    infer(trials, params, config)
    click.secho("Done", fg="green")

    click.echo("Estimating parameters")
    Eniter = config['Eniter']
    config['Eniter'] = 0
    infer(trials, params, config)
    config['Eniter'] = Eniter
    click.secho("Done", fg="green")

    click.echo("Inferring")
    infer(trials, params, config)
    click.secho("Done", fg="green")

    result = {"trials": trials, "params": params, "config": config}

    return result

def sample_posterior(trial, params, nsamples, reg = 1e-6):
    '''Sample from the posterior for a single trial.
    It computes large covariance matrices, so it may be slow.
    If you only need the marginal variance, just use 'v'.

    Returns:
        ndarray of size (nsamples, bins, nfactors)
    '''

    chol = params['cholesky']
    #v = trial['v']  # marginal variance
    mu = trial['mu']
    w = trial['w']

    nbins, nfactors = mu.shape
    chol = chol[nbins]    # for the trials with length nbins

    samples = np.empty((nsamples, nbins, nfactors))

    for kfactor in range(nfactors):
        L = chol[kfactor,...]
        K = L @ L.T
        W = np.diag(w[:, kfactor])
        KK1 = np.linalg.inv(np.linalg.inv(K + reg * np.eye(K.shape[0])) + W)
        samples[:, :, kfactor] = np.random.multivariate_normal(mu[:, kfactor], KK1, size=nsamples)

    return samples


def transform(trials, params, config):
    """Infer latent factors using fitted model
    
    :param trials: list of trials
    :param params: parameters returned by fit
    :param config: configuration returned by fit

    :return:
        trials: trials containing latent factors
    """
    initialize(trials, params, config)
    fill_trials(trials)
    infer(trials, params, config)
    return trials
