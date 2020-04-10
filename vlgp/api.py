import copy
import logging

import click

from . import gmap
from .preprocess import get_params, get_config, fill_trials, fill_params, initialize
from .callback import Saver, show
from .core import vem, update_w, update_v, infer
from .util import cut_trials
from .gp import make_cholesky

__all__ = ["fit"]

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
    callbacks = config["callbacks"]
    if "path" in config:
        saver = Saver()
        callbacks.extend([show, saver.save])
    config["callbacks"] = callbacks

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

