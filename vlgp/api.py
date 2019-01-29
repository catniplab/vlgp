import copy

from .preprocess import get_params, get_config, fill_trials, fill_params, initialize
from .callback import Saver, show
from .core import vem, update_w, update_v, infer
from .util import cut_trials
from .gp import make_cholesky

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
    print("Configuration\n", config)

    print(config)

    # add built-in callbacks
    callbacks = config["callbacks"]
    if "path" in config:
        saver = Saver()
        callbacks.extend([show, saver.save])
    config["callbacks"] = callbacks

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

    params["initial"] = copy.deepcopy(params)
    # VEM
    print("Fitting...")
    vem(subtrials, params, config)
    # E step only for inference given above estimated parameters and hyperparameters
    make_cholesky(trials, params, config)
    update_w(trials, params, config)
    update_v(trials, params, config)
    print("Inferring...")
    infer(trials, params, config)
    print("Done")

    model = {"trials": trials, "params": params, "config": config}
    return model
