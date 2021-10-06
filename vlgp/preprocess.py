import numpy as np


def initialize(trials, params, config):
    """Make skeleton"""
    # TODO: fast initialization for large dataset
    from sklearn.decomposition import FactorAnalysis

    zdim = params["zdim"]
    xdim = params["xdim"]

    # TODO: use only a subsample of trials?
    y = np.concatenate([trial["y"] for trial in trials], axis=0)
    subsample = np.random.choice(y.shape[0], max(y.shape[0] // 10, 50))
    ydim = y.shape[-1]

    if params.get('transform') is None:
        fa = FactorAnalysis(n_components=zdim, random_state=0)
        z = fa.fit_transform(y[subsample, :])
        a = fa.components_
        params['transform'] = fa.transform
    transform = params['transform']
    z = transform(y[subsample, :])
    
    b = np.log(np.maximum(np.mean(y, axis=0, keepdims=True), config["eps"]))
    noise = np.var(y[subsample, :] - z @ a, ddof=0, axis=0)

    # stupid way of update
    # two cases
    # 1) no key
    # 2) empty value (None)
    if params.get("a") is None:
        params.update(a=a)
    if params.get("b") is None:
        params.update(b=b)
    if params.get("noise") is None:
        params.update(noise=noise)

    for trial in trials:
        length = trial["y"].shape[0]

        if trial.get("mu") is None:
            trial.update(mu=transform(trial["y"]))

        if trial.get("x") is None:
            trial.update(x=np.ones((length, xdim, ydim)))

        trial.update({"w": np.zeros((length, zdim)), "v": np.zeros((length, zdim))})


def get_params(trials, zdim, **kwargs):
    """
    Define default initial parameters here
    :param trials:
    :param zdim:
    :param kwargs:
    :return:
    """
    y = trials[0]["y"]
    ydim = y.shape[-1]
    lik = kwargs.get("lik", "poisson")
    xdim = max(kwargs.get("history", 0), 1)

    if not isinstance(lik, list):
        lik = [lik] * ydim
    lik = np.asarray(lik)

    params = {
        "ydim": ydim,
        "zdim": zdim,
        "xdim": xdim,
        "a": kwargs.get("a", None),
        "b": kwargs.get("b", None),
        "noise": kwargs.get("noise", np.full(ydim, fill_value=1.0)),
        "sigma": kwargs.get("sigma", np.full(zdim, fill_value=1.0)),
        "omega": kwargs.get("omega", np.full(zdim, fill_value=kwargs["omega_bound"][1])),
        "rank": 50,  # TODO: consider merge with window in config
        "gp_noise": 1e-4,
        "dt": 1,
        "likelihood": lik,
    }

    return params


def get_config(**kwargs):
    config = {
        "constrain_loading": "fro",
        "constrain_latent": False,
        "use_hessian": True,
        "eps": 1e-8,  # small constant preventing numerical instability
        "tol": 1e-8,  # relative tolerance to check convergence
        "min_iter": 5,  # always run at least so many iterations
        "method": "VB",  # VB or MAP
        "learning_rate": 1.0,  # not used for Hessian
        "max_iter": 20,  # number of iterations of EM
        "Eniter": 25,  # number of interations inside E step
        "Mniter": 25,  # number of interations inside M step
        "Hstep": True,  # learn hyperparameters
        "da_bound": 5.0,  # clip the update to loading matrix
        "db_bound": 5.0,  # clip the update to bias
        "dmu_bound": 5.0,  # clip the update to posterior mean
        "omega_bound": (5e-4, 5e-2),  # limits of lengthscale
        "window": 50,  # window size that the trials are cut into
        "saving_interval": 60 * 30,  # time interval of saving snapshots
        "callbacks": [],  # functions are called every iteration
        "parallel": False,
    }

    updates = {k: v for k, v in kwargs.items() if k in config}  # discard unknown args

    config.update(updates)

    return config


def fill_trials(trials):
    for i, trial in enumerate(trials):
        trial["cut"] = i
        trial.setdefault("w", np.zeros_like(trial["mu"]))
        trial.setdefault("v", np.zeros_like(trial["mu"]))
        trial.setdefault("dmu", np.zeros_like(trial["mu"]))


def fill_params(params):
    params.setdefault("da", np.zeros_like(params["a"]))
    params.setdefault("db", np.zeros_like(params["b"]))
