import numpy as np


def initialize(trials, params, config):
    """Make skeleton"""
    from sklearn.decomposition import FactorAnalysis

    zdim = params['zdim']
    xdim = params['xdim']

    y = np.concatenate([trial['y'] for trial in trials], axis=0)
    ydim = y.shape[-1]
    fa = FactorAnalysis(n_components=zdim, random_state=0)
    z = fa.fit_transform(y)
    a = fa.components_
    b = np.log(np.mean(y, axis=0, keepdims=True))
    noise = np.var(y - z @ a, ddof=0, axis=0)

    params.update(a=a, b=b, noise=noise)

    for trial in trials:
        length = trial['y'].shape[0]
        trial.update({
            'mu': fa.transform(trial['y']),
            'x': np.ones((length, xdim, ydim)),
            'w': np.zeros((length, zdim)),
            'v': np.zeros((length, zdim))
        })


def get_params(trials, zdim, xdim, lik):
    y = trials[0]['y']
    ydim = y.shape[-1]

    if not isinstance(lik, list):
        lik = [lik] * ydim
    lik = np.asarray(lik)

    params = {
        'ydim': ydim,
        'zdim': zdim,
        'xdim': xdim,
        'a': np.zeros((zdim, ydim)),
        'b': np.zeros((xdim, ydim)),
        'noise': np.ones(ydim),
        'sigma': np.full(zdim, fill_value=1.0),
        'omega': np.full(zdim, fill_value=1e-4),
        'rank': 50,  # TODO: consider merge with window in config
        'gp_noise': 1e-4,
        'dt': 1,
        'tau': 500,
        'likelihood': lik
    }

    return params


def get_config(**kwargs):
    config = {
        'constrain_loading': 'fro',
        'constrain_latent': False,
        'use_hessian': True,
        'eps': 1e-8,
        'tol': 1e-5,  # loose
        'method': 'VB',
        'learning_rate': 1.0,  # no for hessian
        'EMniter': 50,
        'Eniter': 5,
        'Mniter': 5,
        'Hstep': True,
        'da_bound': 5.0,
        'db_bound': 5.0,
        'dmu_bound': 5.0,
        'omega_bound': (1e-5, 1e-3),
        'window': 50,
        'saving_interval': 60 * 30,  # sec
        'callbacks': []
    }

    config.update(kwargs)

    return config


def fill_trials(trials):
    for trial in trials:
        trial.setdefault('w', np.zeros_like(trial['mu']))
        trial.setdefault('v', np.zeros_like(trial['mu']))
        trial.setdefault('dmu', np.zeros_like(trial['mu']))


def fill_params(params):
    params.setdefault('da', np.zeros_like(params['a']))
    params.setdefault('db', np.zeros_like(params['b']))
