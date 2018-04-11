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


    return trials, params
