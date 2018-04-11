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

    params = {
        'ydim': ydim,
        'zdim': zdim,
        'xdim': xdim,
        'length': length,
        'a': a,
        'b': b,
        'noise': noise,
        'sigma': np.full(zdim, fill_value=1.0),
        'omega': np.full(zdim, fill_value=1e-4),
        'rank': length // 5,
        'gp_noise': 1e-4,
        'dt': 1,
        'tau': 500,
        'likelihood': np.array(['poisson'] * ydim)
    }

    trials = [{
        'y': trial['y'],
        'mu': fa.transform(trial['y']),
        'x': np.ones((length, 1, ydim)),
        'w': np.zeros((length, zdim)),
        'v': np.zeros((length, zdim)),
    } for trial in data]

    return trials, params
