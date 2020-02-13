import numpy as np

from . import gmap


def speckled_cv(y, C, d, R, K, test_ratio, max_iter):
    test_mask = np.random.rand(*y.shape) < test_ratio  # mask matrix
    y_training = (1 - test_mask) * y

    z, C, d, R = gmap.em(y_training, C, d, R, K, max_iter)
    yhat = z @ C + d[None, :]
    error = elementwise_error(yhat, y, R)

    training_error = np.mean(error[~test_mask])
    test_error = np.mean(error[test_mask])
    return training_error, test_error


def elementwise_error(yhat, y, R, eps=1e-16):
    G = 1 / (eps + np.sqrt(R))
    r = yhat - y
    return (r @ G) ** 2


def gmap_speckled_cv(trials, max_n_factors, test_ratio=0.1, **kwargs):
    dt = kwargs['dt']
    var = kwargs['var']
    scale = kwargs['scale']
    max_iter = kwargs['max_iter']

    training_errors = []
    test_errors = []
    for n_factors in range(1, max_n_factors + 1):
        y, C, d, R, K = gmap.prepare(trials, n_factors, dt=dt, var=var, scale=scale, max_iter=max_iter)
        training_error, test_error = speckled_cv(y, C, d, R, K, test_ratio=test_ratio, max_iter=1)
        training_errors.append(training_error)
        test_errors.append(test_error)

    return training_errors, test_errors
