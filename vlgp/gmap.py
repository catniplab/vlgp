import click
# import jax
import numpy as onp
import jax.numpy as np
from jax.numpy import linalg

from .evaluation import timer
from .gp import sekernel
from .preprocess import get_config, get_params, initialize, fill_params, fill_trials
from .util import cut_trials


def make_prior(trials, n_factors, dt, var, scale):
    for trial in trials:
        n, ydim = trial['y'].shape
        t = np.arange(n) * dt
        K = sekernel(t, var, scale)
        trial['bigK'] = np.kron(np.eye(n_factors), K)


def em(y, C, d, R, K, max_iter):
    zdim, ydim = C.shape
    n = K.shape[0]
    m = y.shape[0]
    bigK = np.kron(np.eye(zdim), K)
    bigR = np.kron(np.eye(n), R)
    Y = y.reshape(-1, ydim)

    for i in range(max_iter):
        # E step
        with timer() as e_elapsed:
            bigC = np.kron(C.T, np.eye(n))
            A = bigK @ bigC.T
            B = bigC @ A + bigR
            residual = y - d[None, :]
            residual = residual.transpose((0, 2, 1)).reshape(m, -1, 1)

            z = A[None, ...] @ linalg.solve(B[None, ...], residual)
            z = z.reshape(m, zdim, -1).transpose((0, 2, 1))
            z -= z.mean(axis=(0, 1), keepdims=True)

        # M step
        with timer() as m_elapsed:
            Z = z.reshape(-1, zdim)
            C, d, r = leastsq(Y, Z)  # Y = Z C + d
            R = np.diag(r ** 2)
            C /= linalg.norm(C)

        click.echo("Iteration {:4d}, E-step {:.2f}s, M-step {:.2f}s".format(i + 1, e_elapsed(), m_elapsed()))

    return z, C, d, R


def infer(trials, C, d, R):
    for trial in trials:
        n, ydim = trial['y'].shape
        _, zdim = trial['mu'].shape

        y = trial['y'] - d[None, :]
        y = y.T.reshape(-1, 1)
        bigC = np.kron(C.T, np.eye(n))
        bigK = trial['bigK']
        bigR = np.kron(np.eye(n), R)

        A = bigK @ bigC.T

        z = A @ linalg.solve(bigC @ A + bigR, y)
        trial['mu'] = z.reshape((zdim, -1)).T


def leastsq(Y, Z, constant=True):
    if constant:
        Z = np.column_stack([Z, np.ones(Z.shape[0])])
    C, r, *_ = onp.linalg.lstsq(Z, Y, rcond=None)
    # C = linalg.solve(Z.T @ Z, Z.T @ Y)
    return C[:-1, :], C[[-1], :], r


def loglik(y, z, C, d, R, var, scale, dt):
    zdim, ydim = C.shape
    m, n, _ = y.shape
    t = np.arange(n) * dt
    K = sekernel(t, var, scale)

    bigK = np.kron(np.eye(zdim), K)

    r = y - z @ C - d[None, :]
    r = r @ (1 / np.sqrt(R))
    Z = z.transpose((0, 2, 1)).reshape(m, -1, 1)

    return np.sum(r ** 2) + np.sum(Z.transpose((0, 2, 1)) @ linalg.solve(bigK[None, ...], Z)) + m * linalg.slogdet(bigK)[1]


def fit(trials, n_factors, **kwargs):
    """
    :param trials: list of trials
    :param n_factors: number of latent factors
    :param kwargs
    :return:
    """
    y, C, d, R, K = prepare(trials, n_factors, **kwargs)

    # EM
    click.echo("Fitting")
    z, C, d, R = em(y, C, d, R, K, kwargs['max_iter'])
    # params['a'], params['b'], params['R'] = C, d, R

    # Inference
    # click.echo("Inferring")
    # infer(trials, C, d, R)
    # click.secho("Done", fg="green")

    return y, z, C, d, R


def prepare(trials, n_factors, **kwargs):
    """
    :param trials: list of trials
    :param n_factors: number of latent factors
    :param kwargs
    :return:
    """
    config = get_config(**kwargs)
    kwargs["omega_bound"] = config["omega_bound"]
    params = get_params(trials, n_factors, **kwargs)

    # initialization
    click.echo("Initializing")
    with timer() as elapsed:
        initialize(trials, params, config)
    click.secho("Initialized {:.2f}s".format(elapsed()), fg="green")

    # fill arrays
    fill_params(params)
    params['R'] = np.eye(trials[0]['y'].shape[1])

    dt = kwargs['dt']
    var = kwargs['var']
    scale = kwargs['scale']
    fill_trials(trials)
    make_prior(trials, n_factors=n_factors, dt=dt, var=var, scale=scale)

    segments = cut_trials(trials, params, config)
    y = np.stack([segment['y'] for segment in segments])

    C, d, R = params['a'], params['b'], params['R']
    n = config["window"]
    t = np.arange(n) * dt
    K = sekernel(t, var, scale)

    return y, C, d, R, K
