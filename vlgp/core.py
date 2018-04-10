"""
The inference algorithm
introduce a new format of fit
trial isolation
unequal trial ready
"""
import logging
import warnings

import numpy as np
from numpy import identity, einsum
from scipy.linalg import solve, norm, svd, LinAlgError, cholesky, cho_solve
from scipy.spatial.distance import squareform, pdist

from .evaluation import timer
from .math import trunc_exp, ichol_gauss

logger = logging.getLogger(__name__)

__all__ = ["vem", "cut_trials"]


def estep(trials, params, config):
    """Update variational distribution q (E step)"""
    niter = config['Eniter']  # maximum number of iterations
    if niter < 1:
        return

    # See the explanation in mstep.
    constrain_loading(trials, params, config)

    # dimenionalities
    ydim = params['ydim']
    xdim = params['xdim']
    zdim = params['zdim']
    rank = params['rank']  # rank of prior covariance
    likelihood = params['likelihood']

    # misc
    dmu_bound = config['dmu_bound']
    tol = config['tol']
    method = config['method']

    poiss_mask = (likelihood == "poisson")
    gauss_mask = (likelihood == "gaussian")
    ###
    # print(poiss_mask)
    # print(gauss_mask)
    ##

    # parameters
    a = params['a']
    b = params['b']
    noise = params['noise']
    gauss_noise = noise[gauss_mask]

    Ir = identity(rank)
    # boolean indexing creates copies
    # pull indexing out of the loop for performance

    for i in range(niter):
        # TODO: parallel trials ?
        for trial in trials:
            y = trial['y']
            x = trial['x']
            mu = trial['mu']
            w = trial['w']
            v = trial['v']
            dmu = trial['dmu']

            prior = params['cholesky'][y.shape[0]]  # TODO: adapt unequal lengths, move into trials

            residual = np.empty_like(y, dtype=float)
            U = np.empty_like(y, dtype=float)

            y_poiss = y[:, poiss_mask]
            y_gauss = y[:, gauss_mask]

            ###
            # print(y_poiss.shape)
            # print(y_gauss.shape)
            ###

            xb = einsum('ijk, jk -> ik', x, b)
            eta = mu @ a + xb
            r = trunc_exp(eta + 0.5 * v @ (a ** 2))

            ###
            # print(xb.shape)
            ###

            # mean of y
            mean_gauss = eta[:, gauss_mask]
            mean_poiss = r[:, poiss_mask]

            ###
            # print(y_poiss.shape, mean_poiss.shape)
            # print(y_gauss.shape, mean_gauss.shape, gauss_noise.shape)
            ###

            for l in range(zdim):
                G = prior[l]
                ###
                # print(G.shape)
                ###

                # working residuals
                # extensible to many other distributions
                # see GLM's working residuals

                residual[:, poiss_mask] = y_poiss - mean_poiss
                residual[:, gauss_mask] = (y_gauss - mean_gauss) / gauss_noise
                ###
                # print(w.shape)
                ###
                wadj = w[:, [l]]  # keep dimension
                ###
                # print(G.shape, wadj.shape)
                ###
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[l, :])) - mu[:, l]
                try:
                    M = solve(Ir + GtWG, (wadj * G).T @ u, sym_pos=True)
                    delta_mu = u - G @ ((wadj * G).T @ u) + G @ (GtWG @ M)
                    clip(delta_mu, dmu_bound)
                except Exception as e:
                    logger.exception(repr(e), exc_info=True)
                    delta_mu = 0

                dmu[:, l] = delta_mu
                mu[:, l] += delta_mu

            # TODO: remove duplicated computation
            eta = mu @ a + xb
            r = trunc_exp(eta + 0.5 * v @ (a ** 2))
            U[:, poiss_mask] = r[:, poiss_mask]
            U[:, gauss_mask] = 1 / gauss_noise
            w = U @ (a.T ** 2)
            if method == 'VB':
                for l in range(zdim):
                    G = prior[l]
                    GtWG = G.T @ (w[:, l, np.newaxis] * G)
                    try:
                        M = solve(Ir + GtWG, GtWG, sym_pos=True)
                        v[:, l] = np.sum(
                            G * (G - G @ GtWG + G @ (GtWG @ M)), axis=1)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)

            # make sure save all changes
            # TODO: make inline modification
            trial['mu'] = mu
            trial['w'] = w
            trial['v'] = v
            trial['dmu'] = dmu

        # center over all trials if not only infer posterior
        # constrain_mu(model)

        if norm(dmu) < tol * norm(mu):
            break


def mstep(trials, params, config):
    """Optimize loading and regression (M step)"""
    niter = config['Mniter']  # maximum number of iterations
    if niter < 1:
        return

    # It's more proper to constrain the latent before mstep.
    # If the parameters are fixed, it's no need to optimize the posterior.
    # Besides, the constraint modifies the loading and bias.
    constrain_latent(trials, params, config)

    # dimenionalities
    ydim = params['ydim']
    xdim = params['xdim']
    zdim = params['zdim']
    rank = params['rank']  # rank of prior covariance
    ntrial = len(trials)  # number of trials

    # parameters
    a = params['a']
    b = params['b']
    likelihood = params['likelihood']
    noise = params['noise']
    poiss_mask = (likelihood == "poisson")
    gauss_mask = (likelihood == "gaussian")
    gauss_noise = noise[gauss_mask]
    da = params['da']
    db = params['db']

    # misc
    use_hessian = config['use_hessian']
    da_bound = config['da_bound']
    db_bound = config['db_bound']
    tol = config['tol']
    method = config['method']
    learning_rate = config['learning_rate']

    y = np.concatenate([trial['y'] for trial in trials], axis=0)
    x = np.concatenate([trial['x'] for trial in trials], axis=0)  # TODO: check dimensionality of x
    mu = np.concatenate([trial['mu'] for trial in trials], axis=0)
    v = np.concatenate([trial['v'] for trial in trials], axis=0)

    for i in range(niter):
        eta = mu @ a + einsum('ijk, jk -> ik', x, b)
        # (time, regression, neuron) x (regression, neuron) -> (time, neuron)  # TODO: use matmul broadcast
        r = trunc_exp(eta + 0.5 * v @ (a ** 2))
        noise = np.var(y - eta, axis=0, ddof=0)  # MLE

        for n in range(ydim):
            if likelihood[n] == "poisson":
                # loading
                mu_plus_v_times_a = mu + v * a[:, n]
                grad_a = mu.T @ y[:, n] - mu_plus_v_times_a.T @ r[:, n]

                if use_hessian:
                    nhess_a = mu_plus_v_times_a.T @ (
                            r[:, n, np.newaxis] * mu_plus_v_times_a)
                    nhess_a[np.diag_indices_from(nhess_a)] += r[:, n] @ v

                    try:
                        delta_a = solve(nhess_a, grad_a, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_a = learning_rate * grad_a
                else:
                    delta_a = learning_rate * grad_a

                clip(delta_a, da_bound)
                da[:, n] = delta_a
                a[:, n] += delta_a

                # regression
                grad_b = x[..., n].T @ (y[:, n] - r[:, n])

                if use_hessian:
                    nhess_b = x[..., n].T @ (r[:, np.newaxis, n] * x[..., n])
                    try:
                        delta_b = solve(nhess_b, grad_b, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_b = learning_rate * grad_b
                else:
                    delta_b = learning_rate * grad_b

                clip(delta_b, db_bound)
                db[:, n] = delta_b
                b[:, n] += delta_b
            elif likelihood[n] == "gaussian":
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                M = mu.T @ mu
                M[np.diag_indices_from(M)] += np.sum(v, axis=0)
                a[:, n] = solve(M, mu.T @ (
                        y[:, n] - x[..., n] @ b[:, n]),
                                sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, n] = solve(x[..., n].T @ x[..., n],
                                x[..., n].T @ (y[:, n] - mu @ a[:, n]),
                                sym_pos=True)
                b[1:, n] = 0
                # TODO: only make history filter components zeros
            else:
                pass

        # update parameters in fit
        # TODO: make inline modification
        params['a'] = a
        params['b'] = b
        params['noise'] = noise
        # normalize loading by latent and rescale latent
        # constrain_a(model)

        if norm(da) < tol * norm(a) and norm(db) < tol * norm(b):
            break


def hstep(trials, params, config):
    """Wrapper of hyperparameters tuning"""
    if not config['Hstep']:
        return

    gp_small_segments(trials, params, config)


def vem(trials, params, config):
    """Variational EM
    This function implements the algorithm.
    """
    # this function should not know if the trials are original or segmented ones
    # the caller determines which to use
    # pass segments to speed up estimation and hyperparameter tuning
    # the caller gets runtime

    callbacks = config['callbacks']

    tol = config['tol']
    niter = config['EMniter']

    # profile and debug purpose
    # invalid every new run
    runtime = {
        'it': 0,
        'e_elapsed': [],
        'm_elapsed': [],
        'h_elapsed': [],
        'em_elapsed': []
    }

    # disposable temporary arrays of updates
    # TODO: maybe put this outside for space efficiency
    params.setdefault('da', np.zeros_like(params['a']))
    params.setdefault('db', np.zeros_like(params['b']))
    for trial in trials:
        trial.setdefault('w', np.zeros_like(trial['mu']))
        trial.setdefault('v', np.zeros_like(trial['mu']))
        trial.setdefault('dmu', np.zeros_like(trial['mu']))
    make_cholesky(trials, params, config)
    update_w(trials, params, config)
    update_v(trials, params, config)

    #######################
    # iterative algorithm #
    #######################

    # disable gabbage collection during the iterative procedure
    for it in range(niter):
        runtime['it'] += 1

        with timer() as em_elapsed:
            ##########
            # E step #
            ##########
            with timer() as estep_elapsed:
                estep(trials, params, config)

            ##########
            # M step #
            ##########
            with timer() as mstep_elapsed:
                mstep(trials, params, config)

            ###################
            # H step #
            ###################
            with timer() as hstep_elapsed:
                hstep(trials, params, config)

        # print("Iter {:d}, ELBO: {:.3f}".format(it, lbound))

        runtime['e_elapsed'].append(estep_elapsed())
        runtime['m_elapsed'].append(mstep_elapsed())
        runtime['h_elapsed'].append(hstep_elapsed())
        runtime['em_elapsed'].append(em_elapsed())

        config['runtime'] = runtime

        for callback in callbacks:
            try:
                callback(trials, params, config)
            finally:
                pass

        #####################
        # convergence check #
        #####################
        mu = np.concatenate([trial['mu'] for trial in trials], axis=0)
        a = params['a']
        b = params['b']
        dmu = np.concatenate([trial['dmu'] for trial in trials], axis=0)
        da = params['da']
        db = params['db']

        converged = norm(dmu) < tol * norm(mu) and \
                    norm(da) < tol * norm(a) and \
                    norm(db) < tol * norm(b)

        should_stop = converged

        if should_stop:
            break

    ##############################
    # end of iterative procedure #
    ##############################


def clip(a, lbound, ubound=None):
    """Clip an array by given bounds in place"""
    if ubound is None:
        assert lbound > 0
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(a, lbound, ubound, out=a)


def constrain_latent(trials, params, config):
    """Center and scale latent mean"""
    constraint = config['constrain_latent']

    if not constraint or constraint == 'none':
        return

    mu = np.concatenate([trial['mu'] for trial in trials], axis=0)
    mean_over_trials = mu.mean(axis=0, keepdims=True)
    std_over_trials = mu.std(axis=0, keepdims=True)

    if constraint in ('location', 'both'):
        for trial in trials:
            trial['mu'] -= mean_over_trials
        # compensate bias
        # commented to isolated from changing external variables
        params['b'][0, :] += np.squeeze(mean_over_trials @ params['a'])

    if constraint in ('scale', 'both'):
        for trial in trials:
            trial['mu'] /= std_over_trials
        # compensate loading
        # commented to isolated from changing external variables
        params['a'] *= std_over_trials.T


def constrain_loading(trials, params, config):
    """Normalize loading matrix"""
    constraint = config['constrain_loading']

    if not constraint or constraint == 'none':
        return

    eps = config['eps']
    a = params['a']

    if constraint == 'svd':
        u, s, v = svd(a, full_matrices=False)
        # A = USV
        us = a @ v.T
        for trial in trials:
            trial['mu'] = trial['mu'] @ us
        params['a'] = v
    else:
        if constraint == 'fro':
            s = norm(a, ord='fro') + eps
        else:
            s = norm(a, ord=constraint, axis=1, keepdims=True) + eps
        params['a'] /= s
        for trial in trials:
            trial['mu'] *= s.T


def gp_small_segments(trials, params, config):
    """Optimize hyperparameters"""
    zdim = params['zdim']
    length = params['length']
    rank = params['rank']
    dt = params['dt']  # binwidth, set to 1 temporarily

    # priors
    sigma = params['sigma']
    omega = params['omega']
    gp_noise = params['gp_noise']

    # trials
    mu = np.stack([trial['mu'] for trial in trials])
    w = np.stack([trial['w'] for trial in trials])
    window = config['window']
    t = np.arange(window) * dt  # absolute time

    for l in range(zdim):
        initial = (sigma[l] ** 2, omega[l], gp_noise)
        bounds = ((1e-3, 1), config['omega_bound'], (gp_noise / 2, gp_noise * 2))
        mask = np.array([0, 1, 0])

        # transpose each latent dimension to (window, #trials/segments)
        (sigmasq, omega_new, _), fun = optimze(t,
                                               mu[:, :, l].T,
                                               w[:, :, l].T,
                                               initial,
                                               bounds,
                                               mask=mask)
        if not np.any(np.isclose(omega_new, config['omega_bound'])):
            omega[l] = omega_new
        sigma[l] = np.sqrt(sigmasq)

    params['sigma'] = sigma
    params['omega'] = omega
    make_cholesky(trials, params, config)


def make_cholesky(trials, params, config):
    """Make incomplate Cholesky decomposition"""
    zdim = params['zdim']
    rank = params['rank']
    sigma = params['sigma']
    omega = params['omega']
    lengths = np.array([trial['y'].shape[0] for trial in trials])
    unique_lengths = np.unique(lengths)
    for t in unique_lengths:
        params['cholesky'] = dict()
        params['cholesky'][t] = np.array([ichol_gauss(t, omega[l], rank) * sigma[l] for l in range(zdim)])


def optimze(t, mu, w, params, bounds, mask):
    """Optimize hyperparameters of a single dimension"""
    from scipy.optimize import minimize
    log_params = np.log(params)
    log_bounds = np.log(bounds)

    def obj_func(x):
        expx = np.exp(x)
        post_cov = construct_posterior_cov(t, w, expx)
        ll, dll = elbo(expx, mask, t, mu, post_cov)
        return -ll, -dll

    try:
        res = minimize(obj_func, log_params, jac=True, bounds=log_bounds)
        log_params = res.x
        fun = res.fun
        # opt, fval, info = fmin_l_bfgs_b(obj_func, log_params,
        #                                 bounds=log_bounds)
    finally:
        pass
    params = np.exp(log_params)

    return params, fun


def elbo(params, mask, *args):
    """ELBO with full posterior covariance matrix"""
    t, mu, post_cov = args
    K, dK = kernel(t, params)
    dK *= mask[np.newaxis, np.newaxis, :]
    try:
        L = cholesky(K, lower=True)
    except LinAlgError:
        return -np.inf, np.zeros_like(params)

    Kinv = cho_solve((L, True), np.eye(K.shape[0]))  # K inverse

    if mu.ndim == 1:
        mu = mu[:, np.newaxis]

    alpha = cho_solve((L, True), mu)
    ll_dims = -0.5 * np.einsum('ik,ik->k', mu, alpha)
    tmp = np.einsum('ik,jk->ijk', alpha, alpha)
    tmp -= Kinv[:, :, np.newaxis]

    for i in range(post_cov.shape[-1]):
        KinvSigma = cho_solve((L, True), post_cov[:, :, i])
        ll_dims[i] -= 0.5 * np.trace(KinvSigma)
        tmp[:, :, i] += KinvSigma @ Kinv

    ll_dims -= np.log(np.diag(L)).sum()
    ll = ll_dims.sum(-1)

    dll_dims = 0.5 * np.einsum('ijl,ijk->kl', tmp, dK)
    dll = dll_dims.sum(-1)

    return ll, dll


def construct_posterior_cov(t, w, params):
    """Make full posterior covariance matrix for hyperparameter tuning"""
    while True:
        K, dK = kernel(t, params)
        try:
            L = cholesky(K, lower=True)
            break
        except LinAlgError:
            # return -np.inf, np.zeros_like(params)
            params[1] += np.log(10)  # increase omega until Cholesky works

    Kinv = cho_solve((L, True), np.eye(K.shape[0]))  # K inverse

    if w.ndim == 1:
        w = w[:, np.newaxis]

    S = np.zeros((K.shape[0], K.shape[0], w.shape[-1]))  # Sigma
    for i in range(w.shape[-1]):
        L_Sinv = cholesky(Kinv + np.diag(w[:, i]), lower=True)
        S[:, :, i] = cho_solve((L_Sinv, True), np.eye(K.shape[0]))

    return S


def kernel(x, params):
    """kernel matrix and derivatives"""
    sigmasq, omega, eps = params

    dists = pdist(x.reshape(-1, 1), metric='sqeuclidean')  # vector of pairwise squared distance
    Dsq = squareform(dists)  # distance matrix
    K = np.exp(- omega * Dsq)  # kernel matrix
    dK_dsigmasq = K
    # K *= 1.0 - eps  # fix variance = 1 - eps (noise variance)
    K *= sigmasq
    dK_dlnomega = - K * Dsq * omega
    K[np.diag_indices_from(K)] += eps
    dK_deps = np.eye(K.shape[0]) * eps
    dK = np.dstack([dK_dsigmasq, dK_dlnomega, dK_deps])
    return K, dK


def cut_trial(trial, window: int):
    """Cut a trial into small segments"""
    import math

    y = trial['y']
    x = trial['x']
    mu = trial['mu']
    w = trial['w']
    v = trial['v']

    length = y.shape[0]

    # allow overlapping segments if the trial length is not a multiplier of window
    # random sample the segment starting points
    num_segments = math.ceil(length / window)
    overlap = num_segments * window - length  # number of overlapping segments
    start = np.cumsum(np.full(num_segments, fill_value=window, dtype=int)) - window
    offset = np.cumsum(np.append([0], np.random.multinomial(overlap, np.ones(num_segments - 1) / (num_segments - 1))))
    start -= offset
    slices = [np.s_[s:s + window] for s in start]
    segments = [{'y': y[s, :], 'x': x[s, ...], 'mu': mu[s, :], 'w': w[s, :], 'v': v[s, :]} for s in slices]
    return segments


def cut_trials(trials, params, config):
    """Cut all trials"""
    window = config['window']
    if window and window is not None:
        return np.concatenate([cut_trial(trial, window) for trial in trials])  # concatenate segments
    else:
        return trials


def update_w(trials, params, config):
    likelihood = params['likelihood']
    poiss_mask = (likelihood == "poisson")
    gauss_mask = (likelihood == "gaussian")

    a = params['a']
    b = params['b']
    noise = params['noise']
    gauss_noise = noise[gauss_mask]

    for trial in trials:
        y = trial['y']
        x = trial['x']
        mu = trial['mu']
        w = trial.setdefault('w', np.zeros_like(mu))
        v = trial.setdefault('v', np.zeros_like(mu))

        # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
        eta = mu @ a + einsum('ijk, jk -> ik', x, b)
        r = trunc_exp(eta + 0.5 * v @ (a ** 2))
        U = np.empty_like(r)
        U[:, poiss_mask] = r[:, poiss_mask]
        U[:, gauss_mask] = 1 / gauss_noise
        trial['w'] = U @ (a.T ** 2)


def update_v(trials, params, config):
    if config['method'] != "VB":
        return

    for trial in trials:
        zdim = params['zdim']
        mu = trial['mu']
        w = trial.setdefault('w', np.zeros_like(mu))
        v = trial.setdefault('v', np.zeros_like(mu))

        prior = params['cholesky'][mu.shape[0]]
        Ir = identity(prior[0].shape[-1])

        for l in range(zdim):
            G = prior[l]
            GtWG = G.T @ (w[:, [l]] * G)
            try:
                v[:, l] = np.sum(G * (G - G @ GtWG + G @ (GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True))), axis=1)
            except LinAlgError:
                warnings.warn("singular I + G'WG")
