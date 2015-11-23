import timeit
import numpy as np
from numpy import empty, empty_like, full, zeros, zeros_like, newaxis, tile, dstack, array
from numpy import identity, diag, einsum, inner, trace, exp, sum, mean, var, abs, sqrt, log
from numpy import inf, finfo, PINF
from scipy import linalg
from sklearn.decomposition.factor_analysis import FactorAnalysis
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from scipy.linalg import norm, lstsq

from constant import *
from algebra import ichol_gauss


def elbo(data, prior, posterior, param):
    nchannel, ntrial, ntime, lag = data['h'].shape  # neuron, trial, time, lag
    nlatent, _, rank = prior['chol'].shape  # latent, trial, time, rank

    eyer = identity(rank)

    y = data['y'].reshape((-1, nchannel))  # concatenate trials
    h = data['h'].reshape((nchannel, -1, lag))  # concatenate trials
    channel = data['channel']

    chol = prior['chol']

    mu = posterior['mu'].reshape((-1, nlatent))
    v = posterior['v'].reshape((-1, nlatent))

    a = param['a']
    b = param['b']
    noise = param['noise']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eta = mu.dot(a) + einsum('ijk, ki -> ji', h.reshape(nchannel, ntime * ntrial, lag), b)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))

    llspike = sum(y[:, spike] * eta[:, spike] - lam[:, spike])

    lllfp = - 0.5 * sum(((y[:, lfp] - eta[:, lfp]) ** 2 + v.dot(a[:, lfp] ** 2)) / noise[lfp] + log(noise[lfp]))

    lb = llspike + lllfp

    for m in range(ntrial):
        mu = posterior['mu'][m, :, :]
        w = posterior['w'][m, :, :]
        for l in range(nlatent):
            G = chol[l, :]
            GTWG = G.T.dot(w[:, l].reshape((ntime, 1)) * G)
            A = GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True))
            mu_div_G = linalg.lstsq(G, mu[:, l])[0]
            tr = ntime - trace(GTWG) + trace(A)
            lndet = np.linalg.slogdet(eyer - GTWG + A)[1]

            lb += -0.5 * inner(mu_div_G, mu_div_G) - 0.5 * tr + 0.5 * lndet

    return lb


def accumulate(accu, grad, decay=0):
    """adagrad

    Args:
        accu: accumulation matrix
        grad: new gradient
        decay: expoential decay

    Returns:

    """
    if decay > 0:
        return decay * accu + (1 - decay) * grad ** 2
    else:
        return accu + grad ** 2


def inferpost(data, prior, posterior, param, optim):
    nchannel, ntrial, ntime, lag = data['h'].shape  # neuron, trial, time, lag
    nlatent, _, rank = prior['chol'].shape  # latent, trial, time, rank

    channel = data['channel']

    chol = prior['chol']

    a = param['a']
    b = param['b']
    noise = param['noise']

    accu_grad_mu = optim['accu_grad_mu']
    decay = optim['decay']
    adagrad = optim['adagrad']
    eps = optim['eps']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eyer = identity(rank)
    R = empty((ntime, nchannel), dtype=float)
    U = empty((ntime, nchannel), dtype=float)

    for m in range(ntrial):
        # trial-wise
        y = data['y'][m, :, :]
        h = data['h'][:, m, :, :]
        mu = posterior['mu'][m, :, :]
        w = posterior['w'][m, :, :]
        v = posterior['v'][m, :, :]

        for l in range(nlatent):
            eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) . (lag, neuron)
            # noise = var(y - eta, axis=0, ddof=0)
            lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
            G = chol[l, :, :]
            grad_mu = (y[:, spike] - lam[:, spike]).dot(a[l, spike]) + \
                      ((y[:, lfp] - eta[:, lfp]) /
                       noise[lfp]).dot(a[l, lfp]) - linalg.lstsq(G.T, linalg.lstsq(G, mu[:, l])[0])[0]

            accu_grad_mu[:, l, m] = accumulate(accu_grad_mu[:, l, m], grad_mu, decay)

            if adagrad:
                wada = (w[:, l] + sqrt(eps + accu_grad_mu[:, l])).reshape((ntime, 1))  # adjusted by adagrad
            else:
                wada = w[:, l].reshape((ntime, 1))
            GTWG = G.T.dot(wada * G)

            R[:, spike] = y[:, spike] - lam[:, spike]
            R[:, lfp] = (y[:, lfp] - eta[:, lfp]) / noise[lfp]

            u = G.dot(G.T.dot(R.dot(a[l, :]))) - mu[:, l]
            delta_mu = u - G.dot((wada * G).T.dot(u)) + \
                      G.dot(GTWG.dot(linalg.solve(eyer + GTWG, (wada * G).T.dot(u), sym_pos=True)))

            mu[:, l] += delta_mu
            mu[:, l] -= mean(mu[:, l])
            scale = linalg.norm(mu[:, l], ord=inf)
            mu[:, l] /= scale

        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        # noise = var(y[:, :, m] - eta, axis=0, ddof=0)
        lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
        U[:, spike] = lam[:, spike]
        U[:, lfp] = 1 / noise[lfp]
        w[:, :] = U.dot(a.T ** 2)
        for l in range(nlatent):
            G = chol[l, :, :]
            GTWG = G.T.dot(w[:, l].reshape((ntime, 1)) * G)
            v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True)))),
                          axis=1)


def inferparam(data, prior, posterior, param, optim):
    nchannel, ntrial, ntime, lag = data['h'].shape  # neuron, trial, time, lag
    nlatent, _, rank = prior['chol'].shape  # latent, trial, time, rank

    y = data['y'].reshape((-1, nchannel))  # concatenate trials
    h = data['h'].reshape((nchannel, -1, lag))  # concatenate trials
    channel = data['channel']

    mu = posterior['mu'].reshape((-1, nlatent))
    v = posterior['v'].reshape((-1, nlatent))

    a = param['a']
    b = param['b']
    noise = param['noise']

    decay = optim['decay']
    adagrad = optim['adagrad']
    eps = optim['eps']
    accu_grad_a = optim['accu_grad_a']
    accu_grad_b = optim['accu_grad_b']

    for n in range(nchannel):
        eta = mu.dot(a) + einsum('ijk, ki -> ji', h.reshape(nchannel, ntime * ntrial, lag), b)
        lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
        if channel[n] == 'spike':
            # a
            va = v * a[:, n]  # (ntime, nlatent)
            wv = diag(lam[:, n].dot(v))
            grad_a = mu.T.dot(y[:, n]) - (mu + va).T.dot(lam[:, n])
            accu_grad_a[:, n] = accumulate(accu_grad_a[:, n], grad_a, decay)

            neghess_a = (mu + va).T.dot(lam[:, n, newaxis] * (mu + va)) + wv
            if adagrad:
                delta_a = linalg.solve(neghess_a + diag(sqrt(eps + accu_grad_a[:, n])), grad_a, sym_pos=True)
            else:
                delta_a = linalg.solve(neghess_a, grad_a, sym_pos=True)
            a[:, n] += delta_a

            # b
            grad_b = h[n, :].T.dot(y[:, n] - lam[:, n])
            accu_grad_b[:, n] = accumulate(accu_grad_b[:, n], grad_b, decay)
            neghess_b = h[n, :].T.dot(lam[:, n, newaxis] * h[n, :])
            if adagrad:
                b[:, n] += linalg.solve(neghess_b + diag(sqrt(eps + accu_grad_b[:, n])), grad_b, sym_pos=True)
            else:
                b[:, n] += linalg.solve(neghess_b, grad_b, sym_pos=True)
        elif channel[n] == 'lfp':
            # a's least squares solution for Gaussian channel
            # (m'm + diag(j'v))^-1 m'(y - Hb)
            a[:, n] = linalg.solve(mu.T.dot(mu) + diag(sum(v, axis=0)), mu.T.dot(y[:, n] - h[n, :].dot(b[:, n])),
                                   sym_pos=True)

            # b's least squares solution for Gaussian channel
            # (H'H)^-1 H'(y - ma)
            b[:, n] = linalg.solve(h[n, :].T.dot(h[n, :]), h[n, :].T.dot(y[:, n] - mu.dot(a[:, n])), sym_pos=True)
        else:
            print('Undefined channel')

    noise[:] = var(y - eta, axis=0, ddof=0)


def inference(data, prior, posterior, param, optim):

    # for backtracking
    goodposterior = {'mu': posterior['mu'].copy(),
                     'w': posterior['w'].copy(),
                     'v': posterior['v'].copy()}
    goodparam = {'a': param['a'].copy(),
                 'b': param['b'].copy(),
                 'noise': param['noise'].copy()}

    lb = empty(optim['niter'], dtype=float)
    elapsed = empty((optim['niter'], 3), dtype=float)

    lb[0] = elbo(data, prior, posterior, param)
    i = 1
    infer_start = timeit.default_timer()
    while not optim['converged'] and i < optim['niter']:
        # infer posterior
        iter_start = timeit.default_timer()
        post_start = timeit.default_timer()
        inferpost(data, prior, posterior, param, optim)
        post_end = timeit.default_timer()
        elapsed[i, 0] = post_end - post_start

        # infer parameter
        param_start = timeit.default_timer()
        inferparam(data, prior, posterior, param, optim)
        param_end = timeit.default_timer()
        elapsed[i, 1] = param_end - param_start

        lb[i] = elbo(data, prior, posterior, param)
        if lb[i] < lb[i - 1]:
            # backtracking
            posterior['mu'][:] = goodposterior['mu'][:]
            posterior['w'][:] = goodposterior['w'][:]
            posterior['v'][:] = goodposterior['v'][:]
            param['a'][:] = goodparam['a'][:]
            param['b'][:] = goodparam['b'][:]
            param['noise'][:] = param['noise'][:]
            print('ELBO decreased. Backtracking.')

            if i > optim['iadagrad'] and not optim['adagrad']:
                optim['adagrad'] = True
                print('Adagrad enabled.')
            else:
                print('Abort.')
        elif abs(lb[i] - lb[i-1]) < optim['tol'] * abs(lb[i-1]):
            optim['converged'] = True

        goodposterior['mu'][:] = posterior['mu'][:]
        goodposterior['w'][:] = posterior['w'][:]
        goodposterior['v'][:] = posterior['v'][:]
        goodparam['a'][:] = param['a'][:]
        goodparam['b'][:] = param['b'][:]
        goodparam['noise'][:] = param['noise'][:]

        iter_end = timeit.default_timer()
        elapsed[i, 2] = iter_end - iter_start

        i += 1
    infer_end = timeit.default_timer()
    optim['elapsed'] = elapsed[:i, :]
    optim['tot'] = infer_end - infer_start

    return lb[:i], posterior, param, optim


def multitrials(spike, lfp, sigma, omega, lag=0, r=500, niter=50, iadagrad=5, tol=1e-5):
    y = dstack(spike, lfp)
    ntrial, ntime, nchannel = y.shape

    channel = array(['spike'] * spike.shape[-1] + ['lfp'] * lfp.shape[-1])

    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)

    for n in range(nchannel):
        for m in range(ntrial):
            h[n, m, :] = add_constant(lagmat(y[m, :, n], maxlag=lag))
    data = {'y': y, 'h': h, 'channel': channel}

    assert sigma.shape == omega.shape
    nlatent = sigma.shape[0]
    chol = empty((nlatent, ntime, r), dtype=float)
    for l in range(nlatent):
        chol[l, :] = ichol_gauss(ntime, omega, r)
    prior = {'chol': chol}

    # initialize posterior
    fa = FactorAnalysis(n_components=nlatent, svd_method='lapack')
    mu = fa.fit_transform(y.reshape((-1, nchannel)))
    mu -= mu.mean(axis=0)
    a = fa.components_ * norm(mu, ord=inf, axis=0, keepdims=True).T
    mu /= norm(mu, ord=inf, axis=0)
    mu = mu.reshape((ntrial, ntime, nlatent))
    posterior = {'mu': mu, 'w': zeros((ntrial, ntime, nlatent)), 'v': zeros((ntrial, ntime, nlatent))}

    # initialize parameters
    b = empty((1 + lag, nchannel), dtype=float)
    for n in range(nchannel):
        b[:, n] = lstsq(h.reshape((nchannel, -1, 1 + lag))[n, :], y.reshape((-1, nchannel))[:, n])[0]

    y = y.reshape((-1, nchannel))
    noise = var(y, axis=0, ddof=0)
    param = {'a': a, 'b': b, 'noise': noise}

    optim = {'niter': niter,
             'iadagrad': iadagrad,
             'adagrad': False,
             'tol': tol,
             'converged': False}

    return inference(data, prior, posterior, param, optim)