import timeit
import warnings

import numpy as np
from numpy import empty, empty_like, full, zeros, zeros_like, newaxis, tile, dstack, array, arange, array_equal, asarray
from numpy import identity, diag, einsum, inner, trace, exp, sum, mean, var, abs, sqrt, log
from numpy import inf, finfo, PINF
from scipy import linalg
from sklearn.decomposition.factor_analysis import FactorAnalysis
from statsmodels.tsa.tsatools import lagmat
from scipy.linalg import lstsq
from numpy.linalg import norm

from link import sexp
from algebra import ichol_gauss, subspace
from util import add_constant


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
    lam = sexp(eta + 0.5 * v.dot(a ** 2))

    llspike = sum(y[:, spike] * eta[:, spike] - lam[:, spike])

    lllfp = - 0.5 * sum(((y[:, lfp] - eta[:, lfp]) ** 2 + v.dot(a[:, lfp] ** 2)) / noise[lfp] + log(noise[lfp]))

    ll = llspike + lllfp

    lb = ll

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

    return lb, ll


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
    nlatent, _, rank = prior['chol'].shape  # latent, time, rank

    channel = data['channel']

    chol = prior['chol']

    a = param['a']
    b = param['b']
    noise = param['noise']

    accu_grad_mu = optim['accu_grad_mu']
    adadecay = optim['adadecay']
    adagrad = optim['adagrad']
    eps = optim['eps']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eyer = identity(rank)
    res = empty((ntime, nchannel), dtype=float)
    U = empty((ntime, nchannel), dtype=float)

    for m in range(ntrial):
        # trial-wise
        y = data['y'][m, :, :]
        h = data['h'][:, m, :, :]
        mu = posterior['mu'][m, :, :]
        w = posterior['w'][m, :, :]
        v = posterior['v'][m, :, :]

        # eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) . (lag, neuron)
        # lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
        for l in range(nlatent):
            eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) . (lag, neuron)
            lam = sexp(eta + 0.5 * v.dot(a ** 2))
            G = chol[l, :, :]
            grad_mu = (y[:, spike] - lam[:, spike]).dot(a[l, spike]) + \
                      ((y[:, lfp] - eta[:, lfp]) /
                       noise[lfp]).dot(a[l, lfp]) - linalg.lstsq(G.T, linalg.lstsq(G, mu[:, l])[0])[0]

            accu_grad_mu[m, :, l] = accumulate(accu_grad_mu[m, :, l], grad_mu, adadecay)

            if adagrad:
                wada = (w[:, l] + sqrt(eps + accu_grad_mu[m, :, l])).reshape((ntime, 1))  # adjusted by adagrad
            else:
                wada = w[:, l].reshape((ntime, 1))
            GTWG = G.T.dot(wada * G)

            res[:, spike] = y[:, spike] - lam[:, spike]
            res[:, lfp] = (y[:, lfp] - eta[:, lfp]) / noise[lfp]

            u = G.dot(G.T.dot(res.dot(a[l, :]))) - mu[:, l]
            delta_mu = u - G.dot((wada * G).T.dot(u)) + \
                      G.dot(GTWG.dot(linalg.solve(eyer + GTWG, (wada * G).T.dot(u), sym_pos=True)))

            mu[:, l] += delta_mu
            mu[:, l] -= mean(mu[:, l])
            scale = linalg.norm(mu[:, l], ord=inf)
            mu[:, l] /= scale

        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        U[:, spike] = lam[:, spike]
        U[:, lfp] = 1 / noise[lfp]
        w[:, :] = U.dot(a.T ** 2)
        for l in range(nlatent):
            G = chol[l, :, :]
            GTWG = G.T.dot(w[:, l].reshape((ntime, 1)) * G)
            v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True)))),
                          axis=1)


def inferparam(data, prior, posterior, param, optim):
    nchannel, ntrial, ntime, lag1 = data['h'].shape  # neuron, trial, time, lag + 1
    nlatent, _, rank = prior['chol'].shape  # latent, time, rank

    y = data['y'].reshape((-1, nchannel))  # concatenate trials
    h = data['h'].reshape((nchannel, -1, lag1))  # concatenate trials
    channel = data['channel']

    mu = posterior['mu'].reshape((-1, nlatent))
    v = posterior['v'].reshape((-1, nlatent))

    a = param['a']
    b = param['b']
    noise = param['noise']

    adadecay = optim['adadecay']
    adagrad = optim['adagrad']
    eps = optim['eps']
    accu_grad_a = optim['accu_grad_a']
    accu_grad_b = optim['accu_grad_b']

    for n in range(nchannel):
        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        if channel[n] == 'spike':
            # a
            va = v * a[:, n]  # (ntime, nlatent)
            wv = diag(lam[:, n].dot(v))
            grad_a = mu.T.dot(y[:, n]) - (mu + va).T.dot(lam[:, n])
            accu_grad_a[:, n] = accumulate(accu_grad_a[:, n], grad_a, adadecay)

            neghess_a = (mu + va).T.dot(lam[:, n, newaxis] * (mu + va)) + wv
            if adagrad:
                delta_a = linalg.solve(neghess_a + diag(sqrt(eps + accu_grad_a[:, n])), grad_a, sym_pos=True)
            else:
                delta_a = linalg.solve(neghess_a, grad_a, sym_pos=True)
            a[:, n] += delta_a

            # b
            grad_b = h[n, :].T.dot(y[:, n] - lam[:, n])
            accu_grad_b[:, n] = accumulate(accu_grad_b[:, n], grad_b, adadecay)
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


def inference(data, prior, posterior, param, opt):
    print('\nInference starting')

    # for backtracking
    goodposterior = {'mu': posterior['mu'].copy(),
                     'w': posterior['w'].copy(),
                     'v': posterior['v'].copy()}
    goodparam = {'a': param['a'].copy(),
                 'b': param['b'].copy(),
                 'noise': param['noise'].copy()}

    lb = zeros(opt['niter'], dtype=float)
    ll = zeros(opt['niter'], dtype=float)
    elapsed = zeros((opt['niter'], 3), dtype=float)
    loadingAngle = zeros(opt['niter'], dtype=float)
    latentAngle = zeros(opt['niter'], dtype=float)

    lb[0], ll[0] = elbo(data, prior, posterior, param)
    i = 1
    converged = False
    infer_start = timeit.default_timer()
    while not converged and i < opt['niter']:
        iter_start = timeit.default_timer()

        # infer posterior
        post_start = timeit.default_timer()
        if opt['infer'] != 'param':
            inferpost(data, prior, posterior, param, opt)
        post_end = timeit.default_timer()
        elapsed[i, 0] = post_end - post_start
        x = data.get('x')
        if x is not None:
            latentAngle[i] = subspace(x.reshape(-1, x.shape[-1]), posterior['mu'].reshape(-1, x.shape[-1]))

        # infer parameter
        param_start = timeit.default_timer()
        if opt['infer'] != 'posterior':
            inferparam(data, prior, posterior, param, opt)
        param_end = timeit.default_timer()
        elapsed[i, 1] = param_end - param_start
        truea = param.get('truea')
        if truea is not None:
            loadingAngle[i] = subspace(truea.T, param['a'].T)

        lb[i], ll[i] = elbo(data, prior, posterior, param)
        if lb[i] < lb[i - 1]:
            # backtracking
            posterior['mu'][:] = goodposterior['mu'][:]
            posterior['w'][:] = goodposterior['w'][:]
            posterior['v'][:] = goodposterior['v'][:]
            param['a'][:] = goodparam['a'][:]
            param['b'][:] = goodparam['b'][:]
            param['noise'][:] = param['noise'][:]
            lb[i] = lb[i - 1]
            print('ELBO decreased. Backtracking.')

            if i > opt['iadagrad'] and not opt['adagrad']:
                opt['adagrad'] = True
                print('Adagrad enabled.')
            else:
                converged = True
        elif abs(lb[i] - lb[i-1]) < opt['tol'] * abs(lb[i-1]):
            converged = True

        goodposterior['mu'][:] = posterior['mu'][:]
        goodposterior['w'][:] = posterior['w'][:]
        goodposterior['v'][:] = posterior['v'][:]
        goodparam['a'][:] = param['a'][:]
        goodparam['b'][:] = param['b'][:]
        goodparam['noise'][:] = param['noise'][:]

        iter_end = timeit.default_timer()
        elapsed[i, 2] = iter_end - iter_start

        print('Iteration[{}], posterior elapsed: {:.2f}, parameter elapsed: {:.2f}, total elapsed: {:.2f}, ELBO: {:.4f}'.format(i, elapsed[i, 0], elapsed[i, 1], elapsed[i, 2], lb[i]))

        i += 1
    infer_end = timeit.default_timer()

    print('Inference ending\n')

    stat = {'ELBO': lb[:i], 'elapsed': elapsed[:i, :], 'loadingAngle': loadingAngle[:i], 'latentAngle': latentAngle[:i],
            'totalElapsed': infer_end - infer_start, 'converged': converged, 'LL': ll[:i]}

    print('{} iterations, ELBO: {:.4f}, elapsed: {:.2f}, converged: {}\n'.format(i - 1, lb[i - 1], stat['totalElapsed'],
                                                                                 stat['converged']))

    result = {'stat': stat, 'prior': prior, 'posterior': posterior, 'parameter': param, 'opt': opt}
    return result


def multitrials(spike, lfp, sigma, omega, x=None, ta=None, tb=None, lag=0, rank=500, niter=50, iadagrad=5, tol=1e-5):
    assert not (spike is None and lfp is None)

    if spike is None:
        spike = empty((0, 0, 0))

    if lfp is None:
        lfp = empty((0, 0, 0))

    spike = np.asarray(spike)
    lfp = np.asarray(lfp)

    if spike.ndim < 3:
        spike = np.atleast_3d(spike)
        spike = np.rollaxis(spike, axis=-1)
    if lfp.ndim < 3:
        lfp = np.atleast_3d(lfp)
        lfp = np.rollaxis(lfp, axis=-1)

    if lfp.size == 0:
        y = spike
    elif spike.size == 0:
        y = lfp
    else:
        print(spike.shape, lfp.shape)
        y = dstack((spike, lfp))
    ntrial, ntime, nchannel = y.shape

    channel = array(['spike'] * spike.shape[-1] + ['lfp'] * lfp.shape[-1])

    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)

    for n in range(nchannel):
        for m in range(ntrial):
            h[n, m, :] = add_constant(lagmat(y[m, :, n], maxlag=lag))
    data = {'x': x, 'y': y, 'h': h, 'channel': channel}

    assert sigma.shape == omega.shape
    nlatent = sigma.shape[0]
    chol = empty((nlatent, ntime, rank), dtype=float)
    for l in range(nlatent):
        chol[l, :] = ichol_gauss(ntime, omega[l], rank) * sigma[l]
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
    param = {'a': a, 'b': b, 'noise': noise, 'truea':ta, 'trueb': tb}

    opt = {'niter': niter,
           'infer': 'both',
           'iadagrad': iadagrad,
           'adagrad': False,
           'adadecay': 0,
           'eps': 1e-6,
           'accu_grad_mu': zeros_like(mu),
           'accu_grad_a': zeros_like(a),
           'accu_grad_b': zeros_like(b),
           'tol': tol}

    return inference(data, prior, posterior, param, opt)


def leaveoneout(model, opt):
    test = model['test']
    y = test['y']
    h = test['h']
    channel = test['channel']
    yhat = zeros_like(y)
    model['test']['yhat'] = yhat

    ntrial, ntime, nchannel = y.shape
    nlatent = model['training']['mu'].shape[-1]

    a = model['parameter']['a']
    b = model['parameter']['b']

    for n, ch in enumerate(channel):
        exceptn = arange(nchannel) != n
        ytrain = y[:, :, exceptn]
        htrain = h[exceptn, :]
        hn = h[n, :]

        data = {'y': ytrain, 'h': htrain, 'channel': channel[exceptn]}

        # initialize posterior
        fa = FactorAnalysis(n_components=nlatent, svd_method='lapack')
        mu = fa.fit_transform(data['y'].reshape((-1, nchannel - 1)))
        mu -= mu.mean(axis=0)
        mu /= norm(mu, ord=inf, axis=0)
        mu = mu.reshape((ntrial, ntime, nlatent))
        posterior = {'mu': mu, 'w': zeros((ntrial, ntime, nlatent)), 'v': zeros((ntrial, ntime, nlatent))}

        # initialize parameters
        param = {'a': model['parameter']['a'][:, exceptn], 'b': model['parameter']['b'][:, exceptn],
                 'noise': model['parameter']['noise'][exceptn]}

        opt['infer'] = 'posterior'

        result = inference(data, model['prior'], posterior, param, opt)

        if channel[n] == 'spike':
            yhat[:, :, n] = sexp(result['posterior']['mu'].reshape((-1, nlatent)).dot(a[:, n]) + hn.reshape((ntime, -1)).dot(b[:, n]))
        else:
            yhat[:, :, n] = result['posterior']['mu'].reshape((-1, nlatent)).dot(a[:, n]) + hn.reshape((ntime, -1)).dot(b[:, n])

    return model


def makedataset(spike, lfp, x=None, lag=0):
    assert not (spike is None and lfp is None)

    if spike is None:
        spike = empty((0, 0, 0))
    if lfp is None:
        lfp = empty((0, 0, 0))

    spike = np.asarray(spike)
    lfp = np.asarray(lfp)

    if spike.ndim < 3:
        spike = np.atleast_3d(spike)
        spike = np.rollaxis(spike, axis=-1)
    if lfp.ndim < 3:
        lfp = np.atleast_3d(lfp)
        lfp = np.rollaxis(lfp, axis=-1)

    if lfp.size == 0:
        y = spike
    elif spike.size == 0:
        y = lfp
    else:
        print(spike.shape, lfp.shape)
        y = dstack((spike, lfp))

    ntrial, ntime, nchannel = y.shape

    channel = array(['spike'] * spike.shape[-1] + ['lfp'] * lfp.shape[-1])

    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    for n in range(nchannel):
        for m in range(ntrial):
            h[n, m, :] = add_constant(lagmat(y[m, :, n], maxlag=lag))

    data = {'x': x, 'y': y, 'h': h, 'channel': channel}
    return data


def makeprior(sigma, omega, ntime, rank=500):
    assert sigma.shape == omega.shape

    if rank > ntime:
        rank = ntime
    nlatent = sigma.shape[0]
    chol = empty((nlatent, ntime, rank), dtype=float)
    for l in range(nlatent):
        chol[l, :] = ichol_gauss(ntime, omega[l], rank) * sigma[l]

    prior = {'chol': chol, 'sigma': sigma, 'omega': omega}
    return prior


def makeopt(infer='both', niter=50, iadagrad=5, adadecay=0, adaeps=1e-6, tol=1e-5):
    opt = {'niter': niter,
           'infer': 'both',
           'iadagrad': iadagrad,
           'adagrad': False,
           'adadecay': adadecay,
           'eps': adaeps,
           'tol': tol}
    return opt


def gpvb(spike, lfp,
         sigma, omega, rank=500,
         x=None,
         lag=0,
         truea=None, trueb=None,
         testidx=None,
         niter=50, tol=1e-5,
         adafter=5, adadecay=0, adaeps=1e-6):
    ntrial = spike.shape[0] if spike.ndim == 3 else 1
    if lfp is None:
        lfp = empty((ntrial, 0, 0))

    if testidx is None:
        training = makedataset(spike, lfp, x, lag)
        result = multitrials(spike, lfp, sigma, omega, x, truea, trueb, lag, rank, niter, adafter, tol)
        model = {'training': training, 'prior': result['prior'], 'parameter': result['parameter']}
        model['training']['stat'] = result['stat']
        model['training']['mu'] = result['posterior']['mu']
        model['training']['w'] = result['posterior']['w']
    else:
        testidx = asarray(testidx)
        testmask = full(ntrial, fill_value=False, dtype=bool)
        testmask[testidx] = True
        training = makedataset(spike[~testmask, :, :], lfp[~testmask, :, :], x, lag)
        test = makedataset(spike[testmask, :, :], lfp[testmask, :, :], x, lag)

        result = multitrials(spike[~testmask, :, :], lfp[~testmask, :, :], sigma, omega, x[~testmask, :, :] if x is not None else None, truea, trueb, lag, rank,
                             niter, adafter, tol)
        model = {'training': training, 'prior': result['prior'], 'parameter': result['parameter'], 'test': test}
        model['training']['stat'] = result['stat']
        model['training']['mu'] = result['posterior']['mu']
        model['training']['w'] = result['posterior']['w']
        leaveoneout(model, result['opt'])

    return model
