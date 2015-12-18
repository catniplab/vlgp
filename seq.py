"""
Functions of sequentially fit
"""
import timeit

from numpy import identity, einsum, trace, inner, empty, mean, inf, diag, newaxis, var, asarray, zeros, zeros_like, \
    sum, array_equal
from numpy.core.umath import sqrt, PINF, log
from numpy.linalg import norm, slogdet
from scipy.linalg import lstsq, eigh, solve
from sklearn.decomposition.factor_analysis import FactorAnalysis

from hyper import learngp
from inference import accumulate
from mathf import ichol_gauss, sexp
from util import add_constant, lagmat


def fillargs(**kwargs):
    kwargs['verbose'] = kwargs.get('verbose', False)
    kwargs['niter'] = kwargs.get('niter', 50)
    kwargs['infer'] = kwargs.get('infer', 'both')
    kwargs['learn_sigma'] = kwargs.get('learn_sigma', False)
    kwargs['learn_omega'] = kwargs.get('learn_omega', False)
    kwargs['nadjhess'] = kwargs.get('nadjhess', 5)
    kwargs['tol'] = kwargs.get('tol', 1e-5)
    kwargs['eps'] = kwargs.get('eps', 1e-6)
    kwargs['nhyper'] = kwargs.get('nhyper', 5)
    kwargs['decay'] = kwargs.get('decay', 0)
    return kwargs


def fit(y, channel, sigma, omega, x=None, alpha=None, beta=None, lag=0, rank=500, **kwargs):
    """Sequentially fit
    Args:
        y:
        channel:
        sigma:
        omega:
        x:
        alpha:
        beta:
        lag:
        rank:
        niter:
        nadjhess:
        tol:
        verbose:
        hyper:

    Returns:

    """
    kwargs = fillargs(**kwargs)
    assert sigma.shape == omega.shape

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., None]
    if y.ndim < 3:
        y = y[None, ...]
    channel = asarray(channel)
    ntrial, ntime, nchannel = y.shape
    nlatent = sigma.shape[0]

    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    for ichannel in range(nchannel):
        for itrial in range(ntrial):
            h[ichannel, itrial, :] = add_constant(lagmat(y[itrial, :, ichannel], lag=lag))

    chol = empty((nlatent, ntime, rank), dtype=float)
    for ilatent in range(nlatent):
        chol[ilatent, :] = ichol_gauss(ntime, omega[ilatent], rank) * sigma[ilatent]

    # initialize posterior
    fa = FactorAnalysis(n_components=nlatent, svd_method='lapack')
    mu = fa.fit_transform(y.reshape((-1, nchannel)))
    mu -= mu.mean(axis=0)
    a = fa.components_ * norm(mu, ord=inf, axis=0, keepdims=True).T
    mu /= norm(mu, ord=inf, axis=0)
    mu = mu.reshape((ntrial, ntime, nlatent))
    L = empty((ntrial, nlatent, ntime, rank))

    # initialize parameters
    b = empty((1 + lag, nchannel), dtype=float)
    for ichannel in range(nchannel):
        b[:, ichannel] = lstsq(h.reshape((nchannel, -1, 1 + lag))[ichannel, :], y.reshape((-1, nchannel))[:, ichannel])[
            0]

    noise = var(y.reshape((-1, nchannel)), axis=0, ddof=0)

    obj = {'y': y, 'channel': channel, 'h': h,
           'sigma': sigma, 'omega': omega, 'chol': chol, 'sigma0': sigma, 'omega0': omega,
           'mu': mu, 'w': zeros((ntrial, ntime, nlatent)), 'v': zeros((ntrial, ntime, nlatent)), 'L': L, 'x': x,
           'a': a, 'b': b, 'noise': noise, 'alpha': alpha, 'beta': beta}

    kwargs['accu_grad_mu'] = zeros_like(mu)
    kwargs['accu_grad_a'] = zeros_like(a)
    kwargs['accu_grad_b'] = zeros_like(b)

    inference = infer(obj, **kwargs)
    return inference


def inferpost(obj, **kwargs):
    nchannel, ntrial, ntime, lag = obj['h'].shape  # neuron, trial, time, lag
    nlatent, _, rank = obj['chol'].shape  # latent, time, rank

    which = obj['which']
    head = which + 1
    channel = obj['channel']
    chol = obj['chol'][:head, :]

    a = obj['a'][:head, :]
    b = obj['b']
    noise = obj['noise']

    accu_grad_mu = kwargs['accu_grad_mu']
    decay = kwargs['decay']
    adjhess = kwargs['adjhess']
    eps = kwargs['eps']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eyer = identity(rank)
    res = empty((ntime, nchannel), dtype=float)
    U = empty((ntime, nchannel), dtype=float)

    for itrial in range(ntrial):
        # trial-wise
        y = obj['y'][itrial, :]
        h = obj['h'][:, itrial, :, :]
        mu = obj['mu'][itrial, :, :head]
        w = obj['w'][itrial, :, :head]
        v = obj['v'][itrial, :, :head]
        L = obj['L'][itrial, :head, :]

        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) . (lag, neuron)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        G = chol[which, :, :]
        grad_mu = (y[:, spike] - lam[:, spike]).dot(a[which, spike]) + \
                  ((y[:, lfp] - eta[:, lfp]) /
                   noise[lfp]).dot(a[which, lfp]) - lstsq(G.T, lstsq(G, mu[:, which])[0])[0]

        accu_grad_mu[itrial, :, which] = accumulate(accu_grad_mu[itrial, :, which], grad_mu, decay)

        if adjhess:
            wadj = (w[:, which] + sqrt(eps + accu_grad_mu[itrial, :, which])).reshape((ntime, 1))  # adjusted Hessian
        else:
            wadj = w[:, which].reshape((ntime, 1))
        GtWG = G.T.dot(wadj * G)

        res[:, spike] = y[:, spike] - lam[:, spike]
        res[:, lfp] = (y[:, lfp] - eta[:, lfp]) / noise[lfp]

        u = G.dot(G.T.dot(res.dot(a[which, :]))) - mu[:, which]
        delta_mu = u - G.dot((wadj * G).T.dot(u)) + G.dot(GtWG.dot(solve(eyer + GtWG, (wadj * G).T.dot(u), sym_pos=True)))

        mu[:, which] += delta_mu
        mu[:, which] -= mean(mu[:, which])
        scale = norm(mu[:, which], ord=inf)
        mu[:, which] /= scale

        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        U[:, spike] = lam[:, spike]
        U[:, lfp] = 1 / noise[lfp]
        w[:] = U.dot(a.T ** 2)

        G = chol[which, :, :]
        GtWG = G.T.dot(w[:, which].reshape((ntime, 1)) * G)
        v[:, which] = sum(G * (G - G.dot(GtWG) + G.dot(GtWG.dot(solve(eyer + GtWG, GtWG, sym_pos=True)))), axis=1)

        A = eyer - GtWG + GtWG.dot(solve(eyer + GtWG, GtWG, sym_pos=True))  # A should be PD but numerically not
        eigval, eigvec = eigh(A)
        eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
        L[which, :] = G.dot(eigvec.dot(diag(sqrt(eigval))))  # lower posterior covariance


def inferparam(obj, **kwargs):
    nchannel, ntrial, ntime, lag1 = obj['h'].shape  # neuron, trial, time, lag + 1
    nlatent, _, rank = obj['chol'].shape  # latent, time, rank

    y = obj['y'].reshape((-1, nchannel))  # concatenate trials
    h = obj['h'].reshape((nchannel, -1, lag1))  # concatenate trials
    channel = obj['channel']
    which = obj['which']
    head = which + 1

    mu = obj['mu'][:, :, :head].reshape((-1, head))
    v = obj['v'][:, :, :head].reshape((-1, head))

    a = obj['a'][:head, :]
    b = obj['b']
    noise = obj['noise']

    decay = kwargs['decay']
    adjhess = kwargs['adjhess']
    eps = kwargs['eps']
    accu_grad_a = kwargs['accu_grad_a'][:head, :]
    accu_grad_b = kwargs['accu_grad_b']

    for ichannel in range(nchannel):
        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        if channel[ichannel] == 'spike':
            # a
            va = v * a[:, ichannel]  # (ntime, nlatent)
            wv = diag(lam[:, ichannel].dot(v))
            grad_a = mu.T.dot(y[:, ichannel]) - (mu + va).T.dot(lam[:, ichannel])
            accu_grad_a[:, ichannel] = accumulate(accu_grad_a[:, ichannel], grad_a, decay)

            neghess_a = (mu + va).T.dot(lam[:, ichannel, newaxis] * (mu + va)) + wv
            if adjhess:
                delta_a = solve(neghess_a + diag(sqrt(eps + accu_grad_a[:, ichannel])), grad_a, sym_pos=True)
            else:
                delta_a = solve(neghess_a, grad_a, sym_pos=True)
            a[:, ichannel] += delta_a

            # b
            grad_b = h[ichannel, :].T.dot(y[:, ichannel] - lam[:, ichannel])
            accu_grad_b[:, ichannel] = accumulate(accu_grad_b[:, ichannel], grad_b, decay)
            neghess_b = h[ichannel, :].T.dot(lam[:, ichannel, newaxis] * h[ichannel, :])
            if adjhess:
                b[:, ichannel] += solve(neghess_b + diag(sqrt(eps + accu_grad_b[:, ichannel])), grad_b, sym_pos=True)
            else:
                b[:, ichannel] += solve(neghess_b, grad_b, sym_pos=True)
        elif channel[ichannel] == 'lfp':
            # a's least squares solution for Gaussian channel
            # (m'm + diag(j'v))^-1 m'(y - Hb)
            a[:, ichannel] = solve(mu.T.dot(mu) + diag(sum(v, axis=0)),
                                   mu.T.dot(y[:, ichannel] - h[ichannel, :].dot(b[:, ichannel])),
                                   sym_pos=True)

            # b's least squares solution for Gaussian channel
            # (H'H)^-1 H'(y - ma)
            b[:, ichannel] = solve(h[ichannel, :].T.dot(h[ichannel, :]),
                                   h[ichannel, :].T.dot(y[:, ichannel] - mu.dot(a[:, ichannel])), sym_pos=True)
        else:
            print('Undefined channel')
    noise[:] = var(y - eta, axis=0, ddof=0)


def infer(obj, **kwargs):
    """Main inference procedure

    Args:
        obj:
        kwargs:

    Returns:

    """
    if kwargs['verbose']:
        print('\nInference starts.')

    ntrial, ntime, nlatent = obj['mu'].shape

    # for backtracking
    good_mu = obj['mu'].copy()
    good_w = obj['w'].copy()
    good_v = obj['v'].copy()
    good_a = obj['a'].copy()
    good_b = obj['b'].copy()
    good_noise = obj['noise'].copy()
    good_sigma = obj['sigma'].copy()
    good_omega = obj['omega'].copy()

    lb = zeros((kwargs['niter'], nlatent), dtype=float)
    ll = zeros((kwargs['niter'], nlatent), dtype=float)
    elapsed = zeros((kwargs['niter'], 3, nlatent), dtype=float)

    for which in range(nlatent):
        if kwargs['verbose']:
            print('Latent {}'.format(which + 1))
        obj['which'] = which
        kwargs['accu_grad_mu'].fill(0.0)
        kwargs['accu_grad_a'].fill(0.0)
        kwargs['accu_grad_b'].fill(0.0)
        lb[0, which], ll[0, which] = elbo(obj)
        iiter = 1
        converged = False
        adjhess = False
        # infer_start = timeit.default_timer()
        while not converged and iiter < kwargs['niter']:
            iter_start = timeit.default_timer()

            # infer posterior
            post_start = timeit.default_timer()
            if kwargs['infer'] != 'param':
                inferpost(obj, **kwargs, adjhess=adjhess)
            post_end = timeit.default_timer()
            elapsed[iiter, 0, which] = post_end - post_start

            # infer parameter
            param_start = timeit.default_timer()
            if kwargs['infer'] != 'posterior':
                inferparam(obj, **kwargs, adjhess=adjhess)
            param_end = timeit.default_timer()
            elapsed[iiter, 1, which] = param_end - param_start

            if iiter % kwargs['nhyper'] == 0 and (kwargs['learn_sigma'] or kwargs['learn_omega']):
                nlatent, ntime, rank = obj['chol'].shape
                gp = learngp(obj, latents=[which], **kwargs)
                obj['sigma'][which] = gp[0][which]
                obj['omega'][which] = gp[1][which]
                if kwargs['verbose']:
                    print('sigma: {} \nomega: {}'.format(obj['sigma'], obj['omega']))
                obj['chol'][which, :] = ichol_gauss(ntime, obj['omega'][which], rank) * obj['sigma'][which]

            lb[iiter, which], ll[iiter, which] = elbo(obj)
            if lb[iiter, which] < lb[iiter - 1, which]:
                # backtracking
                obj['mu'][:] = good_mu
                obj['w'][:] = good_w
                obj['v'][:] = good_v
                obj['a'][:] = good_a
                obj['b'][:] = good_b
                obj['noise'][:] = good_noise
                # if not array_equal(obj['omega'], good_omega):
                obj['sigma'][:] = good_sigma
                obj['omega'][:] = good_omega
                for ilatent in range(nlatent):
                    obj['chol'][ilatent, :] = ichol_gauss(ntime, obj['omega'][ilatent], rank) * obj['sigma'][ilatent]

                lb[iiter] = lb[iiter - 1]
                if kwargs['verbose']:
                    print('ELBO decreased. Backtracking.')
                if iiter > kwargs['nadjhess'] and not adjhess:
                    adjhess = True
                    if kwargs['verbose']:
                        print('Hessian adjustment enabled.')
                # else:
                #     converged = True
            elif abs(lb[iiter, which] - lb[iiter - 1, which]) < kwargs['tol'] * abs(lb[iiter - 1, which]):
                converged = True
            else:
                adjhess = False

            good_mu[:] = obj['mu']
            good_w[:] = obj['w']
            good_v[:] = obj['v']
            good_a[:] = obj['a']
            good_b[:] = obj['b']
            good_noise[:] = obj['noise']
            good_sigma[:] = obj['sigma']
            good_omega[:] = obj['omega']

            iter_end = timeit.default_timer()
            elapsed[iiter, 2, which] = iter_end - iter_start

            if kwargs['verbose']:
                print('[{}], posterior elapsed: {:.2f}, parameter elapsed: {:.2f}, '
                      'ELBO: {:.4f}, LL: {:.4f}'.format(iiter, elapsed[iiter, 0, which], elapsed[iiter, 1, which], lb[iiter, which], ll[iiter, which]))

            iiter += 1
        # infer_end = timeit.default_timer()

    # if kwargs['verbose']:
    #     print('Inference ends.\n')
    #     print('{} iterations, ELBO: {:.4f}, elapsed: {:.2f}, converged: {}\n'.format(iiter - 1,
    #                                                                                  lb[iiter - 1],
    #                                                                                  infer_end - infer_start,
    #                                                                                  converged))
    obj['ELBO'] = lb[:iiter, :]
    obj['Elapsed'] = elapsed[:iiter, :]
    obj['LL'] = ll[:iiter, :]
    return obj


def elbo(obj):
    nchannel, ntrial, ntime, lag = obj['h'].shape  # neuron, trial, time, lag
    _, _, rank = obj['chol'].shape  # latent, time, rank

    eyer = identity(rank)

    y = obj['y'].reshape((-1, nchannel))  # concatenate trials
    h = obj['h'].reshape((nchannel, -1, lag))  # concatenate trials
    channel = obj['channel']
    which = obj['which']
    head = which + 1
    chol = obj['chol']

    mu = obj['mu'][:, :, :head].reshape((-1, head))
    v = obj['v'][:, :, :head].reshape((-1, head))

    a = obj['a'][:head, :]
    b = obj['b']
    noise = obj['noise']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eta = mu.dot(a) + einsum('ijk, ki -> ji', h.reshape((nchannel, ntime * ntrial, lag)), b)
    lam = sexp(eta + 0.5 * v.dot(a ** 2))

    llspike = sum(y[:, spike] * eta[:, spike] - lam[:, spike])

    lllfp = - 0.5 * sum(((y[:, lfp] - eta[:, lfp]) ** 2 + v.dot(a[:, lfp] ** 2)) / noise[lfp] + log(noise[lfp]))

    ll = llspike + lllfp

    lb = ll

    for itrial in range(ntrial):
        mu = obj['mu'][itrial, :, :head]
        w = obj['w'][itrial, :, :head]
        for l in range(head):
            G = chol[l, :]
            GtWG = G.T.dot(w[:, l].reshape((ntime, 1)) * G)
            A = GtWG.dot(solve(eyer + GtWG, GtWG, sym_pos=True))
            G_mldiv_mu = lstsq(G, mu[:, l])[0]
            tr = ntime - trace(GtWG) + trace(A)
            lndet = slogdet(eyer - GtWG + A)[1]

            lb += -0.5 * inner(G_mldiv_mu, G_mldiv_mu) - 0.5 * tr + 0.5 * lndet

    return lb, ll
