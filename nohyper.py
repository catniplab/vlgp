import timeit

import numpy as np
from scipy import linalg
from util import makeregressor
from la import ichol_gauss


def firingrate(h, m, v, a, b, min=0, max=30):
    eta_x = h.dot(b) + m.dot(a) + 0.5 * v.dot(a ** 2)
    np.clip(eta_x, min, max, out=eta_x)
    return np.exp(eta_x)


def elbo(y, h, m, a, b, chol, v):
    """
    Evidence Lower Bound
    :param y: spike trains
    :param h: regressors
    :param m: posterior mean
    :param a: alpha
    :param b: beta
    :param chol: incomplete cholesky decomposition of prior covariance
    :param v: temporal slices of posterior variance
    :return: lower bound
    """
    # T, N = y.shape
    L, T, k = chol.shape
    eyek = np.identity(k)

    eta = h.dot(b) + m.dot(a)
    lam = firingrate(h, m, v, a, b)
    lb = np.sum(y * eta - lam)

    for l in range(L):
        G = chol[l, :]
        w = lam.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        mdivS = linalg.pinv2(G).dot(m[:, l])
        trace = (T - np.trace(GTWG) + np.trace(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))
        lndet = np.linalg.slogdet(eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]

        lb += -0.5 * np.inner(mdivS, mdivS) - 0.5 * trace + 0.5 * lndet
        # lb += 0.5 * lndet
    return lb


def optbeta(y, h, m, a, b, v, chol, step):
    failed = np.full_like(step, fill_value=False, dtype=bool)
    _, N = b.shape
    out = np.empty_like(b)
    for n in range(N):
        lam = firingrate(h, m, v, a, b)
        grad_b = h.T.dot(y[:, n] - lam[:, n])
        neg_hess_b = h.T.dot((h.T * lam[:, n]).T)
        if linalg.norm(grad_b, ord=np.inf) < np.finfo(float).eps * 10:
            delta_b = 0.0
        else:
            try:
                delta_b = linalg.solve(neg_hess_b + np.diag(step[n] * neg_hess_b.diagonal()), grad_b, sym_pos=True)
            except linalg.LinAlgError as e:
                print('beta', e)
                delta_b = 0.0
        out[:, n] = b[:, n] + step[n] * delta_b
        lb_old = elbo(y, h, m, a, b, chol, v)
        lb_new = elbo(y, h, m, a, out, chol, v)
        if np.isnan(lb_new) or lb_new < lb_old:
            out[:, n] = b[:, n]
            failed[n] = True
    return out, failed


def optalpha(y, h, m, a, b, v, chol, step):
    failed = np.full_like(step, fill_value=False, dtype=bool)
    L, N = a.shape
    out = np.empty_like(a)
    lam = firingrate(h, m, v, a, b)
    for l in range(L):
        grad_a = (y - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * a[l, :]
        diag_neg_hess_a = lam.T.dot(m[:, l] ** 2) + 2 * lam.T.dot(m[:, l] * v[:, l]) * a[l, :] + \
                          lam.T.dot(v[:, l] ** 2) * a[l, :] ** 2 + lam.T.dot(v[:, l])
        if linalg.norm(grad_a, ord=np.inf) < np.finfo(float).tiny:
            delta_a = 0.0
        else:
            try:
                delta_a = grad_a / diag_neg_hess_a
            except linalg.LinAlgError as e:
                print('alpha', e)
                delta_a = 0.0
        out[l, :] = a[l, :] + delta_a
        out[l, :] /= linalg.norm(out[l, :], ord=2) / linalg.norm(a[l, :], ord=2)
        lb_old = elbo(y, h, m, a, b, chol, v)
        lb_new = elbo(y, h, m, out, b, chol, v)
        if np.isnan(lb_new) or lb_new < lb_old:
            # out[l, :] = a[l, :]
            failed[l] = True
    return out, failed


def optmean(y, h, m, a, b, v, chol, step=1.0):
    failed = np.full_like(step, fill_value=False, dtype=bool)
    L, T, k = chol.shape
    eyek = np.identity(k)
    out = np.empty_like(m)
    lam = firingrate(h, m, v, a, b)
    for l in range(L):
        G = chol[l]
        w = lam.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        u = (y - lam).dot(a[l, :])
        u2 = G.dot(G.T.dot(u)) - m[:, l]
        GTWu2 = (w * G).T.dot(u2)
        delta_m = u2 - G.dot(GTWu2) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWu2, sym_pos=True)))
        out[:, l] = m[:, l] + step[l] * delta_m
        out[:, l] -= np.mean(out[:, l])
        lb_old = elbo(y, h, m, a, b, chol, v)
        lb_new = elbo(y, h, out, a, b, chol, v)
        if np.isnan(lb_new) or lb_new < lb_old:
            out[:, l] = m[:, l]
            failed[l] = True
    return out, failed


def trainmodel(spike, p, prior_var, prior_scale, a0=None, b0=None, m0=None, normofalpha=1.0, kchol=9, fpinter=5,
               niter=50, tol=1e-5, verbose=False):
    """
    :param spike: (T, N), spike trains
    :param p: order of regression
    :param prior_mean: (T, L), prior mean
    :param prior_var: (L,), prior variance
    :param prior_scale: (L,), prior inverse of squared lengthscale
    :param a0: (L, N), initial value of alpha
    :param b0: (N, intercept + p * N), initial value of beta
    :param m0: (T, L), initial value of posterior mean
    :param fixalpha: bool, switch of not train alpha
    :param fixbeta: bool, switch of not train beta
    :param fixpostmean: bool, switch of not train posterior mean
    :param normofalpha: norm constraint of alpha
    :param intercept: bool, include intercept term or not
    :param hyper: train hyperparameters or not
    :param control: control params
    :return:
        post_mean: posterior mean
        post_cov: posterior covariance
        beta: coefficient of regressor
        alpha: coefficient of latent
        a0: initial value of alpha
        b0: initial value of beta
        lbound: array of lower bounds
        elapsed:
        converged:
    """

    #################################################
    def updatev():
        lam = firingrate(regressor, post_mean, v, alpha, beta)
        for l in range(L):
            G = prior_chol[l, :]
            w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)

    def makechol():
        for l in range(L):
            prior_chol[l, :] = ichol_gauss(T, prior_scale[l], kchol) * np.sqrt(prior_var[l])

    ###################################################

    # dimensions
    T, N = spike.shape
    L = len(prior_var)

    eyek = np.identity(kchol)

    # hyperparameters
    prior_var = prior_var.copy()
    prior_scale = prior_scale.copy()

    # incomplete cholesky decomposition of prior covariance matrix
    prior_chol = np.empty((L, T, kchol), dtype=float)
    makechol()

    # temporal slice of v
    v = np.ones((T, L)) * prior_var

    # read-only variables, protection from unexpected assignment
    spike.setflags(write=0)

    # construct makeregressor
    regressor = makeregressor(spike, p, intercept=True)
    regressor.setflags(write=0)

    # initialize args
    # make alpha copy to avoid changing initial values
    post_mean = m0.copy()
    alpha = a0.copy()

    if b0 is None:
        b0 = linalg.lstsq(regressor, spike)[0]
    beta = b0.copy()

    # valid values of parameters from previous iteration
    good_alpha = alpha.copy()
    good_beta = beta.copy()
    good_post_mean = post_mean.copy()

    stepbeta = np.ones(N, dtype=float)
    stepalpha = np.ones(L, dtype=float)
    stepmean = np.ones(L, dtype=float) * 0.1

    # Optimization
    updatev()
    # initialize lower bound
    lbound = np.full(niter, np.finfo(float).min)
    lbound[0] = elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(it))

        beta[:], failed = optbeta(spike, regressor, post_mean, alpha, beta, v, prior_chol, step=stepbeta)
        stepbeta[failed] *= 2
        stepbeta[np.logical_not(failed)] *= 0.5
        stepbeta.clip(np.finfo(float).eps, 1 / np.finfo(float).eps, out=stepbeta)
        updatev()
        alpha[:], failed = optalpha(spike, regressor, post_mean, alpha, beta, v, prior_chol, step=stepalpha)
        stepalpha[failed] *= 0.5
        stepalpha[np.logical_not(failed)] *= 1.5
        stepalpha.clip(np.finfo(float).eps, 1 / np.finfo(float).eps, out=stepalpha)
        updatev()
        post_mean[:], failed = optmean(spike, regressor, post_mean, alpha, beta, v, prior_chol, step=stepmean)
        stepmean[failed] *= 0.1
        stepmean[np.logical_not(failed)] *= 1.1
        stepmean.clip(np.finfo(float).eps, 1 / np.finfo(float).eps, out=stepmean)
        updatev()

        # store lower bound
        lbound[it] = elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)

        # check convergence
        chg_alpha = np.max(np.abs(good_alpha - alpha))
        chg_beta = np.max(np.abs(good_beta - beta))
        chg_post_mean = np.max(np.abs(good_post_mean - post_mean))

        # converged if the change in ELBO is relatively smaller than a tolerance
        citer = 5
        if np.abs(lbound[it] - lbound[it - citer]) < tol * np.abs(lbound[it - citer]) and it % citer == 0:
            converged = True

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'
                  'change in alpha = {:.10f}\n'
                  'change in beta = {:.10f}\n'
                  'change in posterior mean = {:.10f}'.format(lbound[it], lbound[it] - lbound[it - 1],
                                                              timeit.default_timer() - iter_start,
                                                              chg_alpha, chg_beta, chg_post_mean))

        # store current iteration
        good_alpha[:] = alpha
        good_beta[:] = beta
        good_post_mean[:] = post_mean

        it += 1

    stop = timeit.default_timer()

    return lbound[:it], post_mean, alpha, beta, prior_var, prior_scale, a0, b0, stop - start, converged
