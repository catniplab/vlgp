__author__ = 'yuan'
import timeit

import numpy as np
from scipy import linalg
from util import makeregressor
from la import ichol_gauss


def firingrate(h, m, v, a, b, min=np.NINF, max=30):
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
        # mdivS = linalg.pinv2(G).dot(m[:, l])
        mdivS = linalg.lstsq(G, m[:, l])[0]
        trace = (T - np.trace(GTWG) + np.trace(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))
        lndet = np.linalg.slogdet(eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]

        lb += -0.5 * np.inner(mdivS, mdivS) - 0.5 * trace + 0.5 * lndet
        # lb += 0.5 * lndet
    return lb


def lbhyper(h, m, a, b, chol, v, sigma, omega):
    L, T, k = chol.shape
    eyek = np.identity(k)
    lam = firingrate(h, m, v, a, b)

    lb = 0.0
    for l in range(L):
        G0 = chol[l, :]
        w = lam.dot(a[l, :] ** 2).reshape((T, 1))
        A0 = G0.T.dot(w * G0)
        G = ichol_gauss(T, omega[l], k) * np.sqrt(sigma[l])
        Gstar = linalg.pinv2(G)
        lb += np.inner(Gstar.dot(m[:, l]), Gstar.dot(m[:, l]))
        diagG = G.diagonal()
        lb += np.sum(np.log(diagG[np.nonzero(diagG)] ** 2))
        lb += np.trace(Gstar.dot(G0).dot(Gstar.dot(G0).T) -
                       Gstar.dot(G0).dot(A0.dot(Gstar.dot(G0).T)) +
                       Gstar.dot(G0).dot(A0).dot(linalg.solve(eyek + A0, A0, sym_pos=True).dot(Gstar.dot(G0).T)))
    return -lb / L / T


def lbhyper2(y, h, m, a, b, chol, v, sigma, omega):
    L, T, k = chol.shape
    eyek = np.identity(k)

    lam = firingrate(h, m, v, a, b)

    lb = 0.0
    for l in range(L):
        G = ichol_gauss(T, omega[l], k) * np.sqrt(sigma[l])
        w = lam.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        GTu = G.T.dot((y - lam).dot(a[l, :]))
        trace = (T - np.trace(GTWG) + np.trace(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))
        lndet = np.linalg.slogdet(eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]
        lb += lndet - trace - np.inner(GTu, GTu)
    return lb


def train(y, p, prior_var, prior_scale, a0=None, b0=None, m0=None, anorm=1.0,
          hyper=False, kchol=10, niter=50, tol=1e-5, verbose=True):
    """
    :param y: (T, N), y trains
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
    :param anorm: norm constraint of alpha
    :param intercept: bool, include intercept term or not
    :param hyper: train hyperparameters or not
    :param control: control params
    :return:
        m: posterior mean
        post_cov: posterior covariance
        beta: coefficient of h
        alpha: coefficient of latent
        a0: initial value of alpha
        b0: initial value of beta
        lbound: array of lower bounds
        elapsed:
        converged:
    """

    #################################################
    def updatev():
        lam = firingrate(h, m, v, alpha, beta)
        for l in range(L):
            G = prior_chol[l, :]
            w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)

    def tryv(m, a, b):
        out = np.empty_like(v)
        lam = firingrate(h, m, v, a, b)
        for l in range(L):
            G = prior_chol[l, :]
            w = lam.dot(a[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            out[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)
        return out

    def makechol():
        for l in range(L):
            prior_chol[l, :] = ichol_gauss(T, prior_scale[l], kchol) * np.sqrt(prior_var[l])

    ###################################################

    fixpostmean = False
    fixalpha = False
    fixbeta = False

    # epsilon
    eps = 2 * np.finfo(np.float).eps

    # dimensions
    T, N = y.shape
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
    y.setflags(write=0)

    # construct makeregressor
    h = makeregressor(y, p, intercept=True)
    h.setflags(write=0)

    # initialize args
    # make alpha copy to avoid changing initial values
    if m0 is None:
        m = np.zeros((T, L), dtype=float)
    else:
        m = m0.copy()

    if a0 is None:
        a0 = np.random.randn(L, N)
        a0 /= linalg.norm(a0) / anorm
    alpha = a0.copy()

    if b0 is None:
        b0 = linalg.lstsq(h, y)[0]
    beta = b0.copy()

    # valid values of parameters from previous iteration
    good_alpha = alpha.copy()
    good_beta = beta.copy()
    good_m = m.copy()
    good_v = v.copy()
    good_var = prior_var.copy()
    good_scale = prior_scale.copy()

    # temporary storage for recovery
    new_beta = beta.copy()
    new_alpha = alpha.copy()
    new_m = m.copy()
    last_var = prior_var.copy()

    step_alpha = np.ones(N)
    step_beta = np.ones(N)
    step_m = np.ones(L)

    direction_w = np.ones(L) * 1.1

    deflation = 0.5
    inflation = 1.5
    thld = 0.5

    # Optimization
    updatev()
    # initialize lower bound
    lbound = np.full(niter, np.finfo(float).min)
    lbound[0] = elbo(y, h, m, alpha, beta, prior_chol, v)
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(it))

        #############
        # posterior #
        #############
        good_elbo = lbound[it - 1]
        if not fixpostmean:
            new_m[:] = m
            for l in range(L):
                lam = firingrate(h, m, v, alpha, beta)
                G = prior_chol[l]
                w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
                GTWG = G.T.dot(w * G)

                u = (y - lam).dot(alpha[l, :])
                grad_m = u - np.dot(linalg.pinv2(G).T, np.dot(linalg.pinv2(G), m[:, l]))

                u2 = G.dot(G.T.dot(u)) - m[:, l]
                delta_m = u2 - G.dot((w * G).T.dot(u2)) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (w * G).T.dot(u2),
                                                                                      sym_pos=True)))
                new_m[:, l] = m[:, l] + step_m[l] * delta_m
                new_m[:, l] -= np.mean(new_m[:, l])
                lb = elbo(y, h, new_m, alpha, beta, prior_chol, tryv(new_m, alpha, beta))
                predicted = thld * np.inner(grad_m, delta_m)
                if np.isnan(lb) or lb < good_elbo:
                    step_m[l] *= deflation
                    step_m[l] += eps
                else:
                    if lb - good_elbo > predicted:
                        step_m[l] *= inflation
                    good_elbo = lb
                    m[:, l] = new_m[:, l]
                updatev()

        ###########
        # weights #
        ###########
        if not fixalpha:
            new_alpha[:] = alpha
            for l in range(L):
                lam = firingrate(h, m, v, alpha, beta)
                grad_a = (y - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * alpha[l, :]
                # neg_hess_a = np.diag(lam.T.dot(m[:, l] ** 2) +
                #                      2 * lam.T.dot(m[:, l] * v[:, l]) * alpha[l, :] +
                #                      lam.T.dot(v[:, l] ** 2) * alpha[l, :] ** 2 +
                #                      lam.T.dot(v[:, l]))
                diag_neg_hess_a = lam.T.dot(m[:, l] ** 2) + 2 * lam.T.dot(m[:, l] * v[:, l]) * alpha[l, :] + lam.T.dot(
                    v[:, l] ** 2) * alpha[l, :] ** 2 + lam.T.dot(v[:, l])
                if linalg.norm(grad_a, ord=np.inf) < eps:
                    break
                try:
                    # delta_a = step_alpha[l] * linalg.solve(neg_hess_a, grad_a, sym_pos=True)
                    delta_a = step_alpha[l] * (grad_a / diag_neg_hess_a)
                except linalg.LinAlgError as e:
                    print('alpha', e)
                    continue
                new_alpha[l, :] = alpha[l, :] + delta_a
                new_alpha[l, :] /= linalg.norm(new_alpha[l, :]) / anorm
                lb = elbo(y, h, m, new_alpha, beta, prior_chol, tryv(m, new_alpha, beta))
                predicted = thld * np.inner(grad_a, delta_a)
                if np.isnan(lb) or lb < good_elbo:
                    step_alpha[l] *= deflation
                    step_alpha[l] += eps
                else:
                    if lb - good_elbo > predicted:
                        step_alpha[l] *= inflation
                    good_elbo = lb
                    alpha[l, :] = new_alpha[l, :]
                updatev()

        if not fixbeta:
            new_beta[:] = beta
            for n in range(N):
                lam = firingrate(h, m, v, alpha, beta)
                grad_b = h.T.dot(y[:, n] - lam[:, n])
                neg_hess_b = h.T.dot((h.T * lam[:, n]).T)
                if linalg.norm(grad_b, ord=np.inf) < eps:
                    break
                try:
                    delta_b = step_beta[n] * linalg.solve(neg_hess_b, grad_b, sym_pos=True)
                except linalg.LinAlgError as e:
                    print('beta', e)
                    continue
                elbo(y, h, m, alpha, beta, prior_chol, v)
                new_beta[:, n] = beta[:, n] + delta_b
                lb = elbo(y, h, m, alpha, new_beta, prior_chol, tryv(m, alpha, new_beta))
                predicted = thld * np.inner(grad_b, delta_b)
                if np.isnan(lb) or lb < good_elbo:
                    # Decrease the stepsize if the lower bound decreases.
                    # Add a small positive number to prevent becoming 0.
                    step_beta[n] *= deflation
                    step_beta[n] += eps
                else:
                    if lb - good_elbo > predicted:
                        # Increase the stepsize if the real increment is more than expected.
                        step_beta[n] *= inflation
                    good_elbo = lb
                beta[:, n] = new_beta[:, n]
                updatev()

        ###############
        # hyperparams #
        ###############
        if hyper and it % 10 == 0:
            # low = prior_scale.min()
            # high = prior_scale.max()
            # grid_w = np.array(list(itertools.combinations_with_replacement(np.logspace(low, high, 5), 2)))
            #
            # lb = np.empty(grid_w.shape[0], dtype=float)
            # for i, row in enumerate(grid_w):
            #     new_chol = np.empty_like(prior_chol)
            #     for l in range(L):
            #         new_chol[l, :] = ichol_gauss(T, row[l], kchol) * np.sqrt(prior_var[l])
            #     lb[i] = elbo(y, h, m, alpha, beta, new_chol, v)
            #     # lb[i] = lbhyper(h, m, alpha, beta, prior_chol, v, prior_var, row)
            #     # print(row, lb[i])
            #     # lb[i] = lbhyper2(y, h, m, alpha, beta, prior_chol, v, prior_var, row)
            #     # print(row, lb[i])
            #
            # prior_scale[:] = grid_w[lb.argmax(), :]
            # print('scale ->', prior_scale)
            # makechol()
            # updatev()

            lam = firingrate(h, m, v, alpha, beta)
            for l in range(L):
                last_var[l] = prior_var[l]
                # G = prior_chol[l, :]
                # w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
                # GTG = G.T.dot(G)
                # GTWG = G.T.dot(w * G)
                # prior_var[l] = np.inner(m[:, l], m[:, l]) + \
                #                np.trace(
                #                    GTG - GTG.dot(GTWG) + GTG.dot(GTWG).dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))

                G = prior_chol[l]
                w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
                GTWG = np.dot(G.T, w * G)
                # mdivS = linalg.pinv2(G).dot(m[:, l])
                mdivS = linalg.lstsq(G, m[:, l])[0]
                new_var = prior_var[l] * (np.inner(mdivS, mdivS) + (T - np.trace(GTWG) + np.trace(GTWG.dot(
                    linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))) / T
                prior_var[l] = new_var if new_var > 0 else prior_var[l] / 2
                if verbose:
                    print('prior variance[{:d}]: {:.5f} -> {:.5f}'.format(l, last_var[l], prior_var[l]))
            makechol()
            updatev()

        # store lower bound
        lbound[it] = elbo(y, h, m, alpha, beta, prior_chol, v)

        # check convergence
        chg_alpha = 0.0 if fixalpha else np.max(np.abs(good_alpha - alpha))
        chg_beta = 0.0 if fixbeta else np.max(np.abs(good_beta - beta))
        chg_post_mean = 0.0 if fixpostmean else np.max(np.abs(good_m - m))
        chg_variance = np.max(np.abs(good_var - prior_var)) if hyper else 0.0
        chg_scale = np.max(np.abs(good_scale - prior_scale)) if hyper else 0.0

        # converged if the change in ELBO is relatively smaller than a tolerance
        # if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]):
        #     converged = True

        if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]) and it % 5 == 0:
            converged = True

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'
                  'change in alpha = {:.10f}\n'
                  'change in beta = {:.10f}\n'
                  'change in posterior mean = {:.10f}\n'
                  'change in prior variance = {:.10f}\n'
                  'change in prior scale = {:.10f}'.format(lbound[it], lbound[it] - lbound[it - 1],
                                                           timeit.default_timer() - iter_start,
                                                           chg_alpha, chg_beta, chg_post_mean,
                                                           chg_variance, chg_scale))

        # store current iteration
        good_alpha[:] = alpha
        good_beta[:] = beta
        good_m[:] = m
        good_var[:] = prior_var
        good_scale[:] = prior_scale

        it += 1

    stop = timeit.default_timer()
    return lbound[:it], m, alpha, beta, prior_var, prior_scale, a0, b0, stop - start, converged
