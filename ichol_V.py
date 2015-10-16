import timeit
import itertools
import numpy as np
from scipy import linalg
from util import makeregressor
from la import *


def firingrate(h, m, lv, a, b, lb=-30, ub=30):
    # lv: (L, T, T)
    # v: (T, L)
    v = np.sum(lv ** 2, axis=2).T
    eta_x = h.dot(b) + m.dot(a) + 0.5 * v.dot(a ** 2)
    np.clip(eta_x, lb, ub, out=eta_x)
    return np.exp(eta_x)


def elbo(y, h, m, a, b, chol, lv):
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
    lam = firingrate(h, m, lv, a, b)
    lb = np.sum(y * eta - lam)

    for l in range(L):
        G = chol[l, :]
        w = lam.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        B = GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))
        # mdivS = linalg.pinv2(G).dot(m[:, l])
        mdivS = linalg.lstsq(G, m[:, l])[0]
        trace = (T - np.trace(GTWG) + np.trace(B))
        lndet = np.linalg.slogdet(eyek - GTWG + B)[1]

        lb += -0.5 * np.inner(mdivS, mdivS) - 0.5 * trace + 0.5 * lndet
        # lb += 0.5 * lndet
    return lb


def train(y, p, prior_var, prior_scale, a0=None, b0=None, m0=None, anorm=1.0, hyper=False, kchol=10, niter=50, tol=1e-5,
          verbose=True):
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

    # TODO: replace eyek - * with numpy.fill_diagonal(*, 1 - *.diagonal())

    #################################################
    def updatev():
        for l in range(L):
            lam = firingrate(h, m, lv, alpha, beta)
            G = chol[l]
            w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            w.clip(np.finfo(float).tiny, np.exp(15), out=w)  # avoid zeros and infinities
            GTWG = G.T.dot(w * G)
            A = eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))  # A should be pd but numerically not
            # A = eyek - GTWG.dot(GTWG + GTWG.dot(GTWG)).dot(GTWG)  # wrong
            # lv[l, :] = G.dot(ichol2(A))
            eigval, eigvec = linalg.eigh(A)
            eigval.clip(0, np.PINF, out=eigval)  # remove negative eigenvalues
            lv[l, :] = G.dot(eigvec.dot(np.diag(np.sqrt(eigval))))

    def makechol():
        for l in range(L):
            chol[l, :] = ichol_gauss(T, prior_scale[l], kchol) * np.sqrt(prior_var[l])
    ###################################################

    # dimensions
    T, N = y.shape
    L = prior_var.shape[0]

    eyek = np.identity(kchol)

    # hyperparameters
    prior_var = prior_var.copy()
    prior_scale = prior_scale.copy()

    # incomplete cholesky decomposition of prior covariance matrix
    chol = np.empty((L, T, kchol), dtype=float)
    makechol()

    lv = chol.copy()

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
    good_var = prior_var.copy()
    good_scale = prior_scale.copy()

    # temporary storage for recovery
    last_var = prior_var.copy()

    step_alpha = np.ones(N)
    step_beta = np.ones(N)
    step_m = np.ones(L)

    down = 0.5
    up = 1.5
    thld = 0.5

    # Optimization
    # initialize lower bound
    lbound = np.full(niter, np.finfo(float).min)
    lbound[0] = elbo(y, h, m, alpha, beta, chol, lv)
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(it))

        good_elbo = lbound[it - 1]

        #############
        # posterior #
        #############
        lam = firingrate(h, m, lv, alpha, beta)
        for l in range(L):
            G = chol[l]
            w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)

            u = (y - lam).dot(alpha[l, :])
            # grad_m = u - np.dot(linalg.pinv2(G).T, np.dot(linalg.pinv2(G), m[:, l]))
            grad_m = u - linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]
            # delta_m = lv[l, :].dot(lv[l, :].T.dot(grad_m))

            u2 = G.dot(G.T.dot(u)) - m[:, l]
            tmp = (w * G).T.dot(u2)
            delta_m = u2 - G.dot(tmp) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, tmp, sym_pos=True)))
            # delta_m = u2 - G.dot((w * G).T.dot(u2)) + G.dot(GTWG.dot(linalg.lstsq(eyek + GTWG, (w * G).T.dot(u2))[0]))
            m[:, l] += step_m[l] * delta_m
            # m[:, l] -= np.mean(m[:, l])
            lb = elbo(y, h, m, alpha, beta, chol, lv)
            predicted = thld * np.inner(grad_m, delta_m)
            if np.isfinite(lb) and lb > good_elbo:
                if lb - good_elbo > predicted:
                    step_m[l] *= up
                good_elbo = lb
                m[:, l] -= np.mean(m[:, l])
            else:
                m[:, l] = good_m[:, l]
                step_m[l] *= down
        step_m.clip(10 * np.finfo(float).eps, np.finfo(float).max, out=step_m)
        updatev()

        ###########
        # weights #
        ###########
        v = np.sum(lv ** 2, axis=-1).T  # G .* G
        lam = firingrate(h, m, lv, alpha, beta)
        for l in range(L):
            grad_a = (y - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * alpha[l, :]
            # neg_hess_a = np.diag(lam.T.dot(m[:, l] ** 2) +
            #                      2 * lam.T.dot(m[:, l] * v[:, l]) * alpha[l, :] +
            #                      lam.T.dot(v[:, l] ** 2) * alpha[l, :] ** 2 +
            #                      lam.T.dot(v[:, l]))
            diag_neg_hess_a = lam.T.dot(m[:, l] ** 2) + 2 * lam.T.dot(m[:, l] * v[:, l]) * alpha[l, :] + lam.T.dot(
                v[:, l] ** 2) * alpha[l, :] ** 2 + lam.T.dot(v[:, l])
            # if linalg.norm(grad_a, ord=np.inf) < eps:
            #     break
            # delta_a = step_alpha[l] * linalg.solve(neg_hess_a, grad_a, sym_pos=True)
            delta_a = grad_a / diag_neg_hess_a
            alpha[l, :] += step_alpha[l] * delta_a
            alpha[l, :] /= linalg.norm(alpha[l, :]) / anorm
            # lb = elbo(y, h, m, alpha, beta, chol, lv)
            # predicted = thld * np.inner(grad_a, delta_a)
            # if np.isnan(lb) or lb < good_elbo:
            #     alpha[l, :] = good_alpha[l, :]
            #     step_alpha[l] *= down
            # else:
            #     if lb - good_elbo > predicted:
            #         step_alpha[l] *= up
            #     good_elbo = lb
        # step_alpha.clip(10 * np.finfo(float).eps, np.finfo(float).max, out=step_alpha)
        updatev()

        lam = firingrate(h, m, lv, alpha, beta)
        for n in range(N):
            # lam = firingrate(h, m, lv, alpha, beta)
            grad_b = h.T.dot(y[:, n] - lam[:, n])
            neg_hess_b = h.T.dot((h.T * lam[:, n]).T)
            # if linalg.norm(grad_b, ord=np.inf) < eps:
            #     break
            try:
                # delta_b = linalg.solve(neg_hess_b, grad_b, sym_pos=True)
                delta_b = linalg.lstsq(neg_hess_b, grad_b)[0]
            except linalg.LinAlgError as e:
                print('beta', e)
                continue
            beta[:, n] += step_beta[n] * delta_b
            # lb = elbo(y, h, m, alpha, beta, chol, lv)
            # predicted = thld * np.inner(grad_b, delta_b)
            # if np.isnan(lb) or lb < good_elbo:
                # Decrease the stepsize if the lower bound decreases.
                # Add a small positive number to prevent becoming 0.
                # beta[:, n] = good_beta[:, n]
                # step_beta[n] *= down
            # else:
            #     if lb - good_elbo > predicted:
                    # Increase the stepsize if the real increment is more than expected.
                    # step_beta[n] *= up
                # good_elbo = lb
        # step_beta.clip(10 * np.finfo(float).eps, np.finfo(float).max, out=step_beta)
        updatev()

        ###############
        # hyperparams #
        ###############
        # if hyper and it % 10 == 0:
            # low = -5
            # high = 5
            # grid_w = np.array(list(itertools.combinations_with_replacement(np.logspace(low, high, 5), 2)))
            #
            # lb = np.empty(grid_w.shape[0], dtype=float)
            # for i, row in enumerate(grid_w):
            #     new_chol = np.empty_like(chol)
            #     for l in range(L):
            #         new_chol[l, :] = ichol_gauss(T, row[l], kchol) * np.sqrt(prior_var[l])
            #     lb[i] = elbo(y, h, m, alpha, beta, new_chol, lv)
            #     # lb[i] = lbhyper(h, m, alpha, beta, chol, v, prior_var, row)
            #     # print(row, lb[i])
            #     # lb[i] = lbhyper2(y, h, m, alpha, beta, chol, v, prior_var, row)
            #     # print(row, lb[i])
            #
            # prior_scale[:] = grid_w[lb.argmax(), :]
            # print('scale ->', prior_scale)
            # makechol()
            # updatev()

            # lam = firingrate(h, m, lv, alpha, beta)
            # for l in range(L):
            #     last_var[l] = prior_var[l]
            #     # G = chol[l, :]
            #     # w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            #     # GTG = G.T.dot(G)
            #     # GTWG = G.T.dot(w * G)
            #     # prior_var[l] = np.inner(m[:, l], m[:, l]) + \
            #     #                np.trace(
            #     #                    GTG - GTG.dot(GTWG) + GTG.dot(GTWG).dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))
            #
            #     G = chol[l]
            #     w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            #     GTWG = np.dot(G.T, w * G)
            #     # mdivS = linalg.pinv2(G).dot(m[:, l])
            #     mdivS = linalg.lstsq(G, m[:, l])[0]
            #     new_var = prior_var[l] * (np.inner(mdivS, mdivS) + (T - np.trace(GTWG) + np.trace(GTWG.dot(
            #         linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))) / T
            #     prior_var[l] = new_var if new_var > 0 else prior_var[l] / 2
            #     if verbose:
            #         print('prior variance[{:d}]: {:.5f} -> {:.5f}'.format(l, last_var[l], prior_var[l]))
            # makechol()
            # updatev()

        # store lower bound
        lbound[it] = elbo(y, h, m, alpha, beta, chol, lv)

        # check convergence
        chg_alpha = np.max(np.abs(good_alpha - alpha))
        chg_beta = np.max(np.abs(good_beta - beta))
        chg_post_mean = np.max(np.abs(good_m - m))
        chg_variance = np.max(np.abs(good_var - prior_var)) if hyper else 0.0
        chg_scale = np.max(np.abs(good_scale - prior_scale)) if hyper else 0.0

        # converged if the change in ELBO is relatively smaller than a tolerance
        if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]):
            converged = True

        # if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]) and it % 5 == 0:
        #     converged = True

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
