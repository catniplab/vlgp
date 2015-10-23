import timeit
import numpy as np
from scipy import linalg
from util import makeregressor


def effectivemean():
    return None


def elbo(poisobs, gaussobs, h, m, v, a, b, chol):
    return 0


def train(poisobs, gaussobs, p, chol, m0=None, apois0=None, agauss0=None, bpois0=None, bgauss0=None, niter=50, tol=1e-5,
          verbose=True):
    L, T, cholrank = chol.shape
    npois = poisobs.shape[1]
    ngauss = gaussobs.shape[1]

    # h = selfhistory(poisobs, gaussobs, p)

    apois = apois0.copy()
    agauss = agauss0.copy()
    bpois = bpois0.copy()
    bgauss = bgauss0.copy()
    m = m0.copy()

    good_m = m.copy()
    good_apois = apois.copy()
    good_agauss = agauss.copy()
    good_bpois = bpois.copy()
    good_bgauss = bgauss.copy()

    v = np.empty_like(m, dtype=float)

    lb = np.full(niter, fill_value=np.finfo(float).min, dtype=float)
    lb[0] = 0

    converged = False
    i = 1
    start = timeit.default_timer()
    while not converged and i < niter:
        # estimate b
        # for n in range(npois):
        #     bpois[n] += delta
        #
        # for n in range(ngauss):
        #     bgauss[n] += delta

        # update v

        # estimate latent
        # for l in range(L):
        #     grad_m =
        #     m[l, :] += V dot grad_m
        #
        #     a[:, l] += grad_a / accu_grad + neghess

        # update v

        # estimate gaussian variance
        # for n in range(ngauss):
        #     sigma2[n] = np.mean((gaussobs - mean) ** 2)

        # lb[i] = elbo()
        if np.abs(lb[i] - lb[i - 1]) < tol * np.abs(lb[i - 1]):
            converged = True

        good_m[:] = m
        good_apois[:] = apois
        good_agauss[:] = agauss
        good_bpois[:] = bpois
        good_bgauss[:] = bgauss

        i += 1

    stop = timeit.default_timer()
    return lb, m, v, apois, agauss, bpois, bgauss, stop - start, converged
