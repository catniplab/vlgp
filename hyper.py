"""
Functions optimizing hyperparameters
"""
from numpy import identity, inner, arange, outer, trace, empty_like, dstack, diag, clip
from numpy.core.umath import exp, log, sign
from numpy.linalg import slogdet
from numpy.random.mtrand import choice
from scipy.linalg import lstsq, solve, toeplitz
from scipy.optimize import minimize, minimize_scalar

from mathf import ichol_gauss
from util import sqexpcov


# def kl(theta, sigma, mu, L):
#     """Evidence lower bound for hyperparameter
#     Args:
#         theta:
#         model:
#
#     Returns:
#
#     """
#     omega = exp(theta)
#     ntrial, ntime = mu.shape
#     rank = L.shape[-1]
#     # K = sqexpcov(ntime, omega[ilatent], sigma[ilatent] ** 2)
#     # S = L[ilatent, :].dot(L[ilatent, :].T)
#     # K_mldiv_S = lstsq(K, S)[0]
#     G = ichol_gauss(ntime, omega, rank) * sigma
#     f = 0.0
#     for itrial in range(ntrial):
#         G_mldiv_mu = lstsq(G, mu[itrial, :])[0]
#         G_mldiv_L = lstsq(G, L[itrial, :])[0]
#         K_inv_S = G_mldiv_L.T.dot(G_mldiv_L)
#         f += 0.5 * (inner(G_mldiv_mu, G_mldiv_mu) + trace(K_inv_S) - slogdet(K_inv_S)[1])
#     return f
#
#
# def klprime(theta, sigma, mu, L):
#     """Gradient of ELBO wrt hyperparameter
#
#     Args:
#         omega:
#         model:
#
#     Returns:
#
#     """
#     omega = exp(theta)
#     ntrial, ntime = mu.shape
#     rank = L.shape[-1]
#     K = sqexpcov(ntime, omega, sigma ** 2) + 1e-6 * identity(ntime)
#     dK = Kprime(sigma, theta, ntime)
#     KinvdK = solve(K, dK, sym_pos=True)
#     # logdetK = slogdet(K)[1]
#     df = 0.0
#     for itrial in range(ntrial):
#         S = L[itrial, :].dot(L[itrial, :].T)
#         KinvS = solve(K, S, sym_pos=True)
#         df += 0.5 * trace(
#             (-solve(K, outer(mu[itrial, :], mu[itrial, :]), sym_pos=True) - KinvS + identity(ntime)).dot(KinvdK))
#     return df
#
#
# def learn_hyper(model):
#     theta = empty_like(model['omega'])
#     nlatent = theta.shape[0]
#     for ilatent in range(nlatent):
#         theta[ilatent] = minimize(kl, x0=theta[ilatent], method='CG', args=(
#         model['sigma'][ilatent], model['mu'][:, :, ilatent], model['L'][:, ilatent, :])).x
#     return exp(theta)
#
#
# def Kprime(theta, sigma, n):
#     difsq = toeplitz(arange(n) ** 2)
#     omega = exp(theta)
#     sigmasq = sigma ** 2
#     return -sigmasq * exp(-omega * difsq) * omega * difsq


def KLprime(theta, sigma, n, mu, M, S, eps=1e-6):
    omega = exp(theta)
    sigmasq = sigma ** 2
    nseg = M.shape[-1]
    dsq = arange(n) ** 2
    Dsq = toeplitz(dsq)
    k = sigmasq * exp(-omega * dsq)
    K = toeplitz(k) + eps * identity(n)
    KinvKD = solve(K, K * Dsq * omega, sym_pos=True)
    grad = 0.0
    for iseg in range(nseg):
        KinvS = solve(K, S[:, :, iseg], sym_pos=True)
        KinvM = solve(K, M[:, :, iseg], sym_pos=True)
        # grad += trace(solve(KinvM + KinvS - identity(n), KinvKD, sym_pos=True))
        grad += trace(lstsq(KinvM + KinvS - identity(n), KinvKD)[0])
    return clip(grad / 2, -1, 1)


def KL(theta, sigma, n, mu, M, S, eps=1e-6):
    omega = exp(theta)
    sigmasq = sigma ** 2
    nseg = mu.shape[-1]
    dsq = arange(n) ** 2
    k = sigmasq * exp(-omega * dsq)
    K = toeplitz(k) + eps * identity(n)
    div = 0.0
    for iseg in range(nseg):
        KinvS = solve(K, S[:, :, iseg], sym_pos=True)
        muKinvmu = mu[:, iseg].dot(solve(K, mu[:, iseg], sym_pos=True))
        div += muKinvmu + trace(KinvS) - slogdet(KinvS)[1]
    return div / 2


def learngp(obj, window=100, nseg=10, eps=1e-6):
    ntrial, ntime, nlatent = obj['mu'].shape
    mu = obj['mu'].reshape((-1, nlatent))
    w = obj['w'].reshape((-1, nlatent))
    sigma = obj['sigma'].copy()
    omega = obj['omega'].copy()
    n = ntrial * ntime - window
    start = choice(arange(n), size=nseg)
    win_mu = dstack([mu[i:i + window, :] for i in start])
    win_w = dstack([w[i:i + window, :] for i in start])
    dsq = arange(window) ** 2
    Dsq = toeplitz(dsq)
    for ilatent in range(nlatent):
        # C = exp(-omega[ilatent] * Dsq) + eps * identity(window)
        K = sigma[ilatent] ** 2 * exp(-omega[ilatent] * Dsq) + eps * identity(window)
        S = dstack([K - K.dot(solve(diag(1 / (eps + win_w[:, ilatent, iseg])) + K, K, sym_pos=True)) for iseg in range(nseg)])
        # tmp = 0.0
        # for iseg in range(nseg):
        #     tmp += win_mu[:, ilatent, iseg].dot(solve(C, win_mu[:, ilatent, iseg], sym_pos=True)) + trace(solve(C, S[:, :, iseg], sym_pos=True))
        # sigma[ilatent] = tmp / window
        # M = dstack([outer(win_mu[:, ilatent, iseg], win_mu[:, ilatent, iseg]) for iseg in range(nseg)])
        # mini = minimize(KL, x0=log(omega[ilatent]), jac=KLprime, args=(sigma[ilatent], window, win_mu[:, ilatent, :], M, S))
        mini = minimize_scalar(KL, bounds=(log(omega[ilatent]/10), log(omega[ilatent]*2)), args=(obj['sigma'][ilatent], window, win_mu[:, ilatent, :], None, S), method='bounded')
        omega[ilatent] = exp(mini.x)
    # return sigma, omega
    return omega