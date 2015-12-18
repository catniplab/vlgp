"""
Functions optimizing hyperparameters
"""
from numpy import identity, inner, arange, outer, trace, empty_like, dstack, diag, clip
from numpy.core.umath import exp, log, sign
from numpy.linalg import slogdet
from numpy.random.mtrand import choice
from scipy.linalg import lstsq, solve, toeplitz
from scipy.optimize import minimize, minimize_scalar


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
    return grad / 2


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


def learngp(obj, latents=None, **kwargs):
    window = kwargs.get('window', 100)
    nseg = kwargs.get('nseg', 10)
    eps = kwargs.get('eps', 1e-6)

    of = kwargs.get('omega_factor', 2)
    ntrial, ntime, nlatent = obj['mu'].shape
    mu = obj['mu'].reshape((-1, nlatent))
    w = obj['w'].reshape((-1, nlatent))
    sigma = obj['sigma'].copy()
    omega = obj['omega'].copy()
    if not (kwargs['learn_sigma'] or kwargs['learn_omega']):
        return sigma, omega
    if latents is None:
        latents = range(nlatent)
    n = ntrial * ntime - window
    start = choice(arange(n), size=nseg)
    win_mu = dstack([mu[i:i + window, :] for i in start])
    win_w = dstack([w[i:i + window, :] for i in start])
    dsq = arange(window) ** 2
    Dsq = toeplitz(dsq)
    for ilatent in latents:
        C = exp(-omega[ilatent] * Dsq) + eps * identity(window)
        K = sigma[ilatent] ** 2 * exp(-omega[ilatent] * Dsq) + eps * identity(window)
        S = dstack([K - K.dot(solve(diag(1 / (eps + win_w[:, ilatent, iseg])) + K, K, sym_pos=True)) for iseg in
                    range(nseg)])
        if kwargs['learn_sigma']:
            tmp = 0.0
            for iseg in range(nseg):
                tmp += win_mu[:, ilatent, iseg].dot(solve(C, win_mu[:, ilatent, iseg], sym_pos=True)) + trace(
                    solve(C, S[:, :, iseg], sym_pos=True))
            sigma[ilatent] = tmp / (window * nseg)
        # M = dstack([outer(win_mu[:, ilatent, iseg], win_mu[:, ilatent, iseg]) for iseg in range(nseg)])
        # mini = minimize(KL, x0=log(omega[ilatent]), jac=KLprime, args=(sigma[ilatent], window, win_mu[:, ilatent, :], M, S, eps))
        if kwargs['learn_omega']:
            mini = minimize_scalar(KL, bounds=(log(omega[ilatent] / of), log(omega[ilatent] * of)),
                                   args=(obj['sigma'][ilatent], window, win_mu[:, ilatent, :], None, S, eps),
                                   method='bounded')
            omega[ilatent] = exp(mini.x)
    return sigma, omega
    # return omega