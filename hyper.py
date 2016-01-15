"""
This file contains the functions of hyperparameter optimization
"""
from numpy import identity, arange, trace, dstack, diag
from numpy.core.umath import exp, log, sqrt
from numpy.linalg import slogdet
from numpy.random.mtrand import choice
from scipy.linalg import lstsq, solve, toeplitz
from scipy.optimize import minimize_scalar


def KLprime(theta, sigma, n, mu, M, S, eps=1e-6):
    """Derivative of KL part of ELBO w.r.t. hyperparameter
    Args:
        theta: log of timescale
        sigma: variance
        n: length of time
        mu: posterior mean
        M: precalculated matrix involving mu
        S: correlation matrix
        eps: minimum positive value

    Returns:
        gradient
    """
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
    """KL part of ELBO
    Args:
        theta: log of timescale
        sigma: variance
        n: length of time
        mu: posterior mean
        M: precalculated matrix involving mu
        S: correlation matrix
        eps: minimum positive value

    Returns:
        function value
    """
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
        div += muKinvmu + trace(KinvS) - slogdet(KinvS)[1] - n
    return div / 2


def learngp(obj, latents=None, **kwargs):
    """Main function learning hyperparameters
    Args:
        obj: inference object
        latents: optional true latent
        **kwargs: optional arguments controlling inference

    Returns:
        optimized hyperparameters
    """
    window = kwargs.get('window', 100)
    nseg = kwargs.get('nseg', 10)
    eps = kwargs.get('eps', 1e-6)

    of = kwargs.get('omega_factor', 5)
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
            sigma[ilatent] = sqrt(tmp / (window * nseg))
        # M = dstack([outer(win_mu[:, ilatent, iseg], win_mu[:, ilatent, iseg]) for iseg in range(nseg)])
        # mini = minimize(KL, x0=log(omega[ilatent]), jac=KLprime, args=(sigma[ilatent], window, win_mu[:, ilatent, :],
        # M, S, eps))
        if kwargs['learn_omega']:
            mini = minimize_scalar(KL, bounds=(log(omega[ilatent] / of), log(omega[ilatent] * of)),
                                   args=(obj['sigma'][ilatent], window, win_mu[:, ilatent, :], None, S, eps),
                                   method='bounded')
            omega[ilatent] = exp(mini.x)
    return sigma, omega
    # return omega
