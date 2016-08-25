import logging
import timeit
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import identity, empty, einsum, allclose, diag, var, arange, dstack, exp, sqrt, log, trace, \
    expand_dims, zeros, ones, repeat, copyto, asarray, inf
from numpy.random.mtrand import choice
from scipy.linalg import solve, LinAlgError, toeplitz, norm
from scipy.optimize import minimize_scalar

from .hyper import kl
from .initializer import FAInitializer
from .math import sexp, ichol_gauss
from .optimizer import AdamOptimizer
from .profiler import DefaultProfiler
from .util import add_constant, lagmat


class Model(metaclass=ABCMeta):
    pass


class VLGPModel(Model):
    def __init__(self,
                 dyn_ndim,
                 obs_ndim,
                 nbin,
                 lag):
        self._dyn_ndim = dyn_ndim
        self._obs_ndim = obs_ndim
        self._nbin = nbin
        self._lag = lag
        self._fit = VLGPModelFit()

        # setup logging
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

    def fit(self,
            y,
            obs_types,
            prior_std,
            prior_scale,
            rank,
            learning_rate=0.01,
            initializer_cls=FAInitializer,
            mu0=None,
            a0=None,
            b0=None,
            optimizer_cls=AdamOptimizer,
            optimizer_args=None,
            hessian=True,
            regular_hessian=False,
            method='VB',
            max_inner_niter=1,
            max_outer_niter=500,
            skip_estep=False,
            skip_mstep=False,
            hyperstep=10,
            fixed_prior_std=False,
            fixed_prior_scale=False,
            eps=1e-8,
            rtol=1e-5,
            atol=1e-8,
            profiler_cls=DefaultProfiler,
            profiler_args=None):
        if profiler_args is None:
            profiler_args = {}
        if optimizer_args is None:
            optimizer_args = {}
        logger = self._logger
        dyn_ndim = self._dyn_ndim
        obs_ndim = self._obs_ndim
        nbin = self._nbin
        lag = self._lag
        fit = self._fit

        obs_types = asarray(obs_types)
        # check data
        if y.ndim < 2 or y.ndim > 3 or y.shape[-1] != obs_ndim or y.shape[-2] != nbin:
            raise ValueError('wrong observation shape')
        if obs_types.size != obs_ndim:
            raise ValueError('wrong obs_types size')
        if y.ndim == 2:
            expand_dims(y, axis=0)  # add trial dimension

        ntrial = y.shape[0]
        fit.y = y
        fit.obs_types = obs_types
        fit.prior_std = prior_std
        fit.prior_scale = prior_scale
        sigma = prior_std
        omega = prior_scale

        # make design matrix of regression
        h = empty((obs_ndim, ntrial, nbin, 1 + lag), dtype=float)
        for obs_dim in range(obs_ndim):
            for trial in range(ntrial):
                h[obs_dim, trial, :] = add_constant(lagmat(y[trial, :, obs_dim], lag=lag))

        # make Cholesky of prior
        prior_ichol = empty((dyn_ndim, nbin, rank), dtype=float)
        for dyn_dim in range(dyn_ndim):
            prior_ichol[dyn_dim, :] = ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim]

        # Initialize posterior and loading
        initializer = initializer_cls(dyn_ndim)
        mu, a, b = initializer.init(y, h, mu0, a0, b0)

        ###
        # optimization options
        options = {'hessian': hessian, 'regular_hessian': regular_hessian, 'skip_estep': skip_estep,
                   'skip_mstep': skip_mstep, 'max_inner_niter': max_inner_niter, 'max_outer_niter': max_outer_niter,
                   'fixed_prior_std': fixed_prior_std, 'fixed_prior_scale': fixed_prior_scale, 'eps': eps,
                   'method': method, 'optimizer_mu': empty((ntrial, dyn_ndim), dtype=object),
                   'optimizer_a': empty(obs_ndim, dtype=object), 'optimizer_b': empty(obs_ndim, dtype=object)}

        for each in np.nditer(options['optimizer_mu'], flags=['refs_ok'], op_flags=['readwrite']):
            each[...] = optimizer_cls(nbin, learning_rate, **optimizer_args)
        for each in np.nditer(options['optimizer_a'], flags=['refs_ok'], op_flags=['readwrite']):
            each[...] = optimizer_cls(dyn_ndim, learning_rate, **optimizer_args)
        for each in np.nditer(options['optimizer_b'], flags=['refs_ok'], op_flags=['readwrite']):
            each[...] = optimizer_cls(1 + lag, learning_rate, **optimizer_args)

        ###
        # fit object
        fit.h = h
        fit.prior_ichol = prior_ichol
        fit.mu, fit.a, fit.b = mu, a, b
        fit.noise = var(y.reshape((-1, obs_ndim)), axis=0, ddof=0)
        fit.w = 0 * ones((ntrial, nbin, dyn_ndim), dtype=float)
        fit.v = repeat(sigma[None, ...], ntrial * nbin, axis=1).reshape((ntrial, nbin, dyn_ndim)) if method == 'VB' \
            else zeros((ntrial, nbin, dyn_ndim), dtype=float)  # VB or MAP
        fit.options = options

        #
        profiler = profiler_cls(**profiler_args)

        # algorithm
        # old values
        good_mu = fit.mu.copy()
        good_a = fit.a.copy()
        good_b = fit.b.copy()

        fit.stopwatch = []

        for i in range(1, max_outer_niter + 1):
            tock = tick = timeit.default_timer()
            if not skip_estep:
                self.estep()
                tock = timeit.default_timer()
            fit.stopwatch.append([i, 'E', tock - tick])
            logger.info('[{}], {}-step, time: {:.3f} s'.format(*fit.stopwatch[-1]))

            tock = tick = timeit.default_timer()
            if not skip_mstep:
                self.mstep()
                tock = timeit.default_timer()
            fit.stopwatch.append([i, 'M', tock - tick])
            logger.info('[{}], {}-step, time: {:.3f} s'.format(*fit.stopwatch[-1]))

            profiler.profile(fit)

            # check convergence
            converged = allclose(good_mu, mu, rtol=rtol, atol=atol) and allclose(good_a, a, rtol=rtol,
                                                                                 atol=atol) and allclose(good_b, b,
                                                                                                         rtol=rtol,
                                                                                                         atol=atol)

            if converged:
                break

            # store current values
            copyto(good_mu, mu)  # (dst, src, casting='same_kind', where=None)
            copyto(good_a, a)
            copyto(good_b, b)

            if i % hyperstep == 0:
                self.hstep()
        else:
            logger.warning('max niter reached')

        return fit

    def estep(self):
        fit = self._fit
        options = self._fit.options
        obs_ndim, ntrial, nbin, lag1 = fit.h.shape  # neuron, trial, time, lag + 1
        dyn_ndim, _, rank = fit.prior_ichol.shape  # latent, time, rank

        obs_types = fit.obs_types

        prior_ichol = fit.prior_ichol

        a = fit.a
        b = fit.b
        noise = fit.noise

        spike_dims = obs_types == 'spike'
        lfp_dims = obs_types == 'lfp'

        eye_rank = identity(rank)
        residual = empty((nbin, obs_ndim), dtype=float)
        U = empty((nbin, obs_ndim), dtype=float)

        for trial in range(ntrial):
            # trial slices
            y = fit.y[trial, :]
            h = fit.h[:, trial, :, :]
            mu = fit.mu[trial, :]
            w = fit.w[trial, :]
            v = fit.v[trial, :]

            hb = einsum('ijk, ki -> ji', h, b)
            eta = mu @ a + hb
            lam = sexp(eta + 0.5 * v @ (a ** 2))
            for dyn_dim in range(dyn_ndim):
                # lam, eta = firing_rate(mu, v, a, b, h)
                G = prior_ichol[dyn_dim, :, :]

                residual[:, spike_dims] = y[:, spike_dims] - lam[:, spike_dims]  # residuals of Poisson observations
                residual[:, lfp_dims] = (y[:, lfp_dims] - eta[:, lfp_dims]) / noise[
                    lfp_dims]  # residuals of Gaussian observations

                # inner loop
                optimizer = options['optimizer_mu'][trial, dyn_dim]
                old_slice = mu[:, dyn_dim].copy()
                for _ in range(options['inner_niter']):
                    wadj = w[:, [dyn_dim]]  # keep dimension
                    GtWG = G.T @ (wadj * G)
                    u = G @ (G.T @ (residual @ a[dyn_dim, :])) - old_slice
                    delta_mu = u - G @ ((wadj * G).T @ u) + G @ (
                    GtWG @ solve(eye_rank + GtWG, (wadj * G).T @ u, sym_pos=True))
                    new_slice = old_slice + optimizer.update(delta_mu)

                    # if np.allclose(old_slice, new_slice):
                    #     break
                    copyto(old_slice, new_slice)

                mu[:, dyn_dim] = old_slice

            eta = mu @ a + hb
            lam = sexp(eta + 0.5 * v @ (a ** 2))
            # lam, eta = firing_rate(mu, v, a, b, h)
            U[:, spike_dims] = lam[:, spike_dims]
            U[:, lfp_dims] = 1 / noise[lfp_dims]
            w[:] = U @ (a.T ** 2)
            if options['method'] == 'VB':
                for dyn_dim in range(dyn_ndim):
                    G = prior_ichol[dyn_dim, :, :]
                    GtWG = G.T @ (w[:, [dyn_dim]] * G)
                    v[:, dyn_dim] = (G * (G - G @ GtWG + G @ (GtWG @ solve(eye_rank + GtWG, GtWG, sym_pos=True)))).sum(
                        axis=1)

        # center over all trials
        if not options['skip_mstep']:
            old_shape = fit.mu.shape
            mu_over_trials = fit.mu.reshape((-1, dyn_ndim))
            mean_over_trials = mu_over_trials.mean(axis=0)
            fit.b[0, :] += mean_over_trials @ fit.a  # compensate regression interceptor
            mu_over_trials -= mean_over_trials
            fit.mu = mu_over_trials.reshape(old_shape)

    def mstep(self):
        fit = self._fit
        logger = self._logger
        options = self._fit.options

        eps = fit.options['eps']

        obs_ndim, ntrial, nbin, lag1 = fit.h.shape  # neuron, trial, time, lag + 1
        dyn_ndim, _, rank = fit.prior_ichol.shape  # latent, time, rank

        y = fit.y.reshape((-1, obs_ndim))  # concatenate trials
        h = fit.h.reshape((obs_ndim, -1, lag1))  # concatenate trials
        obs_types = fit.obs_types

        mu = fit.mu.reshape((-1, dyn_ndim))
        v = fit.v.reshape((-1, dyn_ndim))

        a = fit.a
        b = fit.b

        eta = mu @ a + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v @ (a ** 2))
        fit.noise = var(y - eta, axis=0, ddof=0)  # MLE
        for obs_dim in range(obs_ndim):
            optimizer_a = options['optimizer_a'][obs_dim]
            optimizer_b = options['optimizer_b'][obs_dim]

            if obs_types[obs_dim] == 'spike':
                # loading
                va = v * a[:, obs_dim]  # (nbin, dyn_ndim)
                wv = diag(lam[:, obs_dim] @ v)

                # inner loop
                old_a_slice = a[:, obs_dim].copy()
                for _ in range(options['max_inner_niter']):
                    grad_a = mu.T @ y[:, obs_dim] - (mu + va).T @ lam[:, obs_dim]

                    if options['hessian']:
                        neghess_a = (mu + va).T @ (lam[:, [obs_dim]] * (mu + va)) + wv

                        try:
                            delta_a = solve(neghess_a, grad_a, sym_pos=True)
                        except LinAlgError:
                            logger.error('singular loading hessian')
                            delta_a = grad_a
                    else:
                        delta_a = grad_a

                    delta_a = optimizer_a.update(delta_a)
                    new_a_slice = old_a_slice + delta_a
                    # if allclose(old_a_slice, new_a_slice):
                    #     break
                    copyto(old_a_slice, new_a_slice)
                a[:, obs_dim] = old_a_slice

                # bias
                old_b_slice = b[:, obs_dim].copy()
                for _ in range(options['max_inner_niter']):
                    grad_b = h[obs_dim, :].T @ (y[:, obs_dim] - lam[:, obs_dim])

                    if options['hessian']:
                        neghess_b = h[obs_dim, :].T @ (lam[:, [obs_dim]] * h[obs_dim, :])
                        # TODO: inactive neurons never fire across all trials which may cause zero Hessian
                        try:
                            delta_b = solve(neghess_b, grad_b, sym_pos=True)
                        except LinAlgError:
                            logger.error('singular regression hessian')
                            delta_b = grad_b
                    else:
                        delta_b = grad_b

                    delta_b = optimizer_b.update(delta_b)
                    new_b_slice = old_b_slice + delta_b
                    if allclose(old_b_slice, new_b_slice):
                        break
                    copyto(old_b_slice, new_b_slice)
                b[:, obs_dim] = old_b_slice
            elif obs_types[obs_dim] == 'lfp':
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                a[:, obs_dim] = solve(mu.T @ mu + diag(sum(v, axis=0)),
                                      mu.T @ (y[:, obs_dim] - h[obs_dim, :] @ b[:, obs_dim]),
                                      sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, obs_dim] = solve(h[obs_dim, :].T @ h[obs_dim, :],
                                      h[obs_dim, :].T @ (y[:, obs_dim] - mu @ a[:, obs_dim]), sym_pos=True)
            else:
                raise ValueError('unsupported observation type')

        # normalize loading
        if not options['skip_estep']:
            scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
            a /= scale
            mu *= scale.squeeze()  # compensate latent
            fit.mu = mu.reshape(fit.mu.shape)

            # noinspection PyTupleAssignmentBalance
            # U, s, Vh = svd(a, full_matrices=False)
            # fit.mu = reshape(mu @ a @ Vh.T, (ntrial, nbin, dyn_ndim))
            # fit.a = Vh

    def hstep(self):
        fit = self._fit
        options = fit.options
        eps = options['eps']

        segment_size = options.get('segment_size', 100)
        nsegment = options.get('nsegment', 100)
        omega_factor = options.get('omega_factor', 5)

        ntrial, nbin, dyn_ndim = fit.mu.shape
        mu = fit.mu.reshape((-1, dyn_ndim))
        w = fit.w.reshape((-1, dyn_ndim))

        sigma = fit.prior_std.copy()
        omega = fit.prior_scale.copy()

        if options['fixed_prior_std'] and options['fixed_prior_scale']:
            return sigma, omega

        n = ntrial * nbin - segment_size

        segment_heads = choice(arange(n), size=nsegment)
        mu_segments = dstack([mu[head:head + segment_size, :] for head in segment_heads])
        w_segments = dstack([w[head:head + segment_size, :] for head in segment_heads])
        dsq = arange(segment_size) ** 2
        Dsq = toeplitz(dsq)

        for dyn_dim in range(dyn_ndim):
            corr_mat = exp(-omega[dyn_dim] * Dsq) + eps * identity(segment_size)
            cov_mat = sigma[dyn_dim] ** 2 * exp(-omega[dyn_dim] * Dsq) + eps * identity(segment_size)
            # noinspection PyTypeChecker
            s_mat = dstack([cov_mat - cov_mat @ solve(diag(1 / (eps + w_segments[:, dyn_dim, segment])) + cov_mat,
                                                      cov_mat, sym_pos=True) for segment in
                            range(nsegment)])
            if not options['fixed_prior_std']:
                acc = sum([mu_segments[:, dyn_dim, segment] @ solve(corr_mat, mu_segments[:, dyn_dim, segment],
                                                                    sym_pos=True) + trace(
                    solve(corr_mat, s_mat[:, :, segment], sym_pos=True)) for segment in range(nsegment)])
                sigma[dyn_dim] = sqrt(acc / (segment_size * nsegment))
            if not options['fixed_prior_scale']:
                mini = minimize_scalar(kl,
                                       bounds=(log(omega[dyn_dim] / omega_factor), log(omega[dyn_dim] * omega_factor)),
                                       args=(
                                           fit.prior_std[dyn_dim], segment_size, mu_segments[:, dyn_dim, :], None,
                                           s_mat,
                                           eps),
                                       method='bounded')
                omega[dyn_dim] = exp(mini.x)
        return sigma, omega
        # return omega


# ModelFit classes
class ModelFit(metaclass=ABCMeta):
    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load(path):
        pass

    def __getattr__(self, item):
        if item not in self.__dict__:
            raise AttributeError('attribute not defined')
        return self.__dict__[item]

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @property
    @abstractmethod
    def posterior_mean(self):
        pass

    @property
    @abstractmethod
    def loading(self):
        pass

    @property
    @abstractmethod
    def regression(self):
        pass

    @property
    @abstractmethod
    def posterior_variance(self):
        pass

    @property
    @abstractmethod
    def prior(self):
        pass


class VLGPModelFit(ModelFit):
    @property
    def posterior_variance(self):
        return self.v

    @property
    def loading(self):
        return self.a

    @property
    def posterior_mean(self):
        return self.mu

    @property
    def regression(self):
        return self.b

    @property
    def prior(self):
        return self.sigma, self.omega

    @staticmethod
    def load(path):
        pass

    def save(self, path):
        pass
