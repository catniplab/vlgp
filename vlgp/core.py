"""
The inference algorithm
introduce a new format of fit
trial isolation
unequal trial ready
"""
import copy
import logging
import warnings

import numpy as np
from numpy import identity, einsum
from scipy.linalg import solve, norm, svd, LinAlgError

from . import gp
from .base import Model
from .callback import Saver, show
from .preprocess import get_config, get_params, fill_trials, fill_params, initialize
from .util import cut_trials, clip
from .gp import make_cholesky
from .evaluation import timer
from .math import trunc_exp

logger = logging.getLogger(__name__)


def estep(trials, params, config):
    """Update variational distribution q (E step)"""
    niter = config["Eniter"]  # maximum number of iterations
    if niter < 1:
        return

    # See the explanation in mstep.
    # constrain_loading(trials, params, config)

    # dimenionalities
    zdim = params["zdim"]
    rank = params["rank"]  # rank of prior covariance
    likelihood = params["likelihood"]

    # misc
    dmu_bound = config["dmu_bound"]
    tol = config["tol"]
    method = config["method"]

    poiss_mask = likelihood == "poisson"
    gauss_mask = likelihood == "gaussian"
    ###
    # print(poiss_mask)
    # print(gauss_mask)
    ##

    # parameters
    a = params["a"]
    b = params["b"]
    noise = params["noise"]
    gauss_noise = noise[gauss_mask]

    Ir = identity(rank)
    # boolean indexing creates copies
    # pull indexing out of the loop for performance

    for i in range(niter):
        # TODO: parallel trials ?
        for trial in trials:
            y = trial["y"]
            x = trial["x"]
            mu = trial["mu"]
            w = trial["w"]
            v = trial["v"]
            dmu = trial["dmu"]

            prior = params["cholesky"][
                y.shape[0]
            ]  # TODO: adapt unequal lengths, move into trials

            residual = np.empty_like(y, dtype=float)
            U = np.empty_like(y, dtype=float)

            y_poiss = y[:, poiss_mask]
            y_gauss = y[:, gauss_mask]

            ###
            # print(y_poiss.shape)
            # print(y_gauss.shape)
            ###

            xb = einsum("ijk, jk -> ik", x, b)
            eta = mu @ a + xb
            r = trunc_exp(eta + 0.5 * v @ (a ** 2))

            ###
            # print(xb.shape)
            ###

            # mean of y
            mean_gauss = eta[:, gauss_mask]
            mean_poiss = r[:, poiss_mask]

            ###
            # print(y_poiss.shape, mean_poiss.shape)
            # print(y_gauss.shape, mean_gauss.shape, gauss_noise.shape)
            ###

            for l in range(zdim):
                G = prior[l]
                ###
                # print(G.shape)
                ###

                # working residuals
                # extensible to many other distributions
                # see GLM's working residuals

                residual[:, poiss_mask] = y_poiss - mean_poiss
                residual[:, gauss_mask] = (y_gauss - mean_gauss) / gauss_noise
                ###
                # print(w.shape)
                ###
                wadj = w[:, [l]]  # keep dimension
                ###
                # print(G.shape, wadj.shape)
                ###
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[l, :])) - mu[:, l]
                try:
                    M = solve(Ir + GtWG, (wadj * G).T @ u, sym_pos=True)
                    delta_mu = u - G @ ((wadj * G).T @ u) + G @ (GtWG @ M)
                    clip(delta_mu, dmu_bound)
                except Exception as e:
                    logger.exception(repr(e), exc_info=True)
                    delta_mu = 0

                dmu[:, l] = delta_mu
                mu[:, l] += delta_mu

            # TODO: remove duplicated computation
            eta = mu @ a + xb
            r = trunc_exp(eta + 0.5 * v @ (a ** 2))
            U[:, poiss_mask] = r[:, poiss_mask]
            U[:, gauss_mask] = 1 / gauss_noise
            w = U @ (a.T ** 2)
            if method == "VB":
                for l in range(zdim):
                    G = prior[l]
                    GtWG = G.T @ (w[:, l, np.newaxis] * G)
                    try:
                        M = solve(Ir + GtWG, GtWG, sym_pos=True)
                        v[:, l] = np.sum(G * (G - G @ GtWG + G @ (GtWG @ M)), axis=1)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)

            # make sure save all changes
            # TODO: make inline modification
            trial["mu"] = mu
            trial["w"] = w
            trial["v"] = v
            trial["dmu"] = dmu

        # center over all trials if not only infer posterior
        # constrain_mu(model)

        # if norm(dmu) < tol * norm(mu):
        #     break


def mstep(trials, params, config):
    """Optimize loading and regression (M step)"""
    niter = config["Mniter"]  # maximum number of iterations
    if niter < 1:
        return

    # It's more proper to constrain the latent before mstep.
    # If the parameters are fixed, it's no need to optimize the posterior.
    # Besides, the constraint modifies the loading and bias.
    # constrain_latent(trials, params, config)

    # dimenionalities
    ydim = params["ydim"]
    xdim = params["xdim"]
    zdim = params["zdim"]
    rank = params["rank"]  # rank of prior covariance
    ntrial = len(trials)  # number of trials

    # parameters
    a = params["a"]
    b = params["b"]
    likelihood = params["likelihood"]
    noise = params["noise"]
    poiss_mask = likelihood == "poisson"
    gauss_mask = likelihood == "gaussian"
    gauss_noise = noise[gauss_mask]
    da = params["da"]
    db = params["db"]

    # misc
    use_hessian = config["use_hessian"]
    da_bound = config["da_bound"]
    db_bound = config["db_bound"]
    tol = config["tol"]
    method = config["method"]
    learning_rate = config["learning_rate"]

    y = np.concatenate([trial["y"] for trial in trials], axis=0)
    x = np.concatenate(
        [trial["x"] for trial in trials], axis=0
    )  # TODO: check dimensionality of x
    mu = np.concatenate([trial["mu"] for trial in trials], axis=0)
    v = np.concatenate([trial["v"] for trial in trials], axis=0)

    for i in range(niter):
        eta = mu @ a + einsum("ijk, jk -> ik", x, b)
        # (time, regression, neuron) x (regression, neuron) -> (time, neuron)  # TODO: use matmul broadcast
        r = trunc_exp(eta + 0.5 * v @ (a ** 2))
        noise = np.var(y - eta, axis=0, ddof=0)  # MLE

        for n in range(ydim):
            if likelihood[n] == "poisson":
                # loading
                mu_plus_v_times_a = mu + v * a[:, n]
                grad_a = mu.T @ y[:, n] - mu_plus_v_times_a.T @ r[:, n]

                if use_hessian:
                    nhess_a = mu_plus_v_times_a.T @ (
                        r[:, n, np.newaxis] * mu_plus_v_times_a
                    )
                    nhess_a[np.diag_indices_from(nhess_a)] += r[:, n] @ v

                    try:
                        delta_a = solve(nhess_a, grad_a, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_a = learning_rate * grad_a
                else:
                    delta_a = learning_rate * grad_a

                clip(delta_a, da_bound)
                da[:, n] = delta_a
                a[:, n] += delta_a

                # regression
                grad_b = x[..., n].T @ (y[:, n] - r[:, n])

                if use_hessian:
                    nhess_b = x[..., n].T @ (r[:, np.newaxis, n] * x[..., n])
                    try:
                        delta_b = solve(nhess_b, grad_b, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_b = learning_rate * grad_b
                else:
                    delta_b = learning_rate * grad_b

                clip(delta_b, db_bound)
                db[:, n] = delta_b
                b[:, n] += delta_b
            elif likelihood[n] == "gaussian":
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                M = mu.T @ mu
                M[np.diag_indices_from(M)] += np.sum(v, axis=0)
                a[:, n] = solve(M, mu.T @ (y[:, n] - x[..., n] @ b[:, n]), sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, n] = solve(
                    x[..., n].T @ x[..., n],
                    x[..., n].T @ (y[:, n] - mu @ a[:, n]),
                    sym_pos=True,
                )
                b[1:, n] = 0
                # TODO: only make history filter components zeros
            else:
                pass

        # update parameters in fit
        # TODO: make inline modification
        params["a"] = a
        params["b"] = b
        params["noise"] = noise
        # normalize loading by latent and rescale latent
        # constrain_a(model)

        # if norm(da) < tol * norm(a) and norm(db) < tol * norm(b):
        #     break


def hstep(trials, params, config):
    """Wrapper of hyperparameters tuning"""
    if not config["Hstep"]:
        return

    gp.optimize(trials, params, config)


def infer(trials, params, config):
    estep(trials, params, config)


def vem(trials, params, config):
    """Variational EM
    This function implements the algorithm.
    """
    # this function should not know if the trials are original or segmented ones
    # the caller determines which to use
    # pass segments to speed up estimation and hyperparameter tuning
    # the caller gets runtime

    callbacks = config["callbacks"]

    tol = config["tol"]
    niter = config["EMniter"]

    # profile and debug purpose
    # invalid every new run
    runtime = {
        "it": 0,
        "e_elapsed": [],
        "m_elapsed": [],
        "h_elapsed": [],
        "em_elapsed": [],
    }

    #######################
    # iterative algorithm #
    #######################

    # disable gabbage collection during the iterative procedure
    for it in range(niter):
        # print("EM iteration", it + 1)
        runtime["it"] += 1

        with timer() as em_elapsed:
            ##########
            # E step #
            ##########
            with timer() as estep_elapsed:
                constrain_loading(trials, params, config)
                estep(trials, params, config)

            ##########
            # M step #
            ##########
            with timer() as mstep_elapsed:
                constrain_latent(trials, params, config)
                mstep(trials, params, config)

            ###################
            # H step #
            ###################
            with timer() as hstep_elapsed:
                hstep(trials, params, config)

        # print("Iter {:d}, ELBO: {:.3f}".format(it, lbound))

        runtime["e_elapsed"].append(estep_elapsed())
        runtime["m_elapsed"].append(mstep_elapsed())
        runtime["h_elapsed"].append(hstep_elapsed())
        runtime["em_elapsed"].append(em_elapsed())

        config["runtime"] = runtime

        for callback in callbacks:
            try:
                callback(trials, params, config)
            except:
                pass

        #####################
        # convergence check #
        #####################
        mu = np.concatenate([trial["mu"] for trial in trials], axis=0)
        a = params["a"]
        b = params["b"]
        dmu = np.concatenate([trial["dmu"] for trial in trials], axis=0)
        da = params["da"]
        db = params["db"]

        # converged = norm(dmu) < tol * norm(mu) and \
        #             norm(da) < tol * norm(a) and \
        #             norm(db) < tol * norm(b)
        #
        # should_stop = converged

        should_stop = False

        if should_stop:
            break

    ##############################
    # end of iterative procedure #
    ##############################


def constrain_latent(trials, params, config):
    """Center and scale latent mean"""
    constraint = config["constrain_latent"]

    if not constraint or constraint == "none":
        return

    mu = np.concatenate([trial["mu"] for trial in trials], axis=0)
    mean_over_trials = mu.mean(axis=0, keepdims=True)
    std_over_trials = mu.std(axis=0, keepdims=True)

    if constraint in ("location", "both"):
        for trial in trials:
            trial["mu"] -= mean_over_trials
        # compensate bias
        # commented to isolated from changing external variables
        params["b"][0, :] += np.squeeze(mean_over_trials @ params["a"])

    if constraint in ("scale", "both"):
        for trial in trials:
            trial["mu"] /= std_over_trials
        # compensate loading
        # commented to isolated from changing external variables
        params["a"] *= std_over_trials.T


def constrain_loading(trials, params, config):
    """Normalize loading matrix"""
    constraint = config["constrain_loading"]

    if not constraint or constraint == "none":
        return

    eps = config["eps"]
    a = params["a"]

    if constraint == "svd":
        u, s, v = svd(a, full_matrices=False)
        # A = USV
        us = a @ v.T
        for trial in trials:
            trial["mu"] = trial["mu"] @ us
        params["a"] = v
    else:
        if constraint == "fro":
            s = norm(a, ord="fro") + eps
        else:
            s = norm(a, ord=constraint, axis=1, keepdims=True) + eps
        params["a"] /= s
        for trial in trials:
            trial["mu"] *= s.T


def update_w(trials, params, config):
    likelihood = params["likelihood"]
    poiss_mask = likelihood == "poisson"
    gauss_mask = likelihood == "gaussian"

    a = params["a"]
    b = params["b"]
    noise = params["noise"]
    gauss_noise = noise[gauss_mask]

    for trial in trials:
        y = trial["y"]
        x = trial["x"]
        mu = trial["mu"]
        w = trial.setdefault("w", np.zeros_like(mu))
        v = trial.setdefault("v", np.zeros_like(mu))

        # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
        eta = mu @ a + einsum("ijk, jk -> ik", x, b)
        r = trunc_exp(eta + 0.5 * v @ (a ** 2))
        U = np.empty_like(r)
        U[:, poiss_mask] = r[:, poiss_mask]
        U[:, gauss_mask] = 1 / gauss_noise
        trial["w"] = U @ (a.T ** 2)


def update_v(trials, params, config):
    if config["method"] != "VB":
        return

    for trial in trials:
        zdim = params["zdim"]
        mu = trial["mu"]
        w = trial.setdefault("w", np.zeros_like(mu))
        v = trial.setdefault("v", np.zeros_like(mu))

        prior = params["cholesky"][mu.shape[0]]
        Ir = identity(prior[0].shape[-1])

        for l in range(zdim):
            G = prior[l]
            GtWG = G.T @ (w[:, [l]] * G)
            try:
                v[:, l] = np.sum(
                    G
                    * (
                        G - G @ GtWG + G @ (GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True))
                    ),
                    axis=1,
                )
            except LinAlgError:
                warnings.warn("singular I + G'WG")


class VLGP(Model):
    def __init__(self, n_factors, random_state=0, **kwargs):
        self.n_factors = n_factors
        self.random_state = random_state
        self._weight = None
        self._bias = None
        self.setup(**kwargs)

    def fit(self, trials, **kwargs):
        """Fit the vLGP model to data using vEM
        :param trials: list of trials
        :return: the trials containing the latent factors
        """
        config = get_config(**kwargs)

        # add built-in callbacks
        callbacks = config["callbacks"]
        if "path" in config:
            saver = Saver()
            callbacks.extend([show, saver.save])
        config["callbacks"] = callbacks

        params = get_params(trials, self.n_factors, **kwargs)

        print("Initializing...")
        initialize(trials, params, config)

        # fill arrays
        fill_params(params)

        fill_trials(trials)
        make_cholesky(trials, params, config)
        update_w(trials, params, config)
        update_v(trials, params, config)

        subtrials = cut_trials(trials, params, config)
        make_cholesky(subtrials, params, config)

        fill_trials(subtrials)

        params["initial"] = copy.deepcopy(params)
        # VEM
        print("Fitting...")
        vem(subtrials, params, config)
        # E step only for inference given above estimated parameters and hyperparameters
        make_cholesky(trials, params, config)
        update_w(trials, params, config)
        update_v(trials, params, config)
        print("Inferring...")
        infer(trials, params, config)
        print("Done")

        self._weight = params["a"]
        self._bias = params["b"]

        return trials

    def infer(self, trials):
        if not self.isfiited:
            raise ValueError(
                "This model is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
        raise NotImplementedError()

    def __eq__(self, other):
        if (
            isinstance(other, VLGP)
            and self.n_factors == other.n_factors
            and np.array_equal(self.weight, other.weight)
            and np.array_equal(self.bias, other.bias)
        ):
            return True
        return False

    def setup(self, **kwargs):
        pass

    @property
    def isfitted(self):
        return self.weight is not None

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias
