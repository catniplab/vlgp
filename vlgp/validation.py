import numpy as np

from .api import fit


def leave_n_out(y,
                a,
                b,
                sigma,
                omega,
                rank,
                nfold=0,
                path=None,
                callbacks=None,
                **kwargs):
    ntrial, nbin, obs_ndim = y.shape
    dyn_ndim = a.shape[0]
    history_filter = b.shape[0] - 1

    nfold = nfold if nfold > 0 else obs_ndim

    obs_perm = np.random.permutation(obs_ndim)
    folds = np.array_split(obs_perm, nfold)  # k-fold

    for i, fold in enumerate(folds):
        in_mask = np.ones(obs_ndim, dtype=bool)
        in_mask[fold] = False
        y_in = y[:, :, in_mask]
        a_in = a[:, in_mask]
        b_in = b[:, in_mask]
        obs_types = ['spike'] * y_in.shape[2]
        model_in = fit(y_in, dyn_ndim, obs_types=obs_types, a=a_in, b=b_in, history_filter=history_filter,
                       sigma=sigma, omega=omega, rank=rank, path='{}_nfold_{}'.format(path, i),
                       method='VB',
                       niter=50, tol=1e-4, verbose=False,
                       learn_param=False, learn_post=True, learn_hyper=False, e_niter=2,
                       dmu_bound=0.5)


def cv(y,
       dyn_ndim,
       sigma,
       omega,
       rank,
       mfold=0,
       nfold=0,
       path=None,
       history_filter=0,
       callbacks=None,
       **kwargs):
    ntrial, nbin, obs_ndim = y.shape

    mfold = mfold if mfold > 0 else ntrial

    trial_perm = np.random.permutation(ntrial)
    folds = np.array_split(trial_perm, mfold)  # k-fold

    for i, fold in enumerate(folds):
        training_mask = np.ones(ntrial, dtype=bool)
        training_mask[fold] = False
        y_training = y[training_mask, :, :]
        y_test = y[~training_mask, :, :]
        obs_types = ['spike'] * y_training.shape[2]
        model_training = fit(y_training, dyn_ndim=dyn_ndim, obs_types=obs_types, history_filter=history_filter,
                             sigma=sigma, omega=omega, rank=rank, path='{}_mfold_{}'.format(path, i),
                             callbacks=callbacks,
                             method='VB',
                             niter=50, tol=1e-4, verbose=False,
                             learn_param=True, learn_post=True, learn_hyper=True,
                             e_niter=5, m_niter=5, nhyper=5,
                             dmu_bound=0.5, da_bound=0.1, constrain_a=np.inf,
                             omega_bound=(1e-5, 1e-2), gp='cutting', subsample_size=200, hyper_obj='ELBO',
                             successive=False)
        leave_n_out(y_test,
                    a=model_training['a'], b=model_training['b'],
                    sigma=model_training['sigma'], omega=model_training['omega'],
                    rank=rank, nfold=nfold)
