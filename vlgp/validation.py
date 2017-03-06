import h5py
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
        fold_path = '{}_nfold_{}'.format(path, i)
        # DEBUG
        print(fold_path)
        # TODO: Don't use fit. Construct model explicitly. Don't modify HDF5 file. Isolate IO implemention.
        fit(y=y_in, dyn_ndim=dyn_ndim, obs_types=obs_types, a=a_in, b=b_in, history_filter=history_filter,
            sigma=sigma, omega=omega, rank=rank, path=fold_path,
            learn_param=False, learn_post=True, learn_hyper=False, e_niter=2,
            **kwargs)

        with h5py.File(fold_path, 'a') as fout:
            fout['fold'] = fold


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
       random_state=None,
       **kwargs):
    np.random.seed(random_state)

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
        fold_path = '{}_fold_{}'.format(path, i)
        # DEBUG
        print(fold_path)
        model_training = fit(y=y_training, dyn_ndim=dyn_ndim, obs_types=obs_types, history_filter=history_filter,
                             sigma=sigma, omega=omega, rank=rank, path=fold_path,
                             callbacks=callbacks,
                             learn_param=True, learn_post=True, learn_hyper=True,
                             # e_niter=5, m_niter=5,
                             **kwargs)
        with h5py.File(fold_path, 'a') as fout:
            fout['fold'] = fold

        # leave_n_out(y_test,
        #             a=model_training['a'], b=model_training['b'],
        #             sigma=model_training['sigma'], omega=model_training['omega'],
        #             rank=rank, nfold=nfold, path=fold_path)
        leave_out(y_test, model_training, path=fold_path, leave=y_test.shape[-1] // nfold, niter=50, e_niter=1)


def leave_out(y, model, path, leave=1, **kwargs):
    """

    Parameters
    ----------
    y: test set
    model: model fit by training set
    leave: how many neurons are left-out

    Returns
    -------

    """
    from scipy.linalg import svd

    _, nbin, dyn_ndim = model['mu'].shape
    obs_ndim = y.shape[-1]

    # Z = USV'
    # Za = USV'a = (US)(V'a) = (USV'V)(V'a)
    u, s, vt = svd(model['mu'].reshape(-1, dyn_ndim), full_matrices=False)
    a_orth = vt @ model['a']
    b = model['b']

    nfold = obs_ndim // leave

    if 0 < nfold < obs_ndim:
        obs_perm = np.random.permutation(obs_ndim)
    elif nfold == obs_ndim:
        obs_perm = np.arange(obs_ndim)
    else:
        raise ValueError('invalid leave: {}'.format(leave))

    folds = np.array_split(obs_perm, nfold)  # k-fold

    for i, fold in enumerate(folds):
        in_mask = np.ones(obs_ndim, dtype=bool)
        in_mask[fold] = False
        y_in = y[:, :, in_mask]
        a_in = a_orth[:, in_mask]  # orth
        b_in = b[:, in_mask]
        obs_types = ['spike'] * y_in.shape[-1]
        fold_path = '{}_leave_{}_out_{}'.format(path, leave, i)
        # DEBUG
        print('{}'.format(fold_path))
        fit(y=y_in,
            dyn_ndim=dyn_ndim,
            obs_types=obs_types,
            a=a_in,
            b=b_in,
            history_filter=model['history_filter'],
            sigma=model['sigma'],
            omega=model['omega'],
            rank=model['rank'],
            path=fold_path,
            learn_param=False,
            learn_post=True,
            learn_hyper=False,
            **kwargs)

        with h5py.File(fold_path, 'a') as fout:
            fout['fold'] = fold
