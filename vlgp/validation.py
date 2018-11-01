import h5py
import numpy as np

from .api import fit


def cv(
    y,
    z_dim,
    sigma,
    omega,
    rank,
    mfold=0,
    nfold=0,
    path=None,
    history=0,
    callbacks=None,
    random_state=None,
    **kwargs
):
    np.random.seed(random_state)

    ntrial, nbin, y_dim = y.shape

    mfold = mfold if mfold > 0 else ntrial

    trial_perm = np.random.permutation(ntrial)
    folds = np.array_split(trial_perm, mfold)  # k-fold

    for i, fold in enumerate(folds):
        training_mask = np.ones(ntrial, dtype=bool)
        training_mask[fold] = False
        y_training = y[training_mask, :, :]
        y_test = y[~training_mask, :, :]
        lik = ["spike"] * y_training.shape[2]
        fold_path = "{}_fold_{}".format(path, i)
        # DEBUG
        print(fold_path)
        model_training = fit(
            y=y_training,
            z_dim=z_dim,
            lik=lik,
            history=history,
            sigma=sigma,
            omega=omega,
            rank=rank,
            path=fold_path,
            callbacks=callbacks,
            learn_param=True,
            learn_post=True,
            learn_hyper=True,
            # e_niter=5, m_niter=5,
            **kwargs
        )
        with h5py.File(fold_path, "a") as fout:
            fout["fold"] = fold

        leave_out(
            y_test,
            model_training,
            path=fold_path,
            leave=y_test.shape[-1] // nfold,
            niter=50,
            e_niter=1,
        )


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

    _, nbin, z_dim = model["mu"].shape
    y_dim = y.shape[-1]

    # Z = USV'
    # Za = USV'a = (US)(V'a) = (USV'V)(V'a)
    u, s, vt = svd(model["mu"].reshape(-1, z_dim), full_matrices=False)
    a_orth = vt @ model["a"]
    b = model["b"]

    nfold = y_dim // leave

    if 0 < nfold < y_dim:
        y_perm = np.random.permutation(y_dim)
    elif nfold == y_dim:
        y_perm = np.arange(y_dim)
    else:
        raise ValueError("invalid leave: {}".format(leave))

    folds = np.array_split(y_perm, nfold)  # k-fold

    for i, fold in enumerate(folds):
        in_mask = np.ones(y_dim, dtype=bool)
        in_mask[fold] = False
        y_in = y[:, :, in_mask]
        a_in = a_orth[:, in_mask]  # orth
        b_in = b[:, in_mask]
        lik = ["spike"] * y_in.shape[-1]
        fold_path = "{}_leave_{}_out_{}".format(path, leave, i)
        # DEBUG
        print("{}".format(fold_path))
        fit(
            y=y_in,
            z_dim=z_dim,
            lik=lik,
            a=a_in,
            b=b_in,
            history=model["history"],
            sigma=model["sigma"],
            omega=model["omega"],
            rank=model["rank"],
            path=fold_path,
            learn_param=False,
            learn_post=True,
            learn_hyper=False,
            **kwargs
        )

        with h5py.File(fold_path, "a") as fout:
            fout["fold"] = fold
