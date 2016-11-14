from numpy import asarray, newaxis

from vlgp.experimental import initialize, vem, postprocess


def fit(y,
        types,
        z_ndim,
        x=None,
        a=None,
        b=None,
        history_filter=0,
        mu=None,
        z=None,
        alpha=None,
        beta=None,
        sigma=None,
        omega=None,
        rank=None,
        eps=1e-8,
        tol=1e-5,
        **kwargs):
    """
    vLGP main function

    Parameters
    ----------
    y : ndarray
        obserbation
    types : ndarray
        types of observation dimensions, 'spike' or 'lfp'
    z_ndim : int
        number of latent dimensions
    x : ndarray
        regression variables
    a : ndarray, optional
        initial value of loading
    b : ndarray, optional
        initial value of regression
    mu : ndarray, optional
        initial value of posterior mean
    z : ndarray, optional
        true value of latent
    alpha : ndarray, optional
        true value of loading
    beta : ndarray, optional
        true value of regression
    sigma : ndarray, optional
        initial value of prior variance
    omega : ndarray, optional
        initial value of prior timescale
    history_filter : int, optional
    rank : int, optional
        rank of incomplete Cholesky
    eps : double, optional
        a small positive number
    tol : double, optional
        numerical tolerance
    kwargs : dict, optional
        algorithm options. See fill_options()

    Returns
    -------
    dict
        fit
    """

    model = dict()
    y = asarray(y)
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]
    model['y'] = y
    model['z_ndim'] = z_ndim
    model['history_filter'] = history_filter
    model['types'] = types
    model['x'] = x
    model['a'] = a
    model['b'] = b
    model['mu'] = mu
    model['sigma'] = sigma
    model['omega'] = omega
    model['rank'] = rank
    model['z'] = z
    model['alpha'] = alpha
    model['beta'] = beta

    # model['x'] = make_regression(x, y, history_filter)

    options = dict(kwargs)
    options['eps'] = eps
    options['tol'] = tol

    initialize(model)
    vem(model)
    postprocess(model)

    return model
