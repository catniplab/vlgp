def transform(timescale, dt):
    """
    Transform timescale to omega

    Parameters
    ----------
    timescale : float or array
    dt : float

    Returns
    -------
    float
    """

    return 0.5 * (dt / timescale) ** 2
