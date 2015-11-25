def plotdynamics(x, figsize=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from numpy import asarray, atleast_3d, rollaxis

    x = asarray(x)
    if x.ndim < 3:
        x = atleast_3d(x)
        x = rollaxis(x, axis=-1)
    ntrial, ntime, ndyn = x.shape
    if figsize is None:
        figsize = mpl.rcParams['figure.figsize']
    assert len(figsize) == 2
    plt.figure(figsize=(ntrial*figsize[0], figsize[1]))
    for m in range(ntrial):
        ax = plt.subplot2grid((1, ntrial), (0, m))
        ax.plot(x[m, :])
        plt.title('Trial {}'.format(m))
    plt.tight_layout()