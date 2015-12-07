def dynplot(x, ncol=4, figsize=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from numpy import asarray, atleast_3d, rollaxis
    from math import ceil

    x = asarray(x)
    if x.ndim < 3:
        x = atleast_3d(x)
        x = rollaxis(x, axis=-1)
    ntrial, ntime, ndyn = x.shape
    if figsize is None:
        figsize = mpl.rcParams['figure.figsize']
    assert len(figsize) == 2
    if ntrial < ncol:
        ncol = ntrial
    nrow = ceil(ntrial / ncol)

    ymin = x.min() * 1.1
    ymax = x.max() * 1.1

    plt.figure(figsize=(ncol * figsize[0], figsize[1] * nrow))
    for m in range(ntrial):
        i = m // ncol
        j = m % ncol
        ax = plt.subplot2grid((nrow, ncol), (i, j))
        ax.plot(x[m, :])
        plt.ylim([ymin, ymax])
        plt.title('Trial {}'.format(m))
    plt.tight_layout()


def rasterplot(spike, ncol=4, figsize=None, margin=0.1):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from numpy import asarray, atleast_3d, rollaxis, arange
    from math import ceil

    spike = asarray(spike)
    if spike.ndim < 3:
        spike = atleast_3d(spike)
        spike = rollaxis(spike, axis=-1)
    ntrial, ntime, ntrain = spike.shape

    if figsize is None:
        figsize = mpl.rcParams['figure.figsize']
    assert len(figsize) == 2

    if ntrial < ncol:
        ncol = ntrial
    nrow = ceil(ntrial / ncol)
    plt.figure(figsize=(ncol * figsize[0], figsize[1] * nrow))

    for m in range(ntrial):
        i = m // ncol
        j = m % ncol
        ax = plt.subplot2grid((nrow, ncol), (i, j))
        plt.ylim(0, ntrain);
        for n in range(ntrain):
            plt.vlines(arange(ntime)[spike[m, :, n] > 0], n + margin, n + 1 - margin, color='black', lw=1);
        plt.yticks([]);
        ax.axis('off')
        ax.invert_yaxis();
    plt.tight_layout()
