"""
This file contains helper functions to plot latent dynamics and spike trains
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def dynamics(x, ncol=4, figsize=None, fontsize=12):
    """Plot latent dynamics of trials into subplots
    Args:
        x: latent dynamics
        ncol: number of columns of subplot layout
        figsize: figure size

    Returns:

    """
    x = np.asarray(x)
    if x.ndim < 3:
        x = np.atleast_3d(x)
        x = np.rollaxis(x, axis=-1)
    ntrial, ntime, ndyn = x.shape
    if figsize is None:
        figsize = mpl.rcParams['figure.figsize']
    assert len(figsize) == 2
    if ntrial < ncol:
        ncol = ntrial
    nrow = int(np.ceil(ntrial / ncol))

    ymin = x.min() * 1.1
    ymax = x.max() * 1.1

    plt.figure(figsize=(ncol * figsize[0], figsize[1] * nrow))
    for m in range(ntrial):
        i = m // ncol
        j = m % ncol
        ax = plt.subplot2grid((nrow, ncol), (i, j))
        ax.plot(x[m, :])
        ax.axis('off')
        # plt.ylim([ymin, ymax])
        ax.set_title('Trial {}'.format(m + 1), fontsize=fontsize)
    plt.tight_layout()


def spike(spike, ncol=4, figsize=None, margin=0.1, fontsize=12):
    """Raster plot of spike trains
    Args:
        spike: spike trains
        ncol: number of columns of subplot layout
        figsize: figure size
        margin: margin size of each neuron

    Returns:

    """
    spike = np.asarray(spike)
    if spike.ndim < 3:
        spike = np.atleast_3d(spike)
        spike = np.rollaxis(spike, axis=-1)
    ntrial, ntime, ntrain = spike.shape

    if figsize is None:
        figsize = mpl.rcParams['figure.figsize']
    assert len(figsize) == 2

    if ntrial < ncol:
        ncol = ntrial
    nrow = int(np.ceil(ntrial / ncol))
    plt.figure(figsize=(ncol * figsize[0], figsize[1] * nrow))

    for m in range(ntrial):
        i = m // ncol
        j = m % ncol
        ax = plt.subplot2grid((nrow, ncol), (i, j))
        plt.ylim(0, ntrain)
        for n in range(ntrain):
            plt.vlines(np.arange(ntime)[spike[m, :, n] > 0], n + margin, n + 1 - margin, color='k', lw=1)
        plt.yticks([])
        ax.axis('off')
        ax.invert_yaxis()
        ax.set_title('Trial {}'.format(m + 1), fontsize=fontsize)
    plt.tight_layout()
