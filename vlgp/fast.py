"""
Fast version of EM algorithm
---
Cut trials into small segments for EM
Infer latent processes with EM-estimated parameters
"""
import math
from typing import Iterable

import numpy as np


def cut_trial(trial: dict, length=50):
    """Cut a trial into small segments

    :param trial: a trial
    :param length: maximum length of subtrials
    :return: subtrials
    """
    # length determines the running speed, the smaller the faster, but small substrials lose long-term correlation
    # trial['y'] (timestep, neuron)
    y = trial['y']
    nbin = y.shape[0]
    nsub = math.ceil(nbin / length)  # number of subtrials
    subys = np.array_split(y, nsub, axis=0)

    # the subtrials should inherit the other properties of the trial
    template_trial = trial.copy()
    del template_trial['y']
    subtrials = (dict(y=suby, **template_trial) for suby in subys)

    return subtrials


def cut_trials(trials: Iterable) -> Iterable:
    """Cut trials into small segments

    :param trials: list of trials
    :return: subtrials
    """
    from itertools import chain
    return chain([cut_trial(trial) for trial in trials])
