import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timer():
    tick = time.perf_counter()
    yield lambda: tock - tick
    tock = time.perf_counter()


def loglik(fit):
    """Log-likelihood of fitted model"""
    trials = fit["trials"]
    params = fit["params"]
    logrates = [np.exp(trial["mu"] @ params["a"] + trial["x"] @ params["b"]) for trial in trials]
    return np.sum([np.sum(trial["y"] * lograte - np.exp(lograte)) for trial, lograte in zip(trials, logrates)])
