from numpy import vectorize, exp, finfo
from .constant import *


def rectlin(eta):
    vfun = vectorize(lambda x: x if x > 0 else finfo(float).tiny)
    return vfun(eta)


def sexp(eta):
    return exp(eta.clip(MIN_EXP, MAX_EXP))


def identity(eta):
    return eta
