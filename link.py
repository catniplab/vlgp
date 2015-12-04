from numpy import vectorize, exp, finfo, log1p, inf

MIN_EXP = -20
MAX_EXP = 20


def rectlin(x):
    return x.clip(0, inf)


def sexp(x):
    return exp(x.clip(MIN_EXP, MAX_EXP))


def identity(x):
    return x


def log1exp(x):
    return log1p(exp(x))
