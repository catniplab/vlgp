import timeit


def timed(func):
    # TODO: find somewhere to store the time
    def wrapper(*args, **kwargs):
        tick = timeit.default_timer()
        ret = func(*args, **kwargs)
        tock = timeit.default_timer()
        return ret
    return wrapper
