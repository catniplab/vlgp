import timeit


class timed:
    def __init__(self, times):
        self._times = times

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            tick = timeit.default_timer()
            ret = f(*args, **kwargs)
            tock = timeit.default_timer()
            self._times.append(tock - tick)
            return ret
        return wrapper
