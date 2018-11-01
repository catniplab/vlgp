import time
from contextlib import contextmanager


@contextmanager
def timer():
    tick = time.perf_counter()
    yield lambda: tock - tick
    tock = time.perf_counter()
