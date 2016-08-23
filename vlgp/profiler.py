from abc import ABCMeta, abstractmethod


class Profiler(metaclass=ABCMeta):
    @abstractmethod
    def profile(self, fit):
        pass


class DefaultProfiler(Profiler):
    def __init__(self, x=None, a=None, b=None):
        self._x = x
        self._a = a
        self._b = b

    def profile(self, fit):
        pass
