from abc import ABCMeta, abstractmethod


class Profiler(metaclass=ABCMeta):
    @abstractmethod
    def profile(self, fit):
        pass


class DefaultProfiler(Profiler):
    def profile(self, fit):
        pass
