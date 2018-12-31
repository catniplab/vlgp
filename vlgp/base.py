from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
