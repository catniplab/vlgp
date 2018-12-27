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


class VLGP(Model):
    def __init__(self, n_factors, random_state=0, **kwargs):
        self.n_factors = n_factors

    def fit(self, trials):
        """Fit the vLGP model to data using vEM
        :param trials: list of trials
        :return: the trials containing the latent factors
        """
        return trials

    def infer(self, trials):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()
