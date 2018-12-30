from abc import ABCMeta, abstractmethod
import pickle
from pathlib import Path


class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def save(self, file):
        own_fid = False
        if isinstance(file, str) or isinstance(file, Path):
            fid = open(file, "wb")
            own_fid = True
        else:
            fid = file

        try:
            pickle.dump(self, fid)
        finally:
            if own_fid:
                fid.close()

    @staticmethod
    def load(file):
        with open(file, "rb") as f:
            model = pickle.load(f)
        return model


class VLGP(Model):
    def __init__(self, n_factors, random_state=0, **kwargs):
        self.n_factors = n_factors

    def fit(self, trials, **kwargs):
        """Fit the vLGP model to data using vEM
        :param trials: list of trials
        :return: the trials containing the latent factors
        """
        return trials

    def infer(self, trials):
        raise NotImplementedError()
