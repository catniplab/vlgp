from abc import ABCMeta, abstractmethod

import h5py
import numpy as np


class ModelFit(metaclass=ABCMeta):
    """Base class for fitted models"""

    @staticmethod
    @abstractmethod
    def load(path):
        """
        Load fitted model from file system

        Parameters
        ----------
        path : str
            filename

        Returns
        -------
        ModelFit
        """
        pass

    @abstractmethod
    def save(self, path, replace=False):
        """
        Save fitted model to file system

        Parameters
        ----------
        path : str
            filename
        replace : bool
            replace existed file

        Returns
        -------

        """
        pass

    @property
    @abstractmethod
    def fitted(self):
        """Indicate if the fitting is done"""
        pass


class VLGPModelFit(ModelFit):
    def __init__(self, fit):
        assert fit is not None
        self._fit = fit

    @staticmethod
    def load(path):
        with h5py.File(path, 'r') as hf:
            obj = {k: np.asarray(v) for k, v in hf.items()}
        return VLGPModelFit(obj)

    def save(self, path, replace=False):
        with h5py.File(path, 'w' if replace else 'x') as hf:
            for k, v in self._fit.items():
                try:
                    hf.create_dataset(k, data=v, compression="gzip")
                except TypeError:
                    pass

    @property
    def fitted(self):
        return self.__fitted

    @property
    def prior(self):
        return self._fit['sigma'], self._fit['omega']

    @property
    def posterior(self):
        return self._fit['mu'], self._fit['v']

    @property
    def loading(self):
        return self._fit['a']

    @property
    def coefficient(self):
        return self._fit['b']

    @property
    def data(self):
        return self._fit['y'], self._fit['channel']

    @property
    def truth(self):
        return self._fit['x'], self._fit['a'], self._fit['b']

    @property
    def evaluation(self):
        return None
