import abc

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class Normalizer(BaseModel, abc.ABC):
    """
    Abstract base class for normalizing data
    """

    @abc.abstractmethod
    def normalize(self, array: npt.NDArray) -> npt.NDArray:
        return NotImplemented


class MinMaxNormalizer(Normalizer):
    def normalize(self, array: npt.NDArray) -> npt.NDArray:
        """Applies min-max normalization to input array"""
        if array.var() == 0:
            if array[0] == 0:
                return array
            else:
                return np.full_like(array, 1.0)
        return (array - np.min(array)) / (np.max(array) - np.min(array))
