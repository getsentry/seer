import abc

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class Normalizer(BaseModel, abc.ABC):
    @abc.abstractmethod
    def normalize(self, array: npt.NDArray) -> npt.NDArray:
        return NotImplemented


class MinMaxNormalizer(Normalizer):
    def normalize(self, array: npt.NDArray) -> npt.NDArray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))
