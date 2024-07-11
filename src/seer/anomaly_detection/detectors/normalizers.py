import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class Normalizer(BaseModel):
    def normalize(self, array: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError("Subclasses should implement this!")


class MinMaxNormalizer(Normalizer):
    def normalize(self, array: npt.NDArray) -> npt.NDArray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))
