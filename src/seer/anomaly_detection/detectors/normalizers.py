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
        return (array - np.min(array)) / (np.max(array) - np.min(array))  # type: ignore


# class MinMaxStreamNormalizer:
#     """
#     Normalizes a stream of data based on the min and max of a history.
#     """

#     # history_normalized: npt.NDArray[np.float64]
#     # min: float = Field(-np.inf, description="Minimum value of the history")
#     # max: float = Field(np.inf, description="Maximum value of the history")

#     def __init__(self, history: npt.NDArray[np.float64]):
#         mp_dist_baseline_finite = history[np.isfinite(history)]
#         self.min = np.min(mp_dist_baseline_finite)
#         self.max = np.max(mp_dist_baseline_finite)
#         self.history_normalized = self.normalize(history)

#     def normalize(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#         """Applies min-max normalization to input array"""

#         if self.max == self.min:
#             return np.array(
#                 [
#                     np.nan if np.isnan(value) or np.isinf(value) else 0.0 if value == 0.0 else 1.0
#                     for value in array
#                 ]
#             )
#         return np.array(
#             [
#                 (
#                     np.nan
#                     if np.isnan(value) or np.isinf(value)
#                     else (value - self.min) / (self.max - self.min)
#                 )
#                 for value in array
#             ]
#         )

#     model_config = ConfigDict(
#         arbitrary_types_allowed=True,
#     )
