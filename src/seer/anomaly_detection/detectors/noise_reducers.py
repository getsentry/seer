import abc

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided
from pydantic import BaseModel


class NoiseReducer(BaseModel, abc.ABC):
    """
    Abstract base class for selecting the noise parameter for stumpy
    """

    @abc.abstractmethod
    def get_noise_parameter(
        self, timeseries: npt.NDArray, window: int = 0, scale_factor: float = 1.0
    ) -> float:
        return NotImplemented


class VarianceNoiseReducer(NoiseReducer):

    def _get_subsequences(self, arr: npt.NDArray, m: int) -> npt.NDArray:
        """
        Gets the vectorized subsequences of size m of the input array
        """
        n = arr.size - m + 1
        s = arr.itemsize
        subs = as_strided(arr, shape=(n, m), strides=(s, s))
        return subs

    def get_noise_parameter(
        self, timeseries: npt.NDArray, window: int = 0, scale_factor: float = 1.0
    ) -> float:
        """
        Gets the noise parameter by calculating the median variance across sliding non-overlapping windows of the timeseries.

        Parameters:
        -----------
        timeseries : npt.NDArray
            Input time series array
        window : int, default=12
            Size of sliding window for calculating variances. Default is
        scale_factor : float, default=1.0
            Factor to scale the final noise parameter. A higher value will result in more aggressive noise reduction.

        Returns:
        --------
        float
            Noise parameter calculated as median variance * scale_factor
        """

        # Limiting our selection to the last num_days
        # TODO: This should be based on ad_config
        num_days = 7
        num_points = 24 * num_days
        timeseries = timeseries[-num_points:]

        # Paper suggests 10% of the reference data
        if window == 0:
            window = int(0.1 * len(timeseries))

        variances = np.var(self._get_subsequences(timeseries, window), axis=1)

        # Taking the 20th percentile as opposed to the 5th percentile
        noise_parameter = np.percentile(variances, 20)
        return noise_parameter * scale_factor
