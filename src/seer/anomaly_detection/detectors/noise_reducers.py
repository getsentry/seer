import abc

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class NoiseReducer(BaseModel, abc.ABC):
    """
    Abstract base class for selecting the noise parameter for stumpy
    """

    @abc.abstractmethod
    def get_noise_parameter(
        self, timeseries: npt.NDArray, window: int = 12, scale_factor: float = 1.0
    ) -> float:
        return NotImplemented


class VarianceNoiseReducer(NoiseReducer):
    def get_noise_parameter(
        self, timeseries: npt.NDArray, window: int = 12, scale_factor: float = 1.0
    ) -> float:
        """
        Gets the noise parameter by calculating the median variance across sliding non-overlapping windows of the timeseries.

        Parameters:
        -----------
        timeseries : npt.NDArray
            Input time series array
        window : int, default=12
            Size of sliding window
        scale_factor : float, default=1.0
            Factor to scale the final noise parameter

        Returns:
        --------
        float
            Noise parameter calculated as median variance * scale_factor
        """

        # TODO: The window should be ~half a day so should be based on ad_config
        if len(timeseries) == 0 or window <= 0:
            return 0.0

        window = min(window, len(timeseries))

        n_windows = len(timeseries) // window
        windowed_ts = timeseries[: n_windows * window].reshape(n_windows, window)
        variances = np.var(windowed_ts, axis=1)

        noise_parameter = np.median(variances) * scale_factor

        return noise_parameter
