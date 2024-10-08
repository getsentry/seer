import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from seer.anomaly_detection.detectors.normalizers import Normalizer
from seer.anomaly_detection.models import AlgoConfig
from seer.dependency_injection import inject, injected
from seer.exceptions import ServerError


class MPUtils(BaseModel):
    @inject
    def get_mp_dist_from_mp(
        self,
        mp: npt.NDArray,
        pad_to_len: int | None = None,
        algo_config: AlgoConfig = injected,
        normalizer: Normalizer = injected,
    ) -> npt.NDArray[np.float64]:
        """
        Helper method for extracting the matrix profile distances from the matrix profile returned by the
        stumpy library

        Parameters:
        mp: Numpy array
            Contains the matrix returned by the stumpy.stump call

        pad_to_len: int
            If not none then the matrix profile is padded to the required lenght. Since Stumpy ignores MP for the first few time steps
            (as determined by the window size), the padding is done in the front of the matrix profile.

        algo_config: AlgoConfig
            mp_normalize flag in the config is used to determine if returned mp distances should be normalized. Normalization ensures that the
            distances are always between 0 and 1 (both values included)

        Returns:
        The distances as a numpy array of floats
        """
        mp_dist = mp[:, 0]
        if algo_config is not None and algo_config.mp_normalize:
            if normalizer is None:
                raise ServerError("Need normalizer to normalize MP")
            mp_dist = normalizer.normalize(mp_dist)

        if pad_to_len is not None:
            if pad_to_len < len(mp_dist):
                raise ServerError(
                    "Requested length should be greater than or equal to current mp_dist"
                )
            nan_value_count = np.empty(pad_to_len - len(mp_dist))
            nan_value_count.fill(np.nan)
            mp_dist_updated = np.concatenate((nan_value_count, mp_dist))
            return mp_dist_updated.astype(np.float64)
        else:
            return mp_dist.astype(float)
