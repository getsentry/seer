import abc
import logging
from typing import Dict, List, Optional

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TimeSeriesAnomalies(BaseModel, abc.ABC):
    """
    Abstract base class for storing anomaly flags and scores. Need to extend this class to store additional information needed for
    each anomaly detection algorithm.
    """

    flags: list[str] = Field(..., description="Anomaly flags")

    scores: list[float] = Field(..., description="Anomaly scores")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @abc.abstractmethod
    def get_anomaly_algo_data(self, front_pad_to_len: int) -> List[Optional[Dict]]:
        return NotImplemented

    # @root_validator(pre=False)
    # def validate_all_fields_at_the_same_time(cls, field_values):
    #     if len(field_values.get("types")) != len(field_values.get("scores")):
    #         raise ValidationError("Scores and types need to be of the same length.")

    #     return field_values  # this is the value written to the class field


class MPTimeSeriesAnomaliesSingleWindow(TimeSeriesAnomalies):
    """
    Anomalies detected using a single window
    """

    flags: list[str] = Field(..., description="Anomaly flags")

    scores: list[float] = Field(..., description="Anomaly scores")

    thresholds: list[float] = Field(..., description="Score thresholds")

    matrix_profile: npt.NDArray = Field(
        ..., description="The matrix profile of the time series using which anomalies were detected"
    )

    window_size: int = Field(..., description="Window size used to build the matrix profile")

    original_flags: list[str | None] = Field(
        default=[], description="The original flags of the time series"
    )

    def get_anomaly_algo_data(self, front_pad_to_len: int) -> List[Optional[Dict]]:
        algo_data: List[Optional[Dict]] = []
        if len(self.matrix_profile) < front_pad_to_len:
            algo_data = [None] * (front_pad_to_len - len(self.matrix_profile))

        for i, (dist, index, l_index, r_index) in enumerate(self.matrix_profile):
            original_flag = self.original_flags[i] if i < len(self.original_flags) else "none"
            algo_data.append(
                {
                    "dist": dist,
                    "idx": index,
                    "l_idx": l_index,
                    "r_idx": r_index,
                    "original_flag": original_flag,
                }
            )

        return algo_data

    @staticmethod
    def extract_algo_data(map: dict):
        return (
            map.get("dist"),
            map.get("idx"),
            map.get("l_idx"),
            map.get("r_idx"),
            map.get("original_flag"),
        )


class MPTimeSeriesAnomalies(TimeSeriesAnomalies):
    """
    Anomalies detected using both the SuSS and fixed windows
    """

    flags: list[str] = Field(..., description="Anomaly flags")

    scores: list[float] = Field(..., description="Anomaly scores")

    thresholds: list[float] = Field(..., description="Score thresholds")

    matrix_profile_suss: npt.NDArray = Field(
        ...,
        description="The matrix profile of the time series using which anomalies were detected using the SuSS window",
    )

    matrix_profile_fixed: npt.NDArray = Field(
        ...,
        description="The matrix profile of the time series using which anomalies were detected using the fixed window",
    )

    window_size: int = Field(..., description="Window size used to build the matrix profile")

    original_flags: list[str | None] = Field(
        default=[], description="The original flags of the time series"
    )

    use_suss: list[bool] = Field(
        ..., description="Whether the SuSS window was used to detect anomalies"
    )

    def get_anomaly_algo_data(self, front_pad_to_len: int) -> List[Optional[Dict]]:
        algo_data: List[Optional[Dict]] = []

        # Pad the algo_data with None to based on the size of the largest matrix profile
        padding_suss = front_pad_to_len - len(self.matrix_profile_suss)
        padding_fixed = front_pad_to_len - len(self.matrix_profile_fixed)

        algo_data = [None] * min(padding_suss, padding_fixed)

        for i in range(min(padding_suss, padding_fixed), front_pad_to_len):
            # if algo_data[i] is None:
            #     continue

            mp_suss = None
            if i >= padding_suss:
                suss_idx = i - padding_suss
                dist_suss, index_suss, l_index_suss, r_index_suss = self.matrix_profile_suss[
                    suss_idx
                ]
                mp_suss = {
                    "dist": dist_suss,
                    "idx": index_suss,
                    "l_idx": l_index_suss,
                    "r_idx": r_index_suss,
                }

            mp_fixed = None
            if i >= padding_fixed:
                fixed_idx = i - padding_fixed
                dist_fixed, index_fixed, l_index_fixed, r_index_fixed = self.matrix_profile_fixed[
                    fixed_idx
                ]
                mp_fixed = {
                    "dist": dist_fixed,
                    "idx": index_fixed,
                    "l_idx": l_index_fixed,
                    "r_idx": r_index_fixed,
                }
            # TODO: Confirm that these indexes are correct
            original_flag = self.original_flags[i] if i < len(self.original_flags) else "none"
            use_suss = self.use_suss[i] if i < len(self.use_suss) else True

            algo_data.append(
                {
                    "mp_suss": mp_suss,
                    "mp_fixed": mp_fixed,
                    "original_flag": original_flag,
                    "use_suss": use_suss,
                }
            )

        return algo_data

    @staticmethod
    def extract_algo_data(map: dict):
        # Matrix profile for SuSS and fixed windows could be None so much check for that
        mp_suss = (
            {
                "dist": map.get("dist_suss"),
                "idx": map.get("idx_suss"),
                "l_idx": map.get("l_idx_suss"),
                "r_idx": map.get("r_idx_suss"),
            }
            if any(
                map.get(key) is not None
                for key in ["dist_suss", "idx_suss", "l_idx_suss", "r_idx_suss"]
            )
            else None
        )

        mp_fixed = (
            {
                "dist": map.get("dist_fixed"),
                "idx": map.get("idx_fixed"),
                "l_idx": map.get("l_idx_fixed"),
                "r_idx": map.get("r_idx_fixed"),
            }
            if any(
                map.get(key) is not None
                for key in ["dist_fixed", "idx_fixed", "l_idx_fixed", "r_idx_fixed"]
            )
            else None
        )

        return {
            "mp_suss": mp_suss,
            "mp_fixed": mp_fixed,
            "original_flag": map.get("original_flag"),
            "use_suss": map.get("use_suss", True),
        }
