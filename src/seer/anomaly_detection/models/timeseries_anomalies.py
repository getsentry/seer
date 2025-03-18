import abc
import logging
from enum import Enum, StrEnum
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ThresholdType(Enum):
    TREND = 1
    PREDICTION = 2
    MP_DIST_IQR = 3
    LOW_VARIANCE_THRESHOLD = 4
    BOX_COX_THRESHOLD = 5


class Threshold(BaseModel):
    type: ThresholdType
    timestamp: float
    upper: float
    lower: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

    flags: List[str] = Field(..., description="Anomaly flags")

    scores: List[float] = Field(..., description="Anomaly scores")

    thresholds: Optional[List[List[Threshold]]] = Field(..., description="Score thresholds")

    matrix_profile: npt.NDArray = Field(
        ..., description="The matrix profile of the time series using which anomalies were detected"
    )

    window_size: int = Field(..., description="Window size used to build the matrix profile")

    original_flags: List[str | None] = Field(
        default=[], description="The original flags of the time series"
    )

    confidence_levels: List[ConfidenceLevel] = Field(
        ..., description="The confidence levels of the anomalies"
    )

    def get_anomaly_algo_data(self, front_pad_to_len: int) -> List[Optional[Dict]]:
        algo_data: List[Optional[Dict]] = []
        if len(self.matrix_profile) < front_pad_to_len:
            algo_data = [None] * (front_pad_to_len - len(self.matrix_profile))

        for i, (dist, index, l_index, r_index) in enumerate(self.matrix_profile):
            original_flag = self.original_flags[i] if i < len(self.original_flags) else "none"
            confidence_level = (
                self.confidence_levels[i]
                if i < len(self.confidence_levels)
                else ConfidenceLevel.MEDIUM
            )
            algo_data.append(
                {
                    "dist": dist,
                    "idx": index,
                    "l_idx": l_index,
                    "r_idx": r_index,
                    "original_flag": original_flag,
                    "confidence_level": confidence_level,
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
            map.get("confidence_level"),
        )

    def extend(
        self, other: "MPTimeSeriesAnomaliesSingleWindow"
    ) -> "MPTimeSeriesAnomaliesSingleWindow":
        self.window_size = other.window_size
        self.flags.extend(other.flags)
        self.scores.extend(other.scores)
        if self.thresholds is not None and other.thresholds is not None:
            self.thresholds.extend(other.thresholds)
        elif other.thresholds is not None:
            self.thresholds = other.thresholds
        self.matrix_profile = np.append(self.matrix_profile, other.matrix_profile)
        self.original_flags.extend(other.original_flags)
        return self


class MPTimeSeriesAnomalies(TimeSeriesAnomalies):
    """
    Anomalies detected using both the SuSS and fixed windows
    """

    flags: list[str] = Field(..., description="Anomaly flags")

    scores: list[float] = Field(..., description="Anomaly scores")

    thresholds: Optional[List[List[Threshold]]] = Field(..., description="Score thresholds")

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

    confidence_levels: List[ConfidenceLevel] = Field(
        ..., description="The confidence levels of the anomalies"
    )

    def get_anomaly_algo_data(self, front_pad_to_len: int) -> List[Optional[Dict]]:
        algo_data: List[Optional[Dict]] = []

        # Pad the algo_data with None to based on the size of the largest matrix profile
        padding_suss = front_pad_to_len - len(self.matrix_profile_suss)
        padding_fixed = front_pad_to_len - len(self.matrix_profile_fixed)

        algo_data = [None] * min(padding_suss, padding_fixed)

        for i in range(min(padding_suss, padding_fixed), front_pad_to_len):
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
            original_flag = self.original_flags[i] if i < len(self.original_flags) else "none"
            use_suss = True  # Default to true since we are only using SuSS window
            confidence_level = (
                self.confidence_levels[i]
                if i < len(self.confidence_levels)
                else ConfidenceLevel.MEDIUM
            )

            algo_data.append(
                {
                    "mp_suss": mp_suss,
                    "mp_fixed": mp_fixed,
                    "original_flag": original_flag,
                    "use_suss": use_suss,
                    "confidence_level": confidence_level,
                }
            )

        return algo_data

    @staticmethod
    def extract_algo_data(map: dict):
        # Matrix profile for SuSS and fixed windows could be None so we check for that

        mp_suss = None
        mp_fixed = None
        # Indicating older algo_data format
        if "dist" in map:
            mp_suss = {
                "dist": map.get("dist"),
                "idx": map.get("idx"),
                "l_idx": map.get("l_idx"),
                "r_idx": map.get("r_idx"),
            }
        else:
            mp_suss_data = map.get("mp_suss")

            if mp_suss_data is not None:
                mp_suss = {
                    "dist": mp_suss_data.get("dist"),
                    "idx": mp_suss_data.get("idx"),
                    "l_idx": mp_suss_data.get("l_idx"),
                    "r_idx": mp_suss_data.get("r_idx"),
                }

            mp_fixed_data = map.get("mp_fixed")

            if mp_fixed_data is not None:
                mp_fixed = {
                    "dist": mp_fixed_data.get("dist"),
                    "idx": mp_fixed_data.get("idx"),
                    "l_idx": mp_fixed_data.get("l_idx"),
                    "r_idx": mp_fixed_data.get("r_idx"),
                }

        return {
            "mp_suss": mp_suss,
            "mp_fixed": mp_fixed,
            "original_flag": map.get("original_flag"),
            "use_suss": map.get("use_suss", True),
            "confidence_level": map.get("confidence_level"),
        }
