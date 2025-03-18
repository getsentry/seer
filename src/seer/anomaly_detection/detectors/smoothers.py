import abc
from typing import List, Tuple, cast

import numpy as np
import sentry_sdk
from pydantic import BaseModel

from seer.anomaly_detection.models import AlgoConfig
from seer.anomaly_detection.models.external import AnomalyDetectionConfig


class FlagSmoother(BaseModel, abc.ABC):
    """
    Abstract base class for smoothing data (specifically flags)
    """

    @abc.abstractmethod
    def smooth(
        self,
        flags: list,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig,
        smooth_size: int = 0,
        vote_threshold: float = 0.5,
    ) -> list:
        return NotImplemented


class MajorityVoteBatchFlagSmoother(FlagSmoother):
    """
    This class smooths flags using majority voting with a dynamic smoothing window size for batch data.
    """

    def _get_anomalous_slices(
        self, flags: list, time_period: int, algo_config: AlgoConfig
    ) -> List[Tuple[int, int]]:
        """
        Returns a list of slices where anomalies occur. A slice is a tuple of the start and end index of the anomalous region.
        """
        slices = []

        num_gap = (
            algo_config.period_to_smooth_size[time_period] // 2
        )  # Number of non-consistent flags allowed between points based on the smoothing size

        # Sliding window O(n)
        i = 0
        while i < (len(flags)):
            if flags[i] == "none":
                i += 1
                continue

            j = i
            gap_remaining = num_gap
            while j < len(flags) and gap_remaining > 0:
                if flags[j] == "none":
                    gap_remaining -= 1
                else:
                    gap_remaining = num_gap
                j += 1
            if j != i:
                slices.append((i, j - num_gap))
                i = j
            i += 1

        return slices

    def _batch_smooth_flags(
        self,
        orig_flags: list,
        start_idx: int,
        end_idx: int,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig,
        smooth_size: int = 0,
        vote_threshold: float = 0.5,
    ) -> list:
        """
        Use original flags to smooth by majority (or threshold) voting with a small window.
        """

        # Use dynamic smooth size if not provided
        if smooth_size == 0:
            smooth_size = algo_config.period_to_smooth_size[ad_config.time_period]

        new_flags = np.array(orig_flags)

        # Base case is to set the flags after first detected anomaly within window size to be anomalous
        new_flags[start_idx : start_idx + smooth_size] = "anomaly_higher_confidence"

        for i in range(start_idx + smooth_size, end_idx):
            values, counts = np.unique(orig_flags[i - smooth_size : i], return_counts=True)
            flag_counts = dict(zip(values, counts))
            if (
                "anomaly_higher_confidence" in flag_counts
                and flag_counts["anomaly_higher_confidence"] / smooth_size >= vote_threshold
            ):
                new_flags[i] = "anomaly_higher_confidence"

        return cast(list, new_flags[start_idx:end_idx].tolist())

    @sentry_sdk.trace
    def smooth(
        self,
        flags: list,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig,
        smooth_size: int = 0,
        vote_threshold: float = 0.5,
    ) -> list:
        """
        Smooth flags using voting threshold and dynamic window size
        """
        slices = self._get_anomalous_slices(flags, ad_config.time_period, algo_config)
        for start_idx, end_idx in slices:
            flags[start_idx:end_idx] = self._batch_smooth_flags(
                flags, start_idx, end_idx, ad_config, algo_config, smooth_size, vote_threshold
            )
        return flags


class MajorityVoteStreamFlagSmoother(FlagSmoother):
    """
    This class smooths flags using majority voting with a dynamic smoothing window size for stream data
    """

    def _stream_smooth_flags(
        self,
        orig_flags: list,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig,
        vote_threshold: float = 0.5,
        cur_flag: list = [],
    ) -> list:
        """
        Use original flags to smooth by majority (or threshold) voting with a small window.
        """

        orig_flags = orig_flags[-algo_config.stream_smooth_context_sizes[ad_config.time_period] :]

        new_flag = cur_flag
        values, counts = np.unique(orig_flags, return_counts=True)
        flag_counts = dict(zip(values, counts))
        if (
            "anomaly_higher_confidence" in flag_counts
            and flag_counts["anomaly_higher_confidence"] / len(orig_flags) >= vote_threshold
        ):
            new_flag = ["anomaly_higher_confidence"]

        return new_flag

    def smooth(
        self,
        original_flags: list,
        ad_config: AnomalyDetectionConfig,
        algo_config: AlgoConfig,
        smooth_size: int = 0,
        vote_threshold: float = 0.25,
        cur_flag: list = [],
    ) -> list:
        """
        Smooth flags using voting threshold and dynamic window size
        """
        smoothed_flag = self._stream_smooth_flags(
            original_flags, ad_config, algo_config, vote_threshold, cur_flag
        )
        return smoothed_flag
