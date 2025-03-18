import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from seer.anomaly_detection.detectors import (
    CombinedAnomalyScorer,
    FlagsAndScores,
    MPUtils,
    SuSSWindowSizeSelector,
)
from seer.anomaly_detection.detectors.anomaly_detectors import (
    MPBatchAnomalyDetector,
    MPStreamAnomalyDetector,
)
from seer.anomaly_detection.models import (
    AlgoConfig,
    ConfidenceLevel,
    MPTimeSeriesAnomaliesSingleWindow,
)
from seer.anomaly_detection.models.external import AnomalyDetectionConfig
from seer.anomaly_detection.models.timeseries import TimeSeries
from seer.exceptions import ServerError
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestMPBatchAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MPBatchAnomalyDetector()

        self.config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="up", expected_seasonality="auto"
        )  # TODO: Placeholder values as not used in detection yet

        self.algo_config = AlgoConfig(
            mp_ignore_trivial=False,
            mp_normalize=False,
            mp_fixed_window_size=10,
            prophet_uncertainty_samples=25,
            prophet_mcmc_samples=0,
            return_thresholds=False,
            return_predicted_range=False,
        )
        self.ws_selector = MagicMock()
        self.scorer = MagicMock()
        self.mp_utils = MagicMock()

    @patch("stumpy.stump")
    def test_compute_matrix_profile(self, mock_stump):

        # Mock to return dummy values
        mock_stump.return_value = np.array([1, 2, 3, 4])
        self.scorer.batch_score = MagicMock(
            return_value=FlagsAndScores(
                scores=[0.1, 6.5, 4.8, 0.2],
                flags=["none", "anomaly_higher_confidence", "anomaly_higher_confidence", "none"],
                thresholds=[],
                confidence_levels=[
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                ],
            )
        )

        syntheitc_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )
        timeseries = syntheitc_data.timeseries
        timestamps = syntheitc_data.timestamps
        mp_dists = syntheitc_data.mp_dists
        window_sizes = syntheitc_data.window_sizes

        ts_values, ts_timestamps, mp_dist_baseline, window_size = (
            timeseries[0],
            timestamps[0],
            mp_dists[0],
            window_sizes[0],
        )
        ts = TimeSeries(timestamps=ts_timestamps, values=ts_values)

        self.ws_selector.optimal_window_size = MagicMock(return_value=window_size)
        self.mp_utils.get_mp_dist_from_mp = MagicMock(return_value=mp_dist_baseline)

        result = self.detector._compute_matrix_profile(
            ts,
            self.config,
            ws_selector=self.ws_selector,
            algo_config=self.algo_config,
            scorer=self.scorer,
            mp_utils=self.mp_utils,
        )

        assert isinstance(result, MPTimeSeriesAnomaliesSingleWindow)
        assert isinstance(result.flags, list)
        assert result.scores == [0.1, 6.5, 4.8, 0.2]
        assert isinstance(result.scores, list)
        assert result.flags == [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
        ]
        assert isinstance(result.matrix_profile, np.ndarray)
        assert isinstance(result.window_size, int)
        mock_stump.assert_called_once()
        self.scorer.batch_score.assert_called_once()
        self.ws_selector.optimal_window_size.assert_called_once()
        self.mp_utils.get_mp_dist_from_mp.assert_called_once()

    def test_invalid_window_size(self):

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        timeseries = loaded_synthetic_data.timeseries
        timestamps = loaded_synthetic_data.timestamps

        ts = TimeSeries(timestamps=timestamps[0], values=timeseries[0])

        self.ws_selector.optimal_window_size = MagicMock(return_value=-1)

        with self.assertRaises(ServerError, msg="Invalid window size"):
            self.detector._compute_matrix_profile(
                ts,
                self.config,
                ws_selector=self.ws_selector,
                algo_config=self.algo_config,
                scorer=self.scorer,
                mp_utils=self.mp_utils,
            )

    def test_no_values(self):
        ts = TimeSeries(timestamps=np.array([]), values=np.array([]))
        with self.assertRaises(ServerError, msg="No values to detect anomalies for"):
            self.detector._compute_matrix_profile(
                ts, self.config, self.ws_selector, self.algo_config, self.scorer, self.mp_utils
            )

    def test_timestamps_and_values_not_same_length(self):
        ts = TimeSeries(timestamps=np.array([1, 2, 3]), values=np.array([1, 2]))
        with self.assertRaises(ServerError, msg="Timestamps and values are not of the same length"):
            self.detector._compute_matrix_profile(
                ts, self.config, self.ws_selector, self.algo_config, self.scorer, self.mp_utils
            )


class TestMPStreamAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MPStreamAnomalyDetector(
            history_timestamps=np.array([1, 2, 3, 4]),
            history_values=np.array([1.0, 2.0, 3.0, 4.0]),
            history_mp=np.array([[0.3, 0.3, 0.3, 0.3], [0.4, 0.5, 0.6, 0.7]]),
            window_size=3,
            original_flags=["none", "none", "none"],
        )
        self.timeseries = TimeSeries(
            timestamps=np.array([1, 2, 3]), values=np.array([1.1, 2.1, 3.1])
        )
        self.config = AnomalyDetectionConfig(
            time_period=15, sensitivity="medium", direction="both", expected_seasonality="auto"
        )  # TODO: Placeholder values as not used in detection yet
        self.algo_config = AlgoConfig(
            mp_ignore_trivial=False,
            mp_normalize=False,
            mp_fixed_window_size=10,
            prophet_uncertainty_samples=25,
            prophet_mcmc_samples=0,
            return_thresholds=False,
            return_predicted_range=False,
        )

    @patch("stumpy.stumpi")
    @patch("seer.anomaly_detection.detectors.MPScorer")
    @patch("seer.anomaly_detection.detectors.MPUtils")
    def test_detect(self, MockMPUtils, MockMPScorer, MockStumpi):
        mock_stream = MagicMock()
        MockStumpi.return_value = mock_stream
        mock_scorer = MockMPScorer.return_value
        mock_utils = MockMPUtils.return_value

        mock_stream.P_ = np.array([0.1, 0.2])
        mock_stream.I_ = np.array([0, 1])
        mock_stream.left_I_ = np.array([0, 1])
        mock_stream.T_ = np.array([1.1, 2.1])

        mock_utils.get_mp_dist_from_mp.return_value = np.array([0.1, 0.2])

        mock_scorer.stream_score.return_value = FlagsAndScores(
            scores=[0.5], flags=["none"], thresholds=[], confidence_levels=[ConfidenceLevel.MEDIUM]
        )

        anomalies = self.detector.detect(
            timeseries=self.timeseries,
            ad_config=self.config,
            algo_config=self.algo_config,
            scorer=mock_scorer,
            mp_utils=mock_utils,
        )

        assert isinstance(anomalies, MPTimeSeriesAnomaliesSingleWindow)
        assert isinstance(anomalies.flags, list)
        assert isinstance(anomalies.scores, list)
        assert isinstance(anomalies.matrix_profile, np.ndarray)
        assert isinstance(anomalies.window_size, int)
        assert len(anomalies.scores) == 3
        assert len(anomalies.flags) == 3
        assert len(anomalies.matrix_profile) == 3
        mock_scorer.stream_score.assert_called()
        mock_stream.update.assert_called()

    def _detect_anomalies(
        self, history_ts, stream_ts, history_ts_timestamps=None, history_mp=None, window_size=None
    ):
        batch_detector = MPBatchAnomalyDetector()
        history_ts_timestamps = (
            np.arange(1.0, len(history_ts) + 1)
            if history_ts_timestamps is None
            else history_ts_timestamps
        )

        if history_mp is None:
            batch_anomalies = batch_detector._compute_matrix_profile(
                timeseries=TimeSeries(
                    timestamps=history_ts_timestamps, values=np.array(history_ts)
                ),
                ad_config=self.config,
                ws_selector=SuSSWindowSizeSelector(),
                algo_config=self.algo_config,
                scorer=CombinedAnomalyScorer(),
                mp_utils=MPUtils(),
            )
            history_mp = batch_anomalies.matrix_profile
            window_size = batch_anomalies.window_size
        stream_detector = MPStreamAnomalyDetector(
            history_timestamps=np.array(history_ts_timestamps),
            history_values=np.array(history_ts),
            history_mp=history_mp,
            window_size=window_size,
            original_flags=["none", "none", "none"],
        )
        stream_ts_timestamps = np.array(list(range(1, len(stream_ts) + 1))) + len(history_ts)
        stream_anomalies = stream_detector.detect(
            timeseries=TimeSeries(timestamps=stream_ts_timestamps, values=np.array(stream_ts)),
            ad_config=self.config,
            scorer=CombinedAnomalyScorer(),
            mp_utils=MPUtils(),
        )

        return batch_anomalies, stream_anomalies

    def test_stream_detect_spiked_history_spiked_stream_long_ts(self):
        history_ts = [0.5] * 200
        history_ts[-115] = 1.0
        stream_ts = [0.5, 0.5, 2.5, 2.5, *[0.5] * 10]
        expected_stream_flags = [
            "none",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
        ]
        _, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        assert stream_anomalies.flags == expected_stream_flags

    def test_stream_detect_spiked_history_spiked_stream(self):
        history_ts = [0.5] * 20
        history_ts[-15] = 1.0  # Spiked history
        stream_ts = [0.5, 0.5, 5.0, 5.0, *[0.5] * 10]  # Spiked stream
        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        expected_stream_flags = [
            "none",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
        ]
        assert history_anomalies.window_size == 3
        assert stream_anomalies.flags == expected_stream_flags

    def test_stream_detect_flat_history_flat_stream(self):
        history_ts = [0.5] * 200  # Flat history
        stream_ts = [0.5] * 10  # Flat stream
        expected_stream_flags = ["none"] * len(stream_ts)

        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        assert history_anomalies.window_size == 3
        assert stream_anomalies.flags == expected_stream_flags

    def test_stream_detect_flat_history_spiked_stream(self):
        history_ts = [0.5] * 200  # Flat history
        stream_ts = [0.5, 0.5, 3.0, 3.0, *[0.5] * 12]  # Spiked stream
        expected_stream_flags = [
            "none",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
            "none",
        ]

        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        assert history_anomalies.window_size == 3
        assert stream_anomalies.flags == expected_stream_flags

    def test_stream_detect_spiked_history_flat_stream(self):
        history_ts = [0.5] * 200
        history_ts[-15] = 1.0  # Spiked history
        stream_ts = [0.5] * 10  # Flat stream
        expected_stream_flags = ["none"] * 10

        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        assert history_anomalies.window_size == 132
        assert stream_anomalies.flags == expected_stream_flags

    def test_history_values_and_matrix_profile_not_same_length(self):
        history_ts = [0.5] * 200
        history_mp = np.array([0.1, 0.2, 0.3, 0.4])
        stream_ts = [0.5] * 10
        with self.assertRaises(ServerError, msg="Matrix profile is not of the correct length"):
            self._detect_anomalies(history_ts, stream_ts, history_mp=history_mp, window_size=3)

    def test_no_streamed_values(self):
        history_ts = [0.5] * 200
        stream_ts = []
        with self.assertRaises(ServerError, msg="No values to detect anomalies for"):
            self._detect_anomalies(history_ts, stream_ts)

    def test_history_timestamps_and_values_not_same_length(self):
        history_ts = [0.5] * 200
        history_ts_timestamps = np.arange(1, len(history_ts))  # one off error
        stream_ts = [0.5] * 10
        original_flags = ["none"] * 200
        with self.assertRaises(
            ServerError, msg="History values and timestamps are not of the same length"
        ):
            stream_detector = MPStreamAnomalyDetector(
                history_timestamps=np.array(history_ts_timestamps),
                history_values=np.array(history_ts),
                history_mp=np.array([0.1, 0.2, 0.3, 0.4]),
                window_size=3,
                original_flags=original_flags,
            )
            stream_detector.detect(
                timeseries=TimeSeries(
                    timestamps=np.arange(1.0, len(stream_ts) + 1), values=np.array(stream_ts)
                ),
                ad_config=self.config,
                scorer=CombinedAnomalyScorer(),
                mp_utils=MPUtils(),
            )
