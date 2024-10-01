import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from seer.anomaly_detection.detectors import (
    FlagsAndScores,
    MPCascadingScorer,
    MPConfig,
    MPUtils,
    SuSSWindowSizeSelector,
)
from seer.anomaly_detection.detectors.anomaly_detectors import (
    MPBatchAnomalyDetector,
    MPStreamAnomalyDetector,
)
from seer.anomaly_detection.models import MPTimeSeriesAnomalies
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

        self.mp_config = MPConfig(ignore_trivial=False, normalize_mp=False)
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
                thresholds=[0.0, 0.0, 0.0, 0.0],
            )
        )

        timeseries, mp_dists, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        ts_values, mp_dist_baseline, window_size = timeseries[0], mp_dists[0], window_sizes[0]
        ts = TimeSeries(timestamps=np.array([]), values=ts_values)

        self.ws_selector.optimal_window_size = MagicMock(return_value=window_size)
        self.mp_utils.get_mp_dist_from_mp = MagicMock(return_value=mp_dist_baseline)

        result = self.detector._compute_matrix_profile(
            ts,
            self.config,
            ws_selector=self.ws_selector,
            mp_config=self.mp_config,
            scorer=self.scorer,
            mp_utils=self.mp_utils,
        )

        assert isinstance(result, MPTimeSeriesAnomalies)
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

        timeseries, mp_dists, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )

        ts_values, _, _ = timeseries[0], mp_dists[0], window_sizes[0]
        ts = TimeSeries(timestamps=np.array([]), values=ts_values)

        self.ws_selector.optimal_window_size = MagicMock(return_value=-1)

        with self.assertRaises(ServerError, msg="Invalid window size"):
            self.detector._compute_matrix_profile(
                ts,
                self.config,
                ws_selector=self.ws_selector,
                mp_config=self.mp_config,
                scorer=self.scorer,
                mp_utils=self.mp_utils,
            )


class TestMPStreamAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MPStreamAnomalyDetector(
            base_timestamps=np.array([1, 2, 3]),
            base_values=np.array([1.0, 2.0, 3.0]),
            base_mp=np.array([0.1, 0.2, 0.3, 0.4]),
            window_size=2,
        )
        self.timeseries = TimeSeries(
            timestamps=np.array([1, 2, 3]), values=np.array([1.1, 2.1, 3.1])
        )
        self.config = AnomalyDetectionConfig(
            time_period=15, sensitivity="medium", direction="up", expected_seasonality="auto"
        )  # TODO: Placeholder values as not used in detection yet
        self.mp_config = MPConfig(ignore_trivial=False, normalize_mp=False)

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
            scores=[0.5], flags=["none"], thresholds=[0.0]
        )

        anomalies = self.detector.detect(self.timeseries, self.config, mock_scorer, mock_utils)

        assert isinstance(anomalies, MPTimeSeriesAnomalies)
        assert isinstance(anomalies.flags, list)
        assert isinstance(anomalies.scores, list)
        assert isinstance(anomalies.matrix_profile, np.ndarray)
        assert isinstance(anomalies.window_size, int)
        assert len(anomalies.scores) == 3
        assert len(anomalies.flags) == 3
        assert len(anomalies.matrix_profile) == 3
        mock_scorer.stream_score.assert_called()
        mock_stream.update.assert_called()

    def _detect_anomalies(self, history_ts, stream_ts):
        batch_detector = MPBatchAnomalyDetector()
        history_ts_timestamps = np.arange(1.0, len(history_ts) + 1)
        stream_ts_timestamps = np.array(list(range(1, len(stream_ts) + 1))) + len(history_ts)
        batch_anomalies = batch_detector._compute_matrix_profile(
            timeseries=TimeSeries(timestamps=history_ts_timestamps, values=np.array(history_ts)),
            config=self.config,
            ws_selector=SuSSWindowSizeSelector(),
            mp_config=self.mp_config,
            scorer=MPCascadingScorer(),
            mp_utils=MPUtils(),
        )
        stream_detector = MPStreamAnomalyDetector(
            base_timestamps=np.array(history_ts_timestamps),
            base_values=np.array(history_ts),
            base_mp=batch_anomalies.matrix_profile,
            window_size=batch_anomalies.window_size,
        )
        stream_anomalies = stream_detector.detect(
            timeseries=TimeSeries(timestamps=stream_ts_timestamps, values=np.array(stream_ts)),
            config=self.config,
            scorer=MPCascadingScorer(),
            mp_utils=MPUtils(),
        )
        return batch_anomalies, stream_anomalies

    def test_stream_detect_spiked_history_spiked_stream_long_ts(self):
        history_ts = [0.5] * 200
        history_ts[-115] = 1.0
        stream_ts = [0.5, 0.5, 1.2, *[0.5] * 10]
        expected_stream_flags = [
            "none",
            "none",
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
        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        assert history_anomalies.window_size == 90
        assert stream_anomalies.flags == expected_stream_flags

    def test_stream_detect_spiked_history_spiked_stream(self):
        history_ts = [0.5] * 20
        history_ts[-15] = 1.0  # Spiked history
        stream_ts = [0.5, 0.5, 3.5, *[0.5] * 10]  # Spiked stream
        expected_stream_flags = [
            "none",
            "none",
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
            "none",
        ]
        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
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
        stream_ts = [0.5, 0.5, 1.0, *[0.5] * 10]  # Spiked stream
        expected_stream_flags = [
            "none",
            "none",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
        ]

        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        assert history_anomalies.window_size == 3
        assert stream_anomalies.flags == expected_stream_flags

    def test_stream_detect_spliked_history_flat_stream(self):
        history_ts = [0.5] * 200
        history_ts[-15] = 1.0  # Spiked history
        stream_ts = [0.5] * 10  # Flat stream
        expected_stream_flags = ["none"] * 10

        history_anomalies, stream_anomalies = self._detect_anomalies(history_ts, stream_ts)
        assert history_anomalies.window_size == 132
        assert stream_anomalies.flags == expected_stream_flags
