import time
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest
import stumpy
from scipy import stats

from seer.anomaly_detection.detectors import MPUtils, WindowSizeSelector
from seer.anomaly_detection.detectors.mp_boxcox_scorer import MPBoxCoxScorer
from seer.anomaly_detection.models import (
    AlgoConfig,
    AnomalyDetectionConfig,
    PointLocation,
    RelativeLocation,
    Threshold,
    ThresholdType,
)
from seer.dependency_injection import resolve
from seer.exceptions import ClientError, ServerError
from tests.seer.anomaly_detection.test_utils import test_data_with_cycles


@pytest.fixture
def box_cox_scorer():
    return MPBoxCoxScorer()


@pytest.fixture
def mock_location_detector():
    detector = Mock()
    detector.detect.return_value = RelativeLocation(
        location=PointLocation.UP,
        thresholds=[Threshold(type=ThresholdType.PREDICTION, upper=10.0, lower=5.0)],
    )
    return detector


@pytest.fixture
def mock_slow_location_detector():
    def slow_function(streamed_value, streamed_timestamp, history_values, history_timestamps):
        time.sleep(0.05)  # Simulate a 50ms delay.
        return None

    detector = Mock()
    detector.detect.side_effect = slow_function
    return detector


@pytest.fixture
def basic_ad_config():
    return AnomalyDetectionConfig(
        time_period=15,
        sensitivity="medium",
        direction="both",
        expected_seasonality="auto",
    )


class TestBoxCoxScorer:
    @mock.patch("scipy.stats.boxcox")
    def test_box_cox_log_transform(self, mock_boxcox, box_cox_scorer):
        # Test with positive values
        mock_boxcox.return_value = (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.0)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformed, _, _ = box_cox_scorer._box_cox_transform(x)
        assert len(transformed) == len(x)
        expected = np.log(x)
        np.testing.assert_array_almost_equal(transformed, expected)
        assert mock_boxcox.call_count == 1

        # Test with negative values (should shift to positive)
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        transformed, _, _ = box_cox_scorer._box_cox_transform(x)
        assert len(transformed) == len(x)

        # Should shift by min + 1 before transform
        expected = np.log(x - np.min(x) + 1)
        np.testing.assert_array_almost_equal(transformed, expected)
        assert mock_boxcox.call_count == 2

    def test_box_cox_transform(self, box_cox_scorer):
        # Test with positive values
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformed, _, _ = box_cox_scorer._box_cox_transform(x)
        assert len(transformed) == len(x)
        expected, bc_lambda = stats.boxcox(x)
        np.testing.assert_array_almost_equal(transformed, expected)

        # Test with negative values (should shift to positive)
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        transformed, _, _ = box_cox_scorer._box_cox_transform(x)
        assert len(transformed) == len(x)

        # Should shift by min + 1 before transform
        expected, _ = stats.boxcox(np.array([1.0, 2.0, 3.0, 4.0]))

        # expected = np.log(x - np.min(x) + 1)
        expected, bc_lambda = stats.boxcox(x - np.min(x) + 1)
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_get_z_scores(self, box_cox_scorer):
        values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])  # Include outlier
        z_scores, threshold, std, _ = box_cox_scorer._get_z_scores(values, "medium")

        assert len(z_scores) == len(values)
        assert threshold == box_cox_scorer.z_score_thresholds["medium"]
        assert std > 0

        # Test invalid sensitivity
        with pytest.raises(ClientError):
            box_cox_scorer._get_z_scores(values, "invalid")

    def test_batch_score_normal_distribution(
        self, box_cox_scorer, mock_location_detector, basic_ad_config
    ):
        # Generate normally distributed data with one obvious outlier
        values = np.concatenate([np.random.normal(10, 2, 99), [20.0]])
        mp_dist = np.concatenate([np.random.normal(10, 2, 99), [20.0]])  # Not used by BoxCoxScorer
        timestamps = np.arange(len(values), dtype=np.float64)

        result = box_cox_scorer.batch_score(
            values=values,
            timestamps=timestamps,
            mp_dist=mp_dist,
            ad_config=basic_ad_config,
            window_size=10,
            location_detector=mock_location_detector,
        )

        assert len(result.flags) == len(values)
        assert len(result.scores) == len(values)
        assert len(result.thresholds) == len(values)

        # Last point should be flagged as anomaly
        assert result.flags[-1] == "anomaly_higher_confidence"
        assert result.scores[-1] > box_cox_scorer.z_score_thresholds["medium"]

    def test_batch_score_constant_data(
        self, box_cox_scorer, mock_location_detector, basic_ad_config
    ):
        # Test with constant data (std = 0)
        mp_dist = np.ones(100)
        timestamps = np.arange(len(mp_dist), dtype=np.float64)
        values = np.ones(100)

        with pytest.raises(ValueError) as ex:
            box_cox_scorer.batch_score(
                values=values,
                timestamps=timestamps,
                mp_dist=mp_dist,
                ad_config=basic_ad_config,
                window_size=10,
                location_detector=mock_location_detector,
            )
        assert str(ex.value) == "Data must not be constant."

    def test_stream_score(self, box_cox_scorer, mock_location_detector, basic_ad_config):
        # Test streaming with normal history and anomalous new point
        history_mp_dist = np.random.normal(10, 2, 99)
        history_timestamps = np.arange(len(history_mp_dist), dtype=np.float64)
        history_values = np.random.normal(10, 2, 99)
        streamed_mp_dist = 20.0  # Obvious outlier
        streamed_timestamp = float(len(history_values))
        streamed_value = 20.0

        result = box_cox_scorer.stream_score(
            streamed_value=streamed_value,
            streamed_timestamp=streamed_timestamp,
            streamed_mp_dist=streamed_mp_dist,
            history_values=history_values,
            history_timestamps=history_timestamps,
            history_mp_dist=history_mp_dist,
            ad_config=basic_ad_config,
            window_size=10,
            location_detector=mock_location_detector,
        )

        assert len(result.flags) == 1
        assert len(result.scores) == 1
        assert len(result.thresholds) == 1
        assert result.flags[0] == "anomaly_higher_confidence"
        assert result.scores[0] > box_cox_scorer.z_score_thresholds["medium"]

    def test_direction_handling(self, box_cox_scorer, mock_location_detector):
        # Test different direction configurations
        mp_dist = np.arange(1.0, 50.0, 1.0)
        mp_dist[-1] = 200.0  # Last value is anomalous
        timestamps = np.arange(len(mp_dist), dtype=np.float64)
        values = np.arange(1.0, 50.0, 1.0)
        values[-1] = 200.0  # Last value is anomalous
        # Test "up" direction with upward anomaly
        up_config = AnomalyDetectionConfig(
            time_period=15,
            sensitivity="high",
            direction="up",
            expected_seasonality="auto",
        )
        result = box_cox_scorer.batch_score(
            values=values,
            timestamps=timestamps,
            mp_dist=mp_dist,
            ad_config=up_config,
            window_size=10,
            location_detector=mock_location_detector,
        )
        assert result.flags[-1] == "anomaly_higher_confidence"

        # Test "down" direction with upward anomaly
        mock_location_detector.detect.return_value = RelativeLocation(
            location=PointLocation.UP,
            thresholds=[],
        )
        down_config = AnomalyDetectionConfig(
            time_period=15,
            sensitivity="medium",
            direction="down",
            expected_seasonality="auto",
        )
        result = box_cox_scorer.batch_score(
            values=values,
            timestamps=timestamps,
            mp_dist=mp_dist,
            ad_config=down_config,
            window_size=10,
            location_detector=mock_location_detector,
        )
        assert result.flags[-1] == "none"

    def test_sensitivity_levels(self, box_cox_scorer, mock_location_detector):
        # Test different sensitivity levels
        values = np.array([1.0, 2.0, 3.0, 4.0, 6.0])  # Last value is mildly anomalous
        timestamps = np.arange(len(values), dtype=np.float64)
        mp_dist = np.zeros_like(values)

        # Test high sensitivity
        high_config = AnomalyDetectionConfig(
            time_period=15,
            sensitivity="high",
            direction="both",
            expected_seasonality="auto",
        )
        result = box_cox_scorer.batch_score(
            values=values,
            timestamps=timestamps,
            mp_dist=mp_dist,
            ad_config=high_config,
            window_size=10,
            location_detector=mock_location_detector,
        )
        high_anomaly_count = sum(1 for flag in result.flags if flag != "none")

        # Test low sensitivity
        low_config = AnomalyDetectionConfig(
            time_period=15,
            sensitivity="low",
            direction="both",
            expected_seasonality="auto",
        )
        result = box_cox_scorer.batch_score(
            values=values,
            timestamps=timestamps,
            mp_dist=mp_dist,
            ad_config=low_config,
            window_size=10,
            location_detector=mock_location_detector,
        )
        low_anomaly_count = sum(1 for flag in result.flags if flag != "none")

        # High sensitivity should detect more anomalies than low sensitivity
        assert high_anomaly_count >= low_anomaly_count

    def test_abandon_batch_detection_if_time_budget_is_exceeded(self, mock_slow_location_detector):
        box_cox_scorer = MPBoxCoxScorer()

        mp_utils = resolve(MPUtils)
        ws_selector = resolve(WindowSizeSelector)
        ad_config = AnomalyDetectionConfig(
            time_period=60, sensitivity="high", direction="up", expected_seasonality="auto"
        )
        algo_config = resolve(AlgoConfig)
        df = test_data_with_cycles(num_days=29, num_anomalous=15)
        window_size = ws_selector.optimal_window_size(df["value"].values)
        mp = stumpy.stump(
            df["value"].values,
            m=max(3, window_size),
            ignore_trivial=algo_config.mp_ignore_trivial,
            normalize=False,
        )

        # We do not normalize the matrix profile here as normalizing during stream detection later is not straighforward.
        mp_dist = mp_utils.get_mp_dist_from_mp(mp, pad_to_len=len(df["value"].values))
        time_budget_ms = 100
        with pytest.raises(ServerError) as ex:
            box_cox_scorer.batch_score(
                df["value"].values,
                df["timestamp"].values,
                mp_dist,
                ad_config=ad_config,
                algo_config=algo_config,
                window_size=window_size,
                time_budget_ms=time_budget_ms,
                location_detector=mock_slow_location_detector,
            )
            assert mock_slow_location_detector.call_count >= 2
            assert mock_slow_location_detector.call_count <= 10
        # Since slow func sleeps for 50 ms and timeout is 100ms, location detection should be called at least twice and upto 10 which is the batch size.
        assert "Batch detection took too long" in str(ex.value)
