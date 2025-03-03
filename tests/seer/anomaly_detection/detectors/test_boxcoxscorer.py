from unittest import mock

import numpy as np
import pytest
from scipy import stats

from seer.anomaly_detection.detectors.mp_boxcox_scorer import MPBoxCoxScorer
from seer.anomaly_detection.models import AnomalyDetectionConfig
from seer.exceptions import ClientError


@pytest.fixture
def box_cox_scorer():
    return MPBoxCoxScorer()


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

    def test_batch_score_normal_distribution(self, box_cox_scorer, basic_ad_config):
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
        )

        assert len(result.flags) == len(values)
        assert len(result.scores) == len(values)
        assert len(result.thresholds) == len(values)

        # Last point should be flagged as anomaly
        assert result.flags[-1] == "anomaly_higher_confidence"
        assert result.scores[-1] > box_cox_scorer.z_score_thresholds["medium"]

    def test_batch_score_constant_data(self, box_cox_scorer, basic_ad_config):
        # Test with constant data (std = 0)
        mp_dist = np.ones(100)
        timestamps = np.arange(len(mp_dist), dtype=np.float64)
        values = np.ones(100)

        result = box_cox_scorer.batch_score(
            values=values,
            timestamps=timestamps,
            mp_dist=mp_dist,
            ad_config=basic_ad_config,
            window_size=10,
        )

        assert len(result.flags) == len(values)
        assert len(result.scores) == len(values)
        assert len(result.thresholds) == len(values)

        # Last point should be flagged as anomaly
        assert result.flags[-1] == "none"
        assert result.scores[-1] == 0

    def test_stream_score(self, box_cox_scorer, basic_ad_config):
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
        )

        assert len(result.flags) == 1
        assert len(result.scores) == 1
        assert len(result.thresholds) == 1
        assert result.flags[0] == "anomaly_higher_confidence"
        assert result.scores[0] > box_cox_scorer.z_score_thresholds["medium"]

    def test_sensitivity_levels(self, box_cox_scorer):
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
        )
        low_anomaly_count = sum(1 for flag in result.flags if flag != "none")

        # High sensitivity should detect more anomalies than low sensitivity
        assert high_anomaly_count >= low_anomaly_count
