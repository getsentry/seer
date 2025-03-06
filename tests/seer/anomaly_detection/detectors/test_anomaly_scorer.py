from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from seer.anomaly_detection.detectors.anomaly_scorer import CombinedAnomalyScorer
from seer.anomaly_detection.detectors.mp_scorers import FlagsAndScores
from seer.anomaly_detection.models import (
    AlgoConfig,
    AnomalyDetectionConfig,
    Threshold,
    ThresholdType,
)
from seer.exceptions import ServerError


class TestCombinedAnomalyScorer:
    @pytest.fixture
    def mp_flags_and_scores(self):
        return FlagsAndScores(
            flags=["none", "anomaly_higher_confidence", "none"],
            scores=[0.1, 0.9, 0.2],
            thresholds=[
                [
                    Threshold(
                        type=ThresholdType.PREDICTION,
                        timestamp=1672531200.0,
                        upper=0.5,
                        lower=0.5,
                    )
                ]
                * 3
            ],
        )

    @pytest.fixture
    def prophet_df(self):
        return pd.DataFrame(
            {
                "ds": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                "flag": ["none", "anomaly_higher_confidence", "none"],
                "score": [0.5, 2.5, 0.3],
                "y": [10.0, 20.0, 15.0],
                "yhat": [11.0, 12.0, 14.0],
                "yhat_lower": [9.0, 10.0, 12.0],
                "yhat_upper": [13.0, 14.0, 16.0],
            }
        )

    @pytest.fixture
    def sample_data(self):
        return {
            "values": np.array([10.0, 20.0, 15.0], dtype=np.float64),
            "timestamps": np.array(
                [1672531200.0, 1672617600.0, 1672704000.0], dtype=np.float64
            ),  # 2023-01-01, 02, 03
            "mp_dist": np.array([0.1, 0.9, 0.2], dtype=np.float64),
            "window_size": 3,
            "ad_config": AnomalyDetectionConfig(
                sensitivity="medium", direction="both", time_period=30, expected_seasonality="auto"
            ),
            "algo_config": AlgoConfig(
                mp_ignore_trivial=True,
                mp_normalize=False,
                prophet_uncertainty_samples=5,
                prophet_mcmc_samples=0,
                mp_fixed_window_size=10,
                return_thresholds=True,
                return_predicted_range=True,
            ),
        }

    @patch("seer.anomaly_detection.detectors.mp_boxcox_scorer.MPBoxCoxScorer.batch_score")
    @patch(
        "seer.anomaly_detection.detectors.prophet_scorer.ProphetScaledSmoothedScorer.batch_score"
    )
    def test_batch_score_with_prophet(
        self,
        mock_prophet_scorer_batch_score,
        mock_mp_scorer_batch_score,
        mp_flags_and_scores,
        prophet_df,
        sample_data,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()

        # Manually set up the mocked attributes
        mock_mp_scorer_batch_score.return_value = mp_flags_and_scores
        mock_prophet_scorer_batch_score.return_value = prophet_df

        # Test batch_score with both MP and Prophet data
        result = scorer.batch_score(
            values=sample_data["values"],
            timestamps=sample_data["timestamps"],
            mp_dist=sample_data["mp_dist"],
            prophet_df=prophet_df,
            ad_config=sample_data["ad_config"],
            window_size=sample_data["window_size"],
            algo_config=sample_data["algo_config"],
        )

        # Verify the MP scorer was called with correct parameters
        mock_mp_scorer_batch_score.assert_called_once()

        # Verify the Prophet scorer was called
        mock_prophet_scorer_batch_score.assert_called_once()

        # Check that we got a valid result
        assert isinstance(result, FlagsAndScores)
        assert len(result.flags) == 3
        assert len(result.scores) == 3
        assert len(result.thresholds) == 1
        assert len(result.thresholds[0]) == 3

    @patch("seer.anomaly_detection.detectors.mp_boxcox_scorer.MPBoxCoxScorer.batch_score")
    @patch(
        "seer.anomaly_detection.detectors.prophet_scorer.ProphetScaledSmoothedScorer.batch_score"
    )
    def test_batch_score_without_prophet(
        self,
        mock_prophet_scorer_batch_score,
        mock_mp_scorer_batch_score,
        mp_flags_and_scores,
        sample_data,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()
        mock_mp_scorer_batch_score.return_value = mp_flags_and_scores

        # Test batch_score with only MP data (no Prophet)
        result = scorer.batch_score(
            values=sample_data["values"],
            timestamps=sample_data["timestamps"],
            mp_dist=sample_data["mp_dist"],
            prophet_df=None,
            ad_config=sample_data["ad_config"],
            window_size=sample_data["window_size"],
            algo_config=sample_data["algo_config"],
        )

        # Verify the MP scorer was called
        mock_mp_scorer_batch_score.assert_called_once()

        # Verify the Prophet scorer was NOT called
        mock_prophet_scorer_batch_score.assert_not_called()

        # Check that we got the MP result directly
        assert result.flags == ["none", "anomaly_higher_confidence", "none"]
        assert result.scores == [0.1, 0.9, 0.2]
        assert result.thresholds == [
            [
                Threshold(
                    type=ThresholdType.PREDICTION, timestamp=1672531200.0, upper=0.5, lower=0.5
                ),
            ]
            * 3
        ]

    @patch("seer.anomaly_detection.detectors.mp_boxcox_scorer.MPBoxCoxScorer.batch_score")
    @patch(
        "seer.anomaly_detection.detectors.prophet_scorer.ProphetScaledSmoothedScorer.batch_score"
    )
    def test_batch_score_mp_error(
        self, mock_prophet_scorer_batch_score, mock_mp_scorer_batch_score, sample_data, prophet_df
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()

        # Manually set up the mocked attributes
        mock_mp_scorer_batch_score.return_value = None
        mock_prophet_scorer_batch_score.return_value = prophet_df
        with pytest.raises(ServerError, match="No flags and scores from MP scorer"):
            scorer.batch_score(
                values=sample_data["values"],
                timestamps=sample_data["timestamps"],
                mp_dist=sample_data["mp_dist"],
                prophet_df=prophet_df,
                ad_config=sample_data["ad_config"],
                window_size=sample_data["window_size"],
                algo_config=sample_data["algo_config"],
            )

    @patch("seer.anomaly_detection.detectors.mp_boxcox_scorer.MPBoxCoxScorer.stream_score")
    @patch(
        "seer.anomaly_detection.detectors.prophet_scorer.ProphetScaledSmoothedScorer.batch_score"
    )
    def test_stream_score_with_prophet(
        self,
        mock_prophet_scorer_batch_score,
        mock_mp_scorer_batch_score,
        mp_flags_and_scores,
        sample_data,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()
        prophet_df = pd.DataFrame(
            {
                "ds": [
                    sample_data["timestamps"][2],  # Ensure this matches the streamed timestamp
                    sample_data["timestamps"][2] + 3600,
                    sample_data["timestamps"][2] + 7200,
                ],
                "flag": ["none", "anomaly_higher_confidence", "none"],
                "score": [0.5, 2.5, 0.3],
                "y": [10.0, 20.0, 15.0],
                "yhat": [11.0, 12.0, 14.0],
                "yhat_lower": [9.0, 10.0, 12.0],
                "yhat_upper": [13.0, 14.0, 16.0],
            }
        )
        # Manually set up the mocked attributes
        mock_mp_scorer_batch_score.return_value = mp_flags_and_scores
        mock_prophet_scorer_batch_score.return_value = prophet_df

        # Test stream_score with both MP and Prophet data
        result = scorer.stream_score(
            streamed_value=15.0,
            streamed_timestamp=sample_data["timestamps"][2],
            streamed_mp_dist=sample_data["mp_dist"][2],
            history_values=sample_data["values"][:2],
            history_timestamps=sample_data["timestamps"][:2],
            history_mp_dist=sample_data["mp_dist"][:2],
            prophet_df=prophet_df,
            ad_config=sample_data["ad_config"],
            window_size=sample_data["window_size"],
            algo_config=sample_data["algo_config"],
        )

        # Verify the MP scorer was called
        mock_mp_scorer_batch_score.assert_called_once()

        # Verify the Prophet scorer was called
        mock_prophet_scorer_batch_score.assert_called_once()

        # Check that we got a valid result
        assert isinstance(result, FlagsAndScores)

    @patch("seer.anomaly_detection.detectors.mp_boxcox_scorer.MPBoxCoxScorer.stream_score")
    @patch(
        "seer.anomaly_detection.detectors.prophet_scorer.ProphetScaledSmoothedScorer.batch_score"
    )
    def test_stream_score_with_prophet_df_missing_timestamp(
        self,
        mock_prophet_scorer_batch_score,
        mock_mp_scorer_batch_score,
        mp_flags_and_scores,
        sample_data,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()
        prophet_df = pd.DataFrame(
            {
                "ds": [
                    sample_data["timestamps"][2]
                    + 100,  # Ensure this does not match the streamed timestamp
                    sample_data["timestamps"][2] + 3600,
                    sample_data["timestamps"][2] + 7200,
                ],
                "flag": ["none", "anomaly_higher_confidence", "none"],
                "score": [0.5, 2.5, 0.3],
                "y": [10.0, 20.0, 15.0],
                "yhat": [11.0, 12.0, 14.0],
                "yhat_lower": [9.0, 10.0, 12.0],
                "yhat_upper": [13.0, 14.0, 16.0],
            }
        )
        # Manually set up the mocked attributes
        mock_mp_scorer_batch_score.return_value = mp_flags_and_scores
        mock_prophet_scorer_batch_score.return_value = prophet_df

        # Test stream_score with both MP and Prophet data
        result = scorer.stream_score(
            streamed_value=15.0,
            streamed_timestamp=sample_data["timestamps"][2],
            streamed_mp_dist=sample_data["mp_dist"][2],
            history_values=sample_data["values"][:2],
            history_timestamps=sample_data["timestamps"][:2],
            history_mp_dist=sample_data["mp_dist"][:2],
            prophet_df=prophet_df,
            ad_config=sample_data["ad_config"],
            window_size=sample_data["window_size"],
            algo_config=sample_data["algo_config"],
        )

        # Verify the MP scorer was called
        mock_mp_scorer_batch_score.assert_called_once()

        # Verify the Prophet scorer was called
        mock_prophet_scorer_batch_score.assert_not_called()

        # Check that we got a valid result
        assert isinstance(result, FlagsAndScores)

    @patch("seer.anomaly_detection.detectors.mp_boxcox_scorer.MPBoxCoxScorer.stream_score")
    @patch(
        "seer.anomaly_detection.detectors.prophet_scorer.ProphetScaledSmoothedScorer.batch_score"
    )
    def test_stream_score_without_prophet(
        self,
        mock_prophet_scorer_batch_score,
        mock_mp_scorer_stream_score,
        mp_flags_and_scores,
        sample_data,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()

        # Manually set up the mocked attributes
        mock_mp_scorer_stream_score.return_value = mp_flags_and_scores

        # Test stream_score with only MP data (no Prophet)
        result = scorer.stream_score(
            streamed_value=15.0,
            streamed_timestamp=sample_data["timestamps"][2],
            streamed_mp_dist=sample_data["mp_dist"][2],
            history_values=sample_data["values"][:2],
            history_timestamps=sample_data["timestamps"][:2],
            history_mp_dist=sample_data["mp_dist"][:2],
            prophet_df=None,
            ad_config=sample_data["ad_config"],
            window_size=sample_data["window_size"],
            algo_config=sample_data["algo_config"],
        )

        # Verify the MP scorer was called
        mock_mp_scorer_stream_score.assert_called_once()

        # Verify the Prophet scorer was NOT called
        mock_prophet_scorer_batch_score.assert_not_called()

        # Check that we got the MP result directly
        assert result == mp_flags_and_scores

    @patch("seer.anomaly_detection.detectors.mp_boxcox_scorer.MPBoxCoxScorer.stream_score")
    @patch(
        "seer.anomaly_detection.detectors.prophet_scorer.ProphetScaledSmoothedScorer.batch_score"
    )
    def test_stream_score_timestamp_not_in_prophet(
        self,
        mock_prophet_scorer_stream_score,
        mock_mp_scorer_stream_score,
        mp_flags_and_scores,
        sample_data,
        prophet_df,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()

        # Manually set up the mocked attributes
        mock_mp_scorer_stream_score.return_value = mp_flags_and_scores

        mock_prophet_scorer_stream_score.return_value = prophet_df

        # Test stream_score with timestamp not in prophet_df
        unknown_timestamp = np.float64(1672790400.0)  # 2023-01-04

        result = scorer.stream_score(
            streamed_value=15.0,
            streamed_timestamp=unknown_timestamp,
            streamed_mp_dist=sample_data["mp_dist"][2],
            history_values=sample_data["values"][:2],
            history_timestamps=sample_data["timestamps"][:2],
            history_mp_dist=sample_data["mp_dist"][:2],
            prophet_df=prophet_df,
            ad_config=sample_data["ad_config"],
            window_size=sample_data["window_size"],
            algo_config=sample_data["algo_config"],
        )

        # Verify we got the MP result directly
        assert result == mp_flags_and_scores
        mock_mp_scorer_stream_score.assert_called_once()
        mock_prophet_scorer_stream_score.assert_not_called()

    def test_mp_overrides_prophet(
        self,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()
        mp_flags_and_scores = FlagsAndScores(
            flags=["none", "anomaly_higher_confidence", "none"],
            scores=[0.1, 0.9, 0.2],
            thresholds=[
                [
                    Threshold(
                        type=ThresholdType.PREDICTION,
                        timestamp=1672531200.0,
                        upper=0.5,
                        lower=0.5,
                    )
                ]
                * 3
            ],
        )
        # Test merging MP and Prophet results
        timestamps = np.array(
            [1672531200.0, 1672617600.0, 1672704000.0], dtype=np.float64
        )  # 2023-01-01, 02, 03

        prophet_predictions = pd.DataFrame(
            {
                "ds": timestamps,
                "flag": ["none", "none", "none"],
                "score": [0.5, 0.2, 0.3],
                "y": [10.0, 10.0, 15.0],
                "yhat": [11.0, 12.0, 14.0],
                "yhat_lower": [9.0, 10.0, 12.0],
                "yhat_upper": [13.0, 14.0, 16.0],
            }
        )

        ad_config = AnomalyDetectionConfig(
            sensitivity="medium", direction="both", time_period=30, expected_seasonality="auto"
        )

        result = scorer._merge_prophet_mp_results(
            timestamps=timestamps,
            mp_distances=np.array([0.0, 2.0, 0.0]),
            mp_flags_and_scores=mp_flags_and_scores,
            prophet_predictions=prophet_predictions,
            ad_config=ad_config,
        )

        # Check that we got a valid result
        assert isinstance(result, FlagsAndScores)
        assert len(result.flags) == 3

        # For the seconde point, MP should override Prophet
        assert result.flags[1] == "anomaly_higher_confidence"

    def test_prophet_negative_overrides_mp(
        self,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()
        mp_flags_and_scores = FlagsAndScores(
            flags=["none", "anomaly_higher_confidence", "none"],
            scores=[0.1, 0.9, 0.2],
            thresholds=[
                [
                    Threshold(
                        type=ThresholdType.PREDICTION,
                        timestamp=1672531200.0,
                        upper=0.5,
                        lower=0.5,
                    )
                ]
                * 3
            ],
        )
        # Test merging MP and Prophet results
        timestamps = np.array(
            [1672531200.0, 1672617600.0, 1672704000.0], dtype=np.float64
        )  # 2023-01-01, 02, 03

        prophet_predictions = pd.DataFrame(
            {
                "ds": timestamps,
                "flag": ["none", "none", "none"],
                "score": [0.5, 2.5, 0.3],
                "y": [10.0, 20.0, 15.0],
                "yhat": [11.0, 12.0, 14.0],
                "yhat_lower": [9.0, 10.0, 12.0],
                "yhat_upper": [13.0, 14.0, 16.0],
            }
        )

        ad_config = AnomalyDetectionConfig(
            sensitivity="medium", direction="both", time_period=30, expected_seasonality="auto"
        )

        result = scorer._merge_prophet_mp_results(
            timestamps=timestamps,
            mp_distances=np.array([0.0, 0.0, 0.0]),
            mp_flags_and_scores=mp_flags_and_scores,
            prophet_predictions=prophet_predictions,
            ad_config=ad_config,
        )

        # Check that we got a valid result
        assert isinstance(result, FlagsAndScores)
        assert len(result.flags) == 3

        # For the second point, both MP and Prophet have high confidence anomalies
        assert result.flags[1] == "none"

    def test_mp_and_prophet_both_high_confidence_anomaly(
        self,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()
        mp_flags_and_scores = FlagsAndScores(
            flags=["none", "anomaly_higher_confidence", "none"],
            scores=[0.1, 0.9, 0.2],
            thresholds=[
                [
                    Threshold(
                        type=ThresholdType.PREDICTION,
                        timestamp=1672531200.0,
                        upper=0.5,
                        lower=0.5,
                    )
                ]
                * 3
            ],
        )
        # Test merging MP and Prophet results
        timestamps = np.array(
            [1672531200.0, 1672617600.0, 1672704000.0], dtype=np.float64
        )  # 2023-01-01, 02, 03

        prophet_predictions = pd.DataFrame(
            {
                "ds": timestamps,
                "flag": ["none", "anomaly_higher_confidence", "none"],
                "score": [0.5, 2.5, 0.3],
                "y": [10.0, 20.0, 15.0],
                "yhat": [11.0, 12.0, 14.0],
                "yhat_lower": [9.0, 10.0, 12.0],
                "yhat_upper": [13.0, 14.0, 16.0],
            }
        )

        ad_config = AnomalyDetectionConfig(
            sensitivity="medium", direction="both", time_period=30, expected_seasonality="auto"
        )

        result = scorer._merge_prophet_mp_results(
            timestamps=timestamps,
            mp_distances=np.array([0.0, 0.0, 0.0]),
            mp_flags_and_scores=mp_flags_and_scores,
            prophet_predictions=prophet_predictions,
            ad_config=ad_config,
        )

        # Check that we got a valid result
        assert isinstance(result, FlagsAndScores)
        assert len(result.flags) == 3

        # For the second point, both MP and Prophet have high confidence anomalies
        assert result.flags[1] == "anomaly_higher_confidence"

    def test_prophet_high_confidence_anomaly(
        self,
    ):
        # Create the scorer without calling the real __init__
        scorer = CombinedAnomalyScorer()
        mp_flags_and_scores = FlagsAndScores(
            flags=["none", "none", "none"],
            scores=[0.1, 0.9, 0.2],
            thresholds=[
                [
                    Threshold(
                        type=ThresholdType.PREDICTION,
                        timestamp=1672531200.0,
                        upper=0.5,
                        lower=0.5,
                    )
                ]
                * 3
            ],
        )

        # Test merging MP and Prophet results
        timestamps = np.array(
            [1672531200.0, 1672617600.0, 1672704000.0], dtype=np.float64
        )  # 2023-01-01, 02, 03

        prophet_predictions = pd.DataFrame(
            {
                "ds": timestamps,
                "flag": ["none", "anomaly_higher_confidence", "none"],
                "score": [0.5, 2.5, 0.3],
                "y": [10.0, 20.0, 15.0],
                "yhat": [11.0, 12.0, 14.0],
                "yhat_lower": [9.0, 10.0, 12.0],
                "yhat_upper": [13.0, 14.0, 16.0],
            }
        )

        ad_config = AnomalyDetectionConfig(
            sensitivity="medium", direction="both", time_period=30, expected_seasonality="auto"
        )

        result = scorer._merge_prophet_mp_results(
            timestamps=timestamps,
            mp_distances=np.array([0.0, 0.0, 0.0]),
            mp_flags_and_scores=mp_flags_and_scores,
            prophet_predictions=prophet_predictions,
            ad_config=ad_config,
        )

        # Check that we got a valid result
        assert isinstance(result, FlagsAndScores)
        assert len(result.flags) == 3

        # For the second point, both MP and Prophet have high confidence anomalies
        assert result.flags[1] == "anomaly_higher_confidence"

    def test_adjust_prophet_flag_for_location(self):
        ###
        # Tries all combinations of mp_flag, prev_mp_flag, prophet_flag, direction, y, yhat_lower, yhat_upper
        # and checks if the result is as expected
        # Overall its a brute force test. Below are the expected results for each combination
        #
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: none, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: none, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: none, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: none, prophet_flag: anomaly_higher_confidence, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: none, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: none
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: both, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: both, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: both, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: up, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: up, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: up, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: down, y: 15.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: down, y: 20.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence
        #   mp_flag: anomaly_higher_confidence, prev_mp_flag: anomaly_higher_confidence, prophet_flag: anomaly_higher_confidence, direction: down, y: 10.0, yhat_lower: 12.0, yhat_upper: 16.0: result: anomaly_higher_confidence        #
        ###
        scorer = CombinedAnomalyScorer()
        mp_flags = ["none", "anomaly_higher_confidence"]
        prophet_flags = ["none", "anomaly_higher_confidence"]
        prev_mp_flags = ["none", "anomaly_higher_confidence"]
        y_ll_ul = [
            (15.0, 12.0, 16.0),
            (20.0, 12.0, 16.0),
            (10.0, 12.0, 16.0),
        ]
        directions = ["both", "up", "down"]
        expected_results = [
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
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
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
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
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "anomaly_higher_confidence",
            "none",
            "none",
            "none",
            "anomaly_higher_confidence",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
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
        ]
        i = 0
        for mp_flag in mp_flags:
            for prev_mp_flag in prev_mp_flags:
                for prophet_flag in prophet_flags:
                    for direction in directions:
                        for y, yhat_lower, yhat_upper in y_ll_ul:
                            result = scorer._adjust_prophet_flag_for_location(
                                mp_flag=mp_flag,
                                prev_mp_flag=prev_mp_flag,
                                prophet_flag=prophet_flag,
                                y=y,
                                yhat=y,
                                yhat_lower=yhat_lower,
                                yhat_upper=yhat_upper,
                                direction=direction,
                            )
                            assert (
                                result == expected_results[i]
                            ), f"Expected {expected_results[i]} for mp_flag: {mp_flag}, prev_mp_flag: {prev_mp_flag}, prophet_flag: {prophet_flag}, direction: {direction}, y: {y}, yhat_lower: {yhat_lower}, yhat_upper: {yhat_upper}, got {result}"
                            i += 1
