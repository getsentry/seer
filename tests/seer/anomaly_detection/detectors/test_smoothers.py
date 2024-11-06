import unittest

from seer.anomaly_detection.detectors.mp_config import MPConfig
from seer.anomaly_detection.detectors.smoothers import (
    MajorityVoteBatchFlagSmoother,
    MajorityVoteStreamFlagSmoother,
)
from seer.anomaly_detection.models.external import AnomalyDetectionConfig


class TestMajorityVoteBatchFlagSmoother(unittest.TestCase):

    def setUp(self):
        self.smoother = MajorityVoteBatchFlagSmoother()

        self.ad_config = AnomalyDetectionConfig(
            time_period=60, sensitivity="low", direction="up", expected_seasonality="auto"
        )

        self.mp_config = MPConfig(ignore_trivial=False, normalize_mp=False)

    def test_smooth_no_anomalies(self):
        flags = ["none"] * 5
        result = self.smoother.smooth(flags, self.ad_config)
        assert result == flags

    def test_smooth_single_anomaly(self):
        flags = [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "anomaly_higher_confidence",
            "none",
            "none",
            "none",
        ]
        result = self.smoother.smooth(flags, self.ad_config)
        assert result == [
            "none",
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

    def test_smooth_multiple_anomalies(self):
        flags = ["none"] * 300
        flags[100:110] = ["anomaly_higher_confidence"] * 10
        flags[111:120] = ["anomaly_higher_confidence"] * 9

        flags[200:210] = ["anomaly_higher_confidence"] * 10
        flags[211:220] = ["anomaly_higher_confidence"] * 9

        result = self.smoother.smooth(flags, self.ad_config)
        expected = ["none"] * 300
        expected[100:120] = ["anomaly_higher_confidence"] * 20
        expected[200:220] = ["anomaly_higher_confidence"] * 20
        assert result == expected

    def test_smooth_with_custom_smooth_size(self):
        flags = [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]
        result = self.smoother.smooth(flags, self.ad_config, smooth_size=3)
        assert result == [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]

    def test_smooth_with_custom_vote_threshold(self):
        flags_success = [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]
        expected_success = [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]
        result_success = self.smoother.smooth(flags_success, self.ad_config, vote_threshold=0.75)
        assert result_success == expected_success

        flags_failure = [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]
        expected_failure = [
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]
        result_failure = self.smoother.smooth(flags_failure, self.ad_config, vote_threshold=0.75)
        assert result_failure == expected_failure

    def test_smooth_with_different_time_periods(self):
        flags = ["none"] * 300
        flags[100:110] = ["anomaly_higher_confidence"] * 10
        flags[112:120] = ["anomaly_higher_confidence"] * 8

        flags[200:210] = ["anomaly_higher_confidence"] * 10
        flags[212:220] = ["anomaly_higher_confidence"] * 8

        expected = ["none"] * 300
        expected[100:120] = ["anomaly_higher_confidence"] * 20
        expected[200:220] = ["anomaly_higher_confidence"] * 20

        ad_config_5 = AnomalyDetectionConfig(
            time_period=5, sensitivity="medium", direction="both", expected_seasonality="auto"
        )
        result_5 = self.smoother.smooth(flags, ad_config_5)
        assert result_5 == expected


class TestMajorityVoteStreamFlagSmoother(unittest.TestCase):

    def setUp(self):
        self.smoother = MajorityVoteStreamFlagSmoother()

        self.ad_config = AnomalyDetectionConfig(
            time_period=60, sensitivity="low", direction="up", expected_seasonality="auto"
        )

        self.mp_config = MPConfig(ignore_trivial=False, normalize_mp=False)

    def test_stream_smooth_no_anomalies(self):
        flags = ["none"] * 5
        result = self.smoother.smooth(flags, self.ad_config, vote_threshold=0.5)
        assert result == []

    def test_stream_smooth_single_anomaly(self):
        flags = [
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]
        result = self.smoother.smooth(flags, self.ad_config, vote_threshold=0.5)
        assert result == ["anomaly_higher_confidence"]

    def test_stream_smooth_with_custom_vote_threshold(self):
        # Test case where threshold is met
        flags_success = [
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "anomaly_higher_confidence",
            "none",
        ]
        result = self.smoother.smooth(flags_success, self.ad_config, vote_threshold=0.75)
        assert result == ["anomaly_higher_confidence"]

        # Test case where threshold is not met
        flags_failure = ["anomaly_higher_confidence", "none", "none", "none"]
        result = self.smoother.smooth(flags_failure, self.ad_config, vote_threshold=0.75)
        assert result == []

    def test_stream_smooth_with_current_flag(self):
        flags = ["anomaly_higher_confidence", "none", "none"]
        cur_flag = ["anomaly_higher_confidence"]

        # Test where threshold is not met but current flag is anomalous
        result = self.smoother.smooth(flags, self.ad_config, vote_threshold=0.5, cur_flag=cur_flag)
        assert result == ["anomaly_higher_confidence"]

        # Test where threshold is not met and current flag is none
        result = self.smoother.smooth(flags, self.ad_config, vote_threshold=0.5, cur_flag=["none"])
        assert result == ["none"]

    def test_stream_smooth_with_different_time_periods(self):
        # Test with time_period=5
        ad_config_5 = AnomalyDetectionConfig(
            time_period=5, sensitivity="low", direction="up", expected_seasonality="auto"
        )
        flags = ["anomaly_higher_confidence"] * 10 + ["none"]
        result = self.smoother.smooth(flags, ad_config_5, vote_threshold=0.5, cur_flag=["none"])
        assert result == ["anomaly_higher_confidence"]

        # Test with time_period=15
        ad_config_15 = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="up", expected_seasonality="auto"
        )
        flags = ["anomaly_higher_confidence"] * 6 + ["none"] * 2
        result = self.smoother.smooth(flags, ad_config_15, vote_threshold=0.5, cur_flag=["none"])
        assert result == ["anomaly_higher_confidence"]

        # Test with time_period=30
        ad_config_30 = AnomalyDetectionConfig(
            time_period=30, sensitivity="low", direction="up", expected_seasonality="auto"
        )
        flags = ["anomaly_higher_confidence"] * 4 + ["none"] * 1
        result = self.smoother.smooth(flags, ad_config_30, vote_threshold=0.5, cur_flag=["none"])
        assert result == ["anomaly_higher_confidence"]

        # Test with time_period=60
        ad_config_60 = AnomalyDetectionConfig(
            time_period=60, sensitivity="low", direction="up", expected_seasonality="auto"
        )
        flags = ["anomaly_higher_confidence"] * 3 + ["none"] * 1
        result = self.smoother.smooth(flags, ad_config_60, vote_threshold=0.5, cur_flag=["none"])
        assert result == ["anomaly_higher_confidence"]
