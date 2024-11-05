import pytest

from seer.anomaly_detection.detectors.smoothers import MajorityVoteFlagSmoother
from seer.anomaly_detection.models.external import AnomalyDetectionConfig


@pytest.fixture
def flag_smoother():
    return MajorityVoteFlagSmoother()


@pytest.fixture
def ad_config():
    return AnomalyDetectionConfig(
        time_period=60, sensitivity="medium", direction="both", expected_seasonality="auto"
    )


def test_smooth_no_anomalies(flag_smoother, ad_config):
    flags = ["none"] * 5
    result = flag_smoother.smooth(flags, ad_config)
    assert result == flags


def test_smooth_single_anomaly(flag_smoother, ad_config):
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
    result = flag_smoother.smooth(flags, ad_config, stream_smoothing=False)
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


def test_smooth_multiple_anomalies(flag_smoother, ad_config):
    flags = ["none"] * 300
    flags[100:110] = ["anomaly_higher_confidence"] * 10
    flags[111:120] = ["anomaly_higher_confidence"] * 9

    flags[200:210] = ["anomaly_higher_confidence"] * 10
    flags[211:220] = ["anomaly_higher_confidence"] * 9

    result = flag_smoother.smooth(flags, ad_config)
    expected = ["none"] * 300
    expected[100:120] = ["anomaly_higher_confidence"] * 20
    expected[200:220] = ["anomaly_higher_confidence"] * 20
    assert result == expected


def test_smooth_with_custom_smooth_size(flag_smoother, ad_config):
    flags = [
        "none",
        "anomaly_higher_confidence",
        "anomaly_higher_confidence",
        "none",
        "anomaly_higher_confidence",
        "none",
        "none",
    ]
    result = flag_smoother.smooth(flags, ad_config, smooth_size=3)
    assert result == [
        "none",
        "anomaly_higher_confidence",
        "anomaly_higher_confidence",
        "anomaly_higher_confidence",
        "anomaly_higher_confidence",
        "none",
        "none",
    ]


def test_smooth_with_custom_vote_threshold(flag_smoother, ad_config):
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
    result_success = flag_smoother.smooth(flags_success, ad_config, vote_threshold=0.75)
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
    result_failure = flag_smoother.smooth(flags_failure, ad_config, vote_threshold=0.75)
    assert result_failure == expected_failure


def test_smooth_with_different_time_periods(flag_smoother):
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
    result_5 = flag_smoother.smooth(flags, ad_config_5)
    assert result_5 == expected


def test_stream_smooth(flag_smoother):
    flags = ["none", "anomaly_higher_confidence", "anomaly_higher_confidence", "none", "none"]
    ad_config = AnomalyDetectionConfig(
        time_period=5, sensitivity="medium", direction="both", expected_seasonality="auto"
    )

    # Test with empty current flag
    result = flag_smoother.smooth(
        flags=flags, ad_config=ad_config, vote_threshold=0.4, stream_smoothing=True, cur_flag=[]
    )
    assert result == ["anomaly_higher_confidence"]

    # Test with threshold that's too high
    result = flag_smoother.smooth(
        flags=flags, ad_config=ad_config, vote_threshold=0.5, stream_smoothing=True, cur_flag=[]
    )
    assert result == []

    # Test with existing current flag
    result = flag_smoother.smooth(
        flags=flags,
        ad_config=ad_config,
        vote_threshold=0.4,
        stream_smoothing=True,
        cur_flag=["none"],
    )
    assert result == ["anomaly_higher_confidence"]
