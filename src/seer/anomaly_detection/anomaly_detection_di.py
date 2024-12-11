from seer.anomaly_detection.accessors import AlertDataAccessor, DbAlertDataAccessor
from seer.anomaly_detection.detectors import (
    MinMaxNormalizer,
    MPCascadingScorer,
    MPScorer,
    MPUtils,
    Normalizer,
    SuSSWindowSizeSelector,
    WindowSizeSelector,
)
from seer.anomaly_detection.detectors.location_detectors import (
    LocationDetector,
    ProphetLocationDetector,
)
from seer.anomaly_detection.models import AlgoConfig
from seer.dependency_injection import Module

anomaly_detection_module = Module()
test_anomaly_detection_module = Module()


@anomaly_detection_module.provider
def alert_data_accessor_provider() -> AlertDataAccessor:
    return DbAlertDataAccessor()


@anomaly_detection_module.provider
def mp_scorer_provider() -> MPScorer:
    return MPCascadingScorer()


@anomaly_detection_module.provider
def algoconfig_provider() -> AlgoConfig:
    return AlgoConfig(
        mp_ignore_trivial=True,
        mp_normalize=False,
        prophet_uncertainty_samples=1,
        mp_fixed_window_size=10,
        return_thresholds=False,
        return_predicted_range=False,
    )


@anomaly_detection_module.provider
def ws_selector_provider() -> WindowSizeSelector:
    return SuSSWindowSizeSelector()


@anomaly_detection_module.provider
def normalizer_provider() -> Normalizer:
    return MinMaxNormalizer()


@anomaly_detection_module.provider
def mp_utils_provider() -> MPUtils:
    return MPUtils()


@anomaly_detection_module.provider
def location_detector_provider() -> LocationDetector:
    return ProphetLocationDetector()
