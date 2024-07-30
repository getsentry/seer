from seer.anomaly_detection.accessors import AlertDataAccessor, DbAlertDataAccessor
from seer.anomaly_detection.detectors import (
    MinMaxNormalizer,
    MPConfig,
    MPIRQScorer,
    MPScorer,
    MPUtils,
    Normalizer,
    SuSSWindowSizeSelector,
    WindowSizeSelector,
)
from seer.dependency_injection import Module

anomaly_detection_module = Module()
test_anomaly_detection_module = Module()


@anomaly_detection_module.provider
def alert_data_accessor_provider() -> AlertDataAccessor:
    return DbAlertDataAccessor()


@anomaly_detection_module.provider
def mp_scorer_provider() -> MPScorer:
    return MPIRQScorer()


@anomaly_detection_module.provider
def mpconfig_provider() -> MPConfig:
    return MPConfig(
        ignore_trivial=True, normalize_mp=False
    )  # Avoiding complexities around normalizing matrix profile across stream computation for now.


@anomaly_detection_module.provider
def ws_selector_provider() -> WindowSizeSelector:
    return SuSSWindowSizeSelector()


@anomaly_detection_module.provider
def normalizer_provider() -> Normalizer:
    return MinMaxNormalizer()


@anomaly_detection_module.provider
def mp_utils_provider() -> MPUtils:
    return MPUtils()
