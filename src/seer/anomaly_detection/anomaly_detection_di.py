from seer.anomaly_detection.accessors import AlertDataAccessor, DbAlertDataAccessor
from seer.anomaly_detection.detectors import (
    CombinedAnomalyScorer,
    MinMaxNormalizer,
    MPBoxCoxScorer,
    MPCascadingScorer,
    MPLowVarianceScorer,
    MPUtils,
    Normalizer,
    ProphetScaledSmoothedScorer,
    SuSSWindowSizeSelector,
    WindowSizeSelector,
)
from seer.anomaly_detection.detectors.anomaly_scorer import AnomalyScorer
from seer.anomaly_detection.models import AlgoConfig
from seer.dependency_injection import Module

anomaly_detection_module = Module()
test_anomaly_detection_module = Module()


@anomaly_detection_module.provider
def alert_data_accessor_provider() -> AlertDataAccessor:
    return DbAlertDataAccessor()


@anomaly_detection_module.provider
def anomaly_scorer_provider() -> AnomalyScorer:
    return CombinedAnomalyScorer(
        mp_scorer=MPCascadingScorer(scorers=[MPLowVarianceScorer(), MPBoxCoxScorer()]),
        prophet_scorer=ProphetScaledSmoothedScorer(),
    )


@anomaly_detection_module.provider
def algoconfig_provider() -> AlgoConfig:
    return AlgoConfig(
        mp_ignore_trivial=True,
        mp_normalize=False,
        mp_use_approx=True,
        prophet_uncertainty_samples=5,
        prophet_mcmc_samples=0,
        mp_fixed_window_size=10,
        return_thresholds=True,
        return_predicted_range=True,
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
