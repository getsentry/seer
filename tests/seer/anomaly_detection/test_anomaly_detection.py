import unittest

from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.anomaly_detection import AnomalyDetection
from seer.anomaly_detection.models.external import (
    AlertInSeer,
    AnomalyDetectionConfig,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
    TimeSeriesPoint,
    TimeSeriesWithHistory,
)
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestAnomalyDetection(unittest.TestCase):

    def test_store_data(self):

        alert_data_accessor = DbAlertDataAccessor()
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        timeseries, _, _ = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )

        for i, ts in enumerate(timeseries):

            alert = AlertInSeer(id=i)

            request = StoreDataRequest(
                organization_id=i,
                project_id=i,
                alert=alert,
                config=config,
                timeseries=ts,
            )

            response = AnomalyDetection().store_data(
                request=request, alert_data_accessor=alert_data_accessor
            )

            assert "Store Data Response should be successful", response == StoreDataResponse(
                success=True
            )

            # TODO: Clean up DB(?)

    def test_detect_anomalies_batch(self):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        timeseries, _, _ = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )

        for i, ts in enumerate(timeseries):

            anomaly_request = DetectAnomaliesRequest(
                organization_id=i, project_id=i, config=config, context=ts
            )

            response = AnomalyDetection().detect_anomalies(request=anomaly_request)

            assert isinstance(response, DetectAnomaliesResponse)
            assert isinstance(response.timeseries, list)
            assert len(response.timeseries) == len(ts)
            assert isinstance(response.timeseries[0], TimeSeriesPoint)

    def test_detect_anomalies_online(self):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        timeseries, _, _ = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )

        for i, ts in enumerate(timeseries):

            context = AlertInSeer(
                id=i, cur_window=TimeSeriesPoint(timestamp=len(ts) + 1, value=0.5)
            )

            request = DetectAnomaliesRequest(
                organization_id=i, project_id=i, config=config, context=context
            )

            response = AnomalyDetection().detect_anomalies(request=request)

            assert isinstance(response, DetectAnomaliesResponse)
            assert isinstance(response.timeseries, list)
            assert (
                len(response.timeseries) == len(ts) + 1
            )  # Adding one more observation to timeseries
            assert isinstance(response.timeseries[0], TimeSeriesPoint)

    def test_detect_anomalies_combo(self):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        timeseries, _, _ = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )

        n = 5
        for i, ts_history in enumerate(timeseries):

            # Generate new observation window of n points which are the same as the last point
            ts_current = []
            for j in range(1, n + 1):
                ts_current.append(
                    TimeSeriesPoint(timestamp=len(ts_history) + j, value=ts_history[-1].value)
                )

            context = TimeSeriesWithHistory(history=ts_history, current=ts_current)

            request = DetectAnomaliesRequest(
                organization_id=i, project_id=i, config=config, context=context
            )

            response = AnomalyDetection().detect_anomalies(request=request)

            assert isinstance(response, DetectAnomaliesResponse)
            assert isinstance(response.timeseries, list)
            assert len(response.timeseries) == n
            assert isinstance(response.timeseries[0], TimeSeriesPoint)
