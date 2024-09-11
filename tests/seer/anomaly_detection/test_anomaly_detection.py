import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from seer.anomaly_detection.anomaly_detection import AnomalyDetection
from seer.anomaly_detection.models import DynamicAlert, MPTimeSeries
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
from seer.anomaly_detection.models.timeseries_anomalies import MPTimeSeriesAnomalies
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestAnomalyDetection(unittest.TestCase):

    def test_store_data(self):

        mock_alert_data_accessor = MagicMock()
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        timeseries, _, _ = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )
        ts = timeseries[0]

        alert = AlertInSeer(id=0)
        request = StoreDataRequest(
            organization_id=0,
            project_id=0,
            alert=alert,
            config=config,
            timeseries=ts,
        )

        response = AnomalyDetection().store_data(
            request=request, alert_data_accessor=mock_alert_data_accessor
        )

        mock_alert_data_accessor.save_alert.assert_called_once()
        assert "Store Data Response should be successful", response == StoreDataResponse(
            success=True
        )

    def test_detect_anomalies_batch(self):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        timeseries, _, _ = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )
        ts = timeseries[0]

        anomaly_request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=config, context=ts
        )

        response = AnomalyDetection().detect_anomalies(request=anomaly_request)

        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == len(ts)
        assert isinstance(response.timeseries[0], TimeSeriesPoint)

    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.query")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.save_timepoint")
    def test_detect_anomalies_online(self, mock_save_timepoint, mock_query):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        timeseries, _, window_sizes = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )
        ts = timeseries[0]
        window_size = window_sizes[0]

        dummy_mp = np.ones((len(ts) - window_size + 1, 4))

        mock_query.return_value = DynamicAlert(
            organization_id=0,
            project_id=0,
            external_alert_id=0,
            config=config,
            timeseries=MPTimeSeries(
                timestamps=np.linspace(1, len(ts), 1),
                values=ts,
            ),
            anomalies=MPTimeSeriesAnomalies(
                flags=np.array(["anomaly_high_confidence"]),
                scores=np.array([0.4]),
                matrix_profile=dummy_mp,
                window_size=window_size,
            ),
        )

        # Dummy return so we don't hit db
        mock_save_timepoint.return_value = ""

        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=len(ts) + 1, value=0.5))

        request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=config, context=context
        )
        response = AnomalyDetection().detect_anomalies(request=request)

        mock_query.assert_called_once()
        mock_save_timepoint.assert_called_once()
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == 1  # Checking just 1 streamed value
        assert isinstance(response.timeseries[0], TimeSeriesPoint)
        assert response.timeseries[0].timestamp == len(ts) + 1

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
