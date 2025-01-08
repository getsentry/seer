import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from seer.anomaly_detection.anomaly_detection import AnomalyDetection
from seer.anomaly_detection.models import (
    DynamicAlert,
    MPTimeSeries,
    MPTimeSeriesAnomaliesSingleWindow,
)
from seer.anomaly_detection.models.cleanup import CleanupConfig
from seer.anomaly_detection.models.external import (
    AlertInSeer,
    AnomalyDetectionConfig,
    DeleteAlertDataRequest,
    DeleteAlertDataResponse,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
    TimeSeriesPoint,
    TimeSeriesWithHistory,
)
from seer.anomaly_detection.models.timeseries_anomalies import MPTimeSeriesAnomalies
from seer.exceptions import ClientError, ServerError
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestAnomalyDetection(unittest.TestCase):

    def test_store_data(self):

        mock_alert_data_accessor = MagicMock()
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )
        ts = loaded_synthetic_data.timeseries[0]

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

    @patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection._batch_detect")
    def test_store_data_batch_detection_timeout(self, mock_batch_detect):

        mock_alert_data_accessor = MagicMock()
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )
        ts = loaded_synthetic_data.timeseries[0]

        alert = AlertInSeer(id=0)
        request = StoreDataRequest(
            organization_id=0,
            project_id=0,
            alert=alert,
            config=config,
            timeseries=ts,
        )

        # timeseries: List[TimeSeriesPoint],
        # config: AnomalyDetectionConfig,
        # window_size: int | None = None,
        # algo_config: AlgoConfig = injected,

        def slow_function(timeseries, config, window_size=None, algo_config=None):
            time.sleep(0.2)  # Simulate a 200ms delay
            return timeseries, MPTimeSeriesAnomalies(
                flags=[],
                scores=[],
                matrix_profile_suss=np.array([]),
                matrix_profile_fixed=np.array([]),
                window_size=0,
                thresholds=[],
                original_flags=[],
                use_suss=[],
            )

        mock_batch_detect.side_effect = slow_function

        # Lower timeout setting should raise an error
        with self.assertRaises(ServerError) as e:
            AnomalyDetection().store_data(
                request=request,
                alert_data_accessor=mock_alert_data_accessor,
                timeout_ms=100,
            )
        assert "Batch detection took too long" in str(e.exception)
        mock_alert_data_accessor.save_alert.assert_not_called()

        response = AnomalyDetection().store_data(
            request=request,
            alert_data_accessor=mock_alert_data_accessor,
            timeout_ms=300,
        )
        assert isinstance(response, StoreDataResponse)
        assert response.success is True
        mock_alert_data_accessor.save_alert.assert_called_once()

    def test_detect_anomalies_batch(self):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )
        ts = loaded_synthetic_data.timeseries[0]

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
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.reset_cleanup_task")
    def test_detect_anomalies_online(
        self,
        mock_reset_cleanup,
        mock_save_timepoint,
        mock_query,
    ):

        fixed_window_size = 10

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        cleanup_config = CleanupConfig(
            num_old_points=0, timestamp_threshold=0, num_acceptable_points=0
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )
        ts_values = loaded_synthetic_data.timeseries[0]
        ts_timestamps = np.arange(1, len(ts_values) + 1) + datetime.now().timestamp()
        window_size = loaded_synthetic_data.window_sizes[0]

        dummy_mp_suss = np.ones((len(ts_values) - window_size + 1, 4))
        dummy_mp_fixed = np.ones((len(ts_values) - fixed_window_size + 1, 4))

        mock_query.return_value = DynamicAlert(
            organization_id=0,
            project_id=0,
            external_alert_id=0,
            config=config,
            timeseries=MPTimeSeries(
                timestamps=ts_timestamps,
                values=ts_values,
            ),
            anomalies=MPTimeSeriesAnomalies(
                flags=np.array(["anomaly_higher_confidence"] * len(ts_timestamps)),
                scores=np.array([0.4] * len(ts_timestamps)),
                matrix_profile_suss=dummy_mp_suss,
                matrix_profile_fixed=dummy_mp_fixed,
                window_size=window_size,
                thresholds=[],
                original_flags=np.array(["none"] * len(ts_timestamps)),
                use_suss=np.array([True] * len(ts_timestamps)),
            ),
            cleanup_config=cleanup_config,
            only_suss=False,
        )

        # Dummy return so we don't hit db
        mock_save_timepoint.return_value = ""
        mock_reset_cleanup.return_value = ""

        new_timestamp = len(ts_values) + datetime.now().timestamp() + 1
        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=new_timestamp, value=0.5))

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
        assert response.timeseries[0].timestamp == new_timestamp

        # Alert has less data than min requirement
        mock_query.return_value = DynamicAlert(
            organization_id=0,
            project_id=0,
            external_alert_id=0,
            config=config,
            timeseries=MPTimeSeries(
                timestamps=ts_timestamps[:100],
                values=ts_values,
            ),
            anomalies=MPTimeSeriesAnomalies(
                flags=np.array(["anomaly_higher_confidence"] * len(ts_timestamps[:100])),
                scores=np.array([0.4] * len(ts_timestamps[:100])),
                matrix_profile_suss=dummy_mp_suss,
                matrix_profile_fixed=dummy_mp_fixed,
                window_size=window_size,
                thresholds=[],
                original_flags=np.array(["none"] * len(ts_timestamps[:100])),
                use_suss=np.array([True] * len(ts_timestamps[:100])),
            ),
            cleanup_config=cleanup_config,
            only_suss=False,
        )

        with self.assertRaises(Exception):
            AnomalyDetection().detect_anomalies(request=request)

        # Alert is None
        mock_query.return_value = None

        with self.assertRaises(ClientError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "No timeseries data found for alert" in str(e.exception)

        # Alert is not of type MPTimeSeriesAnomalies
        mock_query.return_value = MPTimeSeries(
            timestamps=np.array([]),
            values=np.array([]),
            anomalies=None,
        )

        with self.assertRaises(ServerError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "Invalid state" in str(e.exception)

        # Test validation of original flags length matching timeseries length
        mock_query.return_value = DynamicAlert(
            organization_id=0,
            project_id=0,
            external_alert_id=0,
            config=config,
            timeseries=MPTimeSeries(
                timestamps=ts_timestamps,
                values=ts_values,
            ),
            anomalies=MPTimeSeriesAnomalies(
                flags=["none"] * len(ts_timestamps),
                scores=[0.0] * len(ts_timestamps),
                matrix_profile_suss=dummy_mp_suss,
                matrix_profile_fixed=dummy_mp_fixed,
                window_size=window_size,
                thresholds=[],
                original_flags=["none"] * (len(ts_timestamps) - 1),  # One less than timestamps
                use_suss=[True] * len(ts_timestamps),
            ),
            cleanup_config=cleanup_config,
            only_suss=False,
        )

        with self.assertRaises(ServerError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "Invalid state" in str(e.exception)

    @patch("seer.anomaly_detection.detectors.MPStreamAnomalyDetector.detect")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.query")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.save_timepoint")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.reset_cleanup_task")
    def test_detect_anomalies_online_switch_window(
        self, mock_reset_cleanup, mock_save_timepoint, mock_query, mock_stream_detector
    ):

        fixed_window_size = 10

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        cleanup_config = CleanupConfig(
            num_old_points=0, timestamp_threshold=0, num_acceptable_points=0
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )
        ts_values = loaded_synthetic_data.timeseries[0]
        ts_timestamps = np.arange(1, len(ts_values) + 1) + datetime.now().timestamp()
        window_size = loaded_synthetic_data.window_sizes[0]

        dummy_mp_suss = np.ones((len(ts_values) - window_size + 1, 4))
        dummy_mp_fixed = np.ones((len(ts_values) - fixed_window_size + 1, 4))

        mock_query.return_value = DynamicAlert(
            organization_id=0,
            project_id=0,
            external_alert_id=0,
            config=config,
            timeseries=MPTimeSeries(
                timestamps=ts_timestamps,
                values=ts_values,
            ),
            anomalies=MPTimeSeriesAnomalies(
                flags=np.array(["anomaly_higher_confidence"] * len(ts_timestamps)),
                scores=np.array([0.4] * len(ts_timestamps)),
                matrix_profile_suss=dummy_mp_suss,
                matrix_profile_fixed=dummy_mp_fixed,
                window_size=window_size,
                thresholds=[],
                original_flags=np.array(["none"] * len(ts_timestamps)),
                use_suss=np.array([False] * len(ts_timestamps)),
            ),
            cleanup_config=cleanup_config,
            only_suss=False,
        )

        # Dummy return so we don't hit db
        mock_save_timepoint.return_value = ""
        mock_reset_cleanup.return_value = ""

        # Can return back to the SuSS window after using the fixed window
        mock_stream_detector.return_value = MPTimeSeriesAnomaliesSingleWindow(
            flags=["none"] * len(ts_timestamps),
            scores=[0.0] * len(ts_timestamps),
            matrix_profile=dummy_mp_suss,
            window_size=window_size,
            thresholds=[],
            original_flags=["none"] * len(ts_timestamps),
        )

        new_timestamp = len(ts_values) + datetime.now().timestamp() + 1
        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=new_timestamp, value=0.5))

        request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=config, context=context
        )
        response = AnomalyDetection().detect_anomalies(request=request)

        assert mock_stream_detector.call_count == 2
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == 1  # Checking just 1 streamed value
        assert isinstance(response.timeseries[0], TimeSeriesPoint)
        assert response.timeseries[0].timestamp == new_timestamp

        mock_query.return_value = DynamicAlert(
            organization_id=0,
            project_id=0,
            external_alert_id=0,
            config=config,
            timeseries=MPTimeSeries(
                timestamps=ts_timestamps,
                values=ts_values,
            ),
            anomalies=MPTimeSeriesAnomalies(
                flags=np.array(["anomaly_higher_confidence"] * len(ts_timestamps)),
                scores=np.array([0.4] * len(ts_timestamps)),
                matrix_profile_suss=dummy_mp_suss,
                matrix_profile_fixed=dummy_mp_fixed,
                window_size=window_size,
                thresholds=[],
                original_flags=np.array(["none"] * len(ts_timestamps)),
                use_suss=np.array([True] * len(ts_timestamps)),
            ),
            cleanup_config=cleanup_config,
            only_suss=True,
        )

        response = AnomalyDetection().detect_anomalies(request=request)

        assert mock_stream_detector.call_count == 3
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == 1  # Checking just 1 streamed value
        assert isinstance(response.timeseries[0], TimeSeriesPoint)
        assert response.timeseries[0].timestamp == new_timestamp

    def test_detect_anomalies_combo(self):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )

        n = 5
        for i, ts_history in enumerate(loaded_synthetic_data.timeseries):

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

    def test_detect_anomalies_combo_large_current(self):

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )
        ts_history = loaded_synthetic_data.timeseries[0]
        n = 600

        # Generate new observation window of n points which are the same as the last point
        ts_current = []
        for j in range(1, n + 1):
            ts_current.append(
                TimeSeriesPoint(timestamp=len(ts_history) + j, value=ts_history[-1].value)
            )

        context = TimeSeriesWithHistory(history=ts_history, current=ts_current)

        request = DetectAnomaliesRequest(
            organization_id=1, project_id=1, config=config, context=context
        )

        response = AnomalyDetection().detect_anomalies(request=request)

        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == n
        assert isinstance(response.timeseries[0], TimeSeriesPoint)

    def test_delete_alert_data_success(self):
        mock_alert_data_accessor = MagicMock()

        request = DeleteAlertDataRequest(organization_id=0, project_id=0, alert=AlertInSeer(id=1))
        response = AnomalyDetection().delete_alert_data(
            request=request, alert_data_accessor=mock_alert_data_accessor
        )
        assert response == DeleteAlertDataResponse(success=True)

    def test_delete_alert_data_failure(self):
        mock_alert_data_accessor = MagicMock()
        mock_alert_data_accessor.delete_alert_data.side_effect = ClientError("Alert id 1 not found")

        request = DeleteAlertDataRequest(organization_id=0, project_id=0, alert=AlertInSeer(id=1))
        with self.assertRaises(ClientError) as e:
            AnomalyDetection().delete_alert_data(
                request=request, alert_data_accessor=mock_alert_data_accessor
            )
        assert "Alert id 1 not found" in str(e.exception)
