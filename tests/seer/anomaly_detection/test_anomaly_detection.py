import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from seer.anomaly_detection.anomaly_detection import AnomalyDetection
from seer.anomaly_detection.models import (
    AlertAlgorithmType,
    ConfidenceLevel,
    DynamicAlert,
    MPTimeSeries,
    MPTimeSeriesAnomaliesSingleWindow,
)
from seer.anomaly_detection.models.cleanup_predict import CleanupPredictConfig
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
from seer.anomaly_detection.models.timeseries import ProphetPrediction
from seer.anomaly_detection.models.timeseries_anomalies import MPTimeSeriesAnomalies
from seer.db import TaskStatus
from seer.exceptions import ClientError, ServerError
from tests.seer.anomaly_detection.test_utils import convert_synthetic_ts


class TestAnomalyDetection(unittest.TestCase):

    def setUp(self):
        def create_dynamic_alert(config, data_length):
            return DynamicAlert(
                organization_id=0,
                project_id=0,
                external_alert_id=0,
                config=config,
                timeseries=MPTimeSeries(
                    timestamps=self.ts_timestamps[:data_length],
                    values=self.ts_values[:data_length],
                ),
                anomalies=MPTimeSeriesAnomalies(
                    flags=np.array(["anomaly_higher_confidence"] * data_length),
                    scores=np.array([0.4] * data_length),
                    matrix_profile_suss=self.dummy_mp_suss,
                    matrix_profile_fixed=self.dummy_mp_fixed,
                    window_size=self.ts_window_size,
                    thresholds=[],
                    original_flags=np.array(["none"] * data_length),
                    use_suss=np.array([True] * data_length),
                    confidence_levels=[
                        ConfidenceLevel.MEDIUM,
                    ]
                    * data_length,
                    algorithm_types=[AlertAlgorithmType.NONE] * data_length,
                ),
                prophet_predictions=ProphetPrediction(
                    timestamps=np.array([]),
                    y=np.array([]),
                    yhat=np.array([]),
                    yhat_lower=np.array([]),
                    yhat_upper=np.array([]),
                ),
                cleanup_predict_config=self.cleanup_predict_config,
                only_suss=False,
                data_purge_flag=TaskStatus.NOT_QUEUED,
                last_queued_at=None,
            )

        self.config_15 = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="up", expected_seasonality="auto"
        )

        self.config_60 = AnomalyDetectionConfig(
            time_period=60, sensitivity="low", direction="up", expected_seasonality="auto"
        )

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )
        self.ts_timepoints = loaded_synthetic_data.timeseries[0]

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=False
        )
        self.ts_values = loaded_synthetic_data.timeseries[0]
        self.ts_timestamps = np.arange(1, len(self.ts_values) + 1) + datetime.now().timestamp()
        self.ts_window_size = loaded_synthetic_data.window_sizes[0]
        self.dummy_mp_suss = np.ones((len(self.ts_values) - self.ts_window_size + 1, 4))
        self.fixed_window_size = 10
        self.dummy_mp_fixed = np.ones((len(self.ts_values) - self.fixed_window_size + 1, 4))
        self.cleanup_predict_config = CleanupPredictConfig(
            num_old_points=0,
            timestamp_threshold=0,
            num_acceptable_points=0,
            num_predictions_remaining=0,
            num_acceptable_predictions=0,
        )
        self.default_dynamic_alert_15min = create_dynamic_alert(
            self.config_15, len(self.ts_timestamps)
        )

        self.default_dynamic_alert_60min_100_points = create_dynamic_alert(self.config_60, 100)
        self.default_dynamic_alert_60min_167_points = create_dynamic_alert(self.config_60, 167)

    def test_store_data(self):
        mock_alert_data_accessor = MagicMock()
        alert = AlertInSeer(id=0)
        request = StoreDataRequest(
            organization_id=0,
            project_id=0,
            alert=alert,
            config=self.config_15,
            timeseries=self.ts_timepoints,
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

        alert = AlertInSeer(id=0)
        request = StoreDataRequest(
            organization_id=0,
            project_id=0,
            alert=alert,
            config=config,
            timeseries=self.ts_timepoints,
        )

        def slow_function(
            timeseries, config, window_size=None, algo_config=None, time_budget_ms=None
        ):
            time.sleep(0.2)  # Simulate a 200ms delay
            return (
                timeseries,
                MPTimeSeriesAnomalies(
                    flags=[],
                    scores=[],
                    matrix_profile_suss=np.array([]),
                    matrix_profile_fixed=np.array([]),
                    window_size=0,
                    thresholds=[],
                    original_flags=[],
                    use_suss=[],
                    confidence_levels=[],
                    algorithm_types=[],
                ),
                pd.DataFrame(
                    {
                        "ds": pd.Series([], dtype=np.float64),
                        "y": pd.Series([], dtype=np.float64),
                        "actual": pd.Series([], dtype=np.float64),
                        "yhat": pd.Series([], dtype=np.float64),
                        "yhat_lower": pd.Series([], dtype=np.float64),
                        "yhat_upper": pd.Series([], dtype=np.float64),
                    },
                ),
            )

        mock_batch_detect.side_effect = slow_function

        # Lower timeout setting should raise an error
        with self.assertRaises(ServerError) as e:
            AnomalyDetection().store_data(
                request=request,
                alert_data_accessor=mock_alert_data_accessor,
                time_budget_ms=100,
            )
        assert "Batch detection took too long" in str(e.exception)
        mock_alert_data_accessor.save_alert.assert_not_called()

        response = AnomalyDetection().store_data(
            request=request,
            alert_data_accessor=mock_alert_data_accessor,
            time_budget_ms=300,
        )
        assert isinstance(response, StoreDataResponse)
        assert response.success is True
        mock_alert_data_accessor.save_alert.assert_called_once()

    def test_detect_anomalies_batch(self):
        anomaly_request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=self.config_15, context=self.ts_timepoints
        )
        response = AnomalyDetection().detect_anomalies(request=anomaly_request)
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == len(self.ts_timepoints)
        assert isinstance(response.timeseries[0], TimeSeriesPoint)

    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.query")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.save_timepoint")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.reset_cleanup_predict_task")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.can_queue_cleanup_predict_task")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.queue_data_purge_flag")
    def test_detect_anomalies_online(
        self,
        mock_queue_data_purge_flag,
        mock_can_queue_cleanup_predict_task,
        mock_reset_cleanup_predict,
        mock_save_timepoint,
        mock_query,
    ):
        mock_query.return_value = self.default_dynamic_alert_15min

        # Dummy return so we don't hit db
        mock_save_timepoint.return_value = ""
        mock_reset_cleanup_predict.return_value = ""
        mock_can_queue_cleanup_predict_task.return_value = False

        new_timestamp = len(self.ts_values) + datetime.now().timestamp() + 1
        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=new_timestamp, value=0.5))

        request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=self.config_15, context=context
        )
        response = AnomalyDetection().detect_anomalies(request=request)

        mock_query.assert_called_once()
        mock_save_timepoint.assert_called_once()
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == 1  # Checking just 1 streamed value
        assert isinstance(response.timeseries[0], TimeSeriesPoint)
        assert response.timeseries[0].timestamp == new_timestamp
        assert mock_queue_data_purge_flag.call_count == 0

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
        # Reduce anomalies length

        mock_query.return_value = self.default_dynamic_alert_15min.model_copy(
            update={
                "anomalies": self.default_dynamic_alert_15min.anomalies.model_copy(
                    update={
                        "original_flags": ["none"]
                        * (len(self.ts_timestamps) - 1),  # One less than timestamps,
                    }
                )
            }
        )

        with self.assertRaises(ServerError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "Invalid state" in str(e.exception)

    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.query")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.save_timepoint")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.reset_cleanup_predict_task")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.can_queue_cleanup_predict_task")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.queue_data_purge_flag")
    def test_detect_anomalies_online_less_than_167_points(
        self,
        mock_queue_data_purge_flag,
        mock_can_queue_cleanup_predict_task,
        mock_reset_cleanup_predict,
        mock_save_timepoint,
        mock_query,
    ):
        # Alert has much less data than min requirement of 168 and data pruning should not be triggered
        mock_can_queue_cleanup_predict_task.return_value = True

        mock_query.return_value = self.default_dynamic_alert_60min_100_points
        # Dummy return so we don't hit db
        mock_save_timepoint.return_value = ""
        mock_reset_cleanup_predict.return_value = ""
        mock_can_queue_cleanup_predict_task.return_value = False

        new_timestamp = len(self.ts_values) + datetime.now().timestamp() + 1
        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=new_timestamp, value=0.5))
        request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=self.config_15, context=context
        )

        with self.assertRaises(ClientError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "Not enough timeseries data" in str(e.exception)
        assert mock_save_timepoint.call_count == 1
        assert (
            mock_queue_data_purge_flag.call_count == 0
        )  # data purge should not be fired if alert has less than 167 points
        assert mock_can_queue_cleanup_predict_task.call_count == 0

    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.query")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.save_timepoint")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.reset_cleanup_predict_task")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.can_queue_cleanup_predict_task")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.queue_data_purge_flag")
    def test_detect_anomalies_online_with_167_points(
        self,
        mock_queue_data_purge_flag,
        mock_can_queue_cleanup_predict_task,
        mock_reset_cleanup_predict,
        mock_save_timepoint,
        mock_query,
    ):
        mock_query.return_value = self.default_dynamic_alert_60min_167_points

        mock_save_timepoint.return_value = ""
        mock_reset_cleanup_predict.return_value = ""
        mock_can_queue_cleanup_predict_task.return_value = True
        mock_queue_data_purge_flag.return_value = ""

        new_timestamp = len(self.ts_values) + datetime.now().timestamp() + 1
        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=new_timestamp, value=0.5))
        request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=self.config_60, context=context
        )

        with self.assertRaises(ClientError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "Not enough timeseries data" in str(e.exception)
        assert mock_save_timepoint.call_count == 1
        assert mock_can_queue_cleanup_predict_task.call_count == 1
        assert mock_queue_data_purge_flag.call_count == 1

    @patch("seer.anomaly_detection.detectors.MPStreamAnomalyDetector.detect")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.query")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.save_timepoint")
    @patch("seer.anomaly_detection.accessors.DbAlertDataAccessor.reset_cleanup_predict_task")
    def test_detect_anomalies_online_switch_window(
        self, mock_reset_cleanup_predict, mock_save_timepoint, mock_query, mock_stream_detector
    ):
        mock_query.return_value = self.default_dynamic_alert_15min

        # Dummy return so we don't hit db
        mock_save_timepoint.return_value = ""
        mock_reset_cleanup_predict.return_value = ""

        # Can return back to the SuSS window after using the fixed window
        mock_stream_detector.return_value = MPTimeSeriesAnomaliesSingleWindow(
            flags=["none"] * len(self.ts_timestamps),
            scores=[0.0] * len(self.ts_timestamps),
            matrix_profile=self.dummy_mp_suss,
            window_size=self.ts_window_size,
            thresholds=[],
            original_flags=["none"] * len(self.ts_timestamps),
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
            ]
            * len(self.ts_timestamps),
            algorithm_types=[AlertAlgorithmType.NONE] * len(self.ts_timestamps),
        )

        new_timestamp = len(self.ts_values) + datetime.now().timestamp() + 1
        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=new_timestamp, value=0.5))

        request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=self.config_15, context=context
        )
        response = AnomalyDetection().detect_anomalies(request=request)

        assert (
            mock_stream_detector.call_count == 1
        )  # Only called once because the window should not switch
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == 1  # Checking just 1 streamed value
        assert isinstance(response.timeseries[0], TimeSeriesPoint)
        assert response.timeseries[0].timestamp == new_timestamp

        mock_query.return_value = self.default_dynamic_alert_15min

        response = AnomalyDetection().detect_anomalies(request=request)

        assert mock_stream_detector.call_count == 2
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == 1  # Checking just 1 streamed value
        assert isinstance(response.timeseries[0], TimeSeriesPoint)
        assert response.timeseries[0].timestamp == new_timestamp

    def test_detect_anomalies_online_none_value(self):

        new_timestamp = len(self.ts_values) + datetime.now().timestamp() + 1
        context = AlertInSeer(id=0, cur_window=TimeSeriesPoint(timestamp=new_timestamp, value=None))

        request = DetectAnomaliesRequest(
            organization_id=0, project_id=0, config=self.config_15, context=context
        )
        with self.assertRaises(ClientError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "Time series point has None value" in str(e.exception)

    def test_detect_anomalies_combo(self):

        loaded_synthetic_data = convert_synthetic_ts(
            "tests/seer/anomaly_detection/test_data/synthetic_series", as_ts_datatype=True
        )

        n = 5
        for i, ts_history in enumerate(loaded_synthetic_data.timeseries):

            # Generate new observation window of n points which are the same as the last point
            ts_current = []
            last_history_timestamp = ts_history[-1].timestamp
            last_history_value = ts_history[-1].value
            for j in range(1, n + 1):
                ts_current.append(
                    TimeSeriesPoint(
                        timestamp=last_history_timestamp
                        + 15 * 60 * j,  # timestamps with 15 min intervals (in seconds)
                        value=last_history_value,
                    )
                )

            context = TimeSeriesWithHistory(history=ts_history, current=ts_current)

            request = DetectAnomaliesRequest(
                organization_id=i, project_id=i, config=self.config_15, context=context
            )

            response = AnomalyDetection().detect_anomalies(request=request)

            assert isinstance(response, DetectAnomaliesResponse)
            assert isinstance(response.timeseries, list)
            assert len(response.timeseries) == n
            assert isinstance(response.timeseries[0], TimeSeriesPoint)

    def test_detect_anomalies_combo_large_current(self):
        ts_history = self.ts_timepoints
        ts_history = self.ts_timepoints
        last_history_timestamp = ts_history[-1].timestamp
        last_history_value = ts_history[-1].value
        n = 700  # should be greater than 7 days * 24 hours * 60 minutes * 15 minutes = 672

        # Generate new observation window of n points which are the same as the last point
        ts_current = []
        for j in range(1, n + 1):
            ts_current.append(
                TimeSeriesPoint(
                    timestamp=last_history_timestamp + self.config_15.time_period * 60 * j,
                    value=last_history_value,
                )
            )

        context = TimeSeriesWithHistory(history=ts_history, current=ts_current)

        request = DetectAnomaliesRequest(
            organization_id=1, project_id=1, config=self.config_15, context=context
        )

        response = AnomalyDetection().detect_anomalies(request=request, time_budget_ms=20000)

        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.timeseries, list)
        assert len(response.timeseries) == n
        assert isinstance(response.timeseries[0], TimeSeriesPoint)

    @patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection._batch_detect_internal")
    def test_detect_anomalies_combo_batch_timeout(self, mock_batch_detect):
        def slow_function(
            ts_internal,
            config,
            window_size,
            algo_config,
            time_budget_ms=None,
            prophet_forecast_len_days=None,
        ):  # -> Tuple[MPTimeSeriesAnomaliesSingleWindow, pd.DataFrame]:
            time.sleep(0.15)  # Simulate a 150ms delay
            return (
                None,
                pd.DataFrame(
                    {
                        "ds": pd.Series([], dtype=np.float64),
                        "y": pd.Series([], dtype=np.float64),
                        "actual": pd.Series([], dtype=np.float64),
                        "yhat": pd.Series([], dtype=np.float64),
                        "yhat_lower": pd.Series([], dtype=np.float64),
                        "yhat_upper": pd.Series([], dtype=np.float64),
                    },
                ),
            )

        mock_batch_detect.side_effect = slow_function
        ts_history = self.ts_timepoints[:180]
        n = 400
        last_history_timestamp = ts_history[-1].timestamp
        last_history_value = ts_history[-1].value
        # Generate new observation window of n points which are the same as the last point
        ts_current = []
        for j in range(1, n + 1):
            ts_current.append(
                TimeSeriesPoint(
                    timestamp=last_history_timestamp + self.config_60.time_period * 60 * j,
                    value=last_history_value,
                )
            )

        context = TimeSeriesWithHistory(history=ts_history, current=ts_current)
        request = DetectAnomaliesRequest(
            organization_id=1, project_id=1, config=self.config_60, context=context
        )

        # Test that detection with small time budget raises timeout error
        with self.assertRaises(ServerError) as e:
            AnomalyDetection().detect_anomalies(request=request, time_budget_ms=200)
        mock_batch_detect.assert_called_once()
        assert "Batch detection took too long" in str(e.exception)

    @patch("seer.anomaly_detection.detectors.MPStreamAnomalyDetector.detect")
    def test_detect_anomalies_combo_current_timeout(self, mock_stream_detect):
        def slow_function(
            timeseries,
            ad_config,
            algo_config=None,
            prophet_df=None,
            time_budget_ms=None,
            scorer=None,
            mp_utils=None,
        ):
            raise ServerError("Stream detection took too long")

        mock_stream_detect.side_effect = slow_function
        ts_history = self.ts_timepoints[:180]
        n = 400
        last_history_timestamp = ts_history[-1].timestamp
        last_history_value = ts_history[-1].value
        # Generate new observation window of n points which are the same as the last point
        ts_current = []
        for j in range(1, n + 1):
            ts_current.append(
                TimeSeriesPoint(
                    timestamp=last_history_timestamp + self.config_60.time_period * 60 * j,
                    value=last_history_value,
                )
            )

        context = TimeSeriesWithHistory(history=ts_history, current=ts_current)
        request = DetectAnomaliesRequest(
            organization_id=1, project_id=1, config=self.config_60, context=context
        )

        # Test that detection with small time budget raises timeout error
        with self.assertRaises(ServerError) as e:
            AnomalyDetection().detect_anomalies(request=request, time_budget_ms=1000)
        mock_stream_detect.assert_called_once()
        assert "Stream detection took too long" in str(e.exception)

    def test_detect_anomalies_combo_insufficient_history(self):
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )
        n = 10
        ts_history = self.ts_timepoints[:n]

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

        with self.assertRaises(ClientError) as e:
            AnomalyDetection().detect_anomalies(request=request)
        assert "Insufficient history data" in str(e.exception)

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
