import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np

from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.models import (
    AlertAlgorithmType,
    Anomaly,
    ConfidenceLevel,
    MPTimeSeriesAnomalies,
    MPTimeSeriesAnomaliesSingleWindow,
    Threshold,
    ThresholdType,
)
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.db import (
    DbDynamicAlert,
    DbDynamicAlertTimeSeries,
    DbDynamicAlertTimeSeriesHistory,
    DbProphetAlertTimeSeries,
    DbProphetAlertTimeSeriesHistory,
    Session,
    TaskStatus,
)


class TestDbAlertDataAccessor(unittest.TestCase):
    def test_save_alert(self):
        organization_id = 100
        project_id = 101
        external_alert_id = 10
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        point1 = TimeSeriesPoint(timestamp=500.0, value=42.42)
        point2 = TimeSeriesPoint(timestamp=1000.0, value=500.0)
        anomalies = MPTimeSeriesAnomalies(
            flags=["none", "none"],
            scores=[1.0, 0.95],
            matrix_profile_suss=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            matrix_profile_fixed=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
            thresholds=[],
            original_flags=["none", "none"],
            use_suss=[True, True],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[AlertAlgorithmType.NONE, AlertAlgorithmType.NONE],
        )
        # Verify saving
        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=[point1, point2],
            anomalies=anomalies,
            anomaly_algo_data={"window_size": 1},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        # Make sure that no prophet predictions are saved
        alert_no_prophet = alert_data_accessor.query(external_alert_id=external_alert_id)
        assert alert_no_prophet is not None
        assert alert_no_prophet.prophet_predictions is not None
        assert len(alert_no_prophet.prophet_predictions.timestamps) == 0
        assert len(alert_no_prophet.prophet_predictions.yhat) == 0
        assert len(alert_no_prophet.prophet_predictions.yhat_lower) == 0
        assert len(alert_no_prophet.prophet_predictions.yhat_upper) == 0
        assert alert_no_prophet.cleanup_predict_config.num_predictions_remaining == 0

        with Session() as session:
            self.assertEqual(
                session.query(DbDynamicAlert).count(), 1, "One and only one dynamic alert row saved"
            )
            self.assertEqual(
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=external_alert_id)
                .count(),
                1,
                f"One and only one dynamic alert with the external_alert_id {external_alert_id}",
            )

            # Include some prophet predictions
            alert = (
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=external_alert_id)
                .one_or_none()
            )

            prophet_timestamps = [
                datetime.now() + timedelta(minutes=i * config.time_period) for i in range(12)
            ]
            prophet_yhats = [42.42 + i for i in range(12)]
            prophet_yhat_lowers = [42.42 + i - 1 for i in range(12)]
            prophet_yhat_uppers = [42.42 + i + 1 for i in range(12)]

            for i, timestamp in enumerate(prophet_timestamps):
                prophet_prediction = DbProphetAlertTimeSeries(
                    dynamic_alert_id=alert.id,
                    timestamp=timestamp,
                    yhat=prophet_yhats[i],
                    yhat_lower=prophet_yhat_lowers[i],
                    yhat_upper=prophet_yhat_uppers[i],
                )
                session.add(prophet_prediction)
            session.commit()

        alert_from_db = alert_data_accessor.query(external_alert_id=external_alert_id)

        self.assertIsNotNone(alert_from_db, "Should retrieve the alert record")
        self.assertEqual(
            alert_from_db.organization_id,
            organization_id,
            "Organization id should match",
        )
        self.assertEqual(alert_from_db.project_id, project_id, "Project id should match")
        self.assertEqual(
            alert_from_db.external_alert_id,
            external_alert_id,
            "external_alert_id id should match",
        )
        self.assertEqual(
            alert_from_db.config.time_period,
            config.time_period,
            "time_period in config should match",
        )

        self.assertEqual(
            alert_from_db.config.sensitivity,
            config.sensitivity,
            "sensitivity in config should match",
        )

        self.assertEqual(
            alert_from_db.config.direction, config.direction, "direction in config should match"
        )

        self.assertEqual(
            alert_from_db.config.expected_seasonality,
            config.expected_seasonality,
            "seasonality in config should match",
        )

        self.assertEqual(
            len(alert_from_db.timeseries.timestamps), 2, "Must have two data points in timeseries"
        )
        self.assertEqual(
            len(alert_from_db.timeseries.values), 2, "Must have two data points in timeseries"
        )
        self.assertEqual(alert_from_db.timeseries.timestamps[0], point1.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[0], point1.value)
        self.assertEqual(alert_from_db.timeseries.timestamps[1], point2.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[1], point2.value)

        assert len(alert_from_db.prophet_predictions.timestamps) == 12
        assert len(alert_from_db.prophet_predictions.yhat) == 12
        assert len(alert_from_db.prophet_predictions.yhat_lower) == 12
        assert len(alert_from_db.prophet_predictions.yhat_upper) == 12

        # Should be 11 because the first point is for the current timestamp
        assert alert_from_db.cleanup_predict_config.num_predictions_remaining == 11

        # Verify updating an existing alert
        organization_id = 1001
        project_id = 1002
        config = AnomalyDetectionConfig(
            time_period=30, sensitivity="low", direction="up", expected_seasonality="auto"
        )

        # Adding a new timepoint with an existing timestamp should fail
        with self.assertRaises(Exception):
            alert_data_accessor.save_timepoint(
                external_alert_id,
                point1,
                anomaly_algo_data={
                    "mp_suss": {"dist": 1.0, "idx": 10, "l_idx": -1, "r_idx": -1},
                    "mp_fixed": {"dist": 1.0, "idx": 10, "l_idx": -1, "r_idx": -1},
                },
            )

        # Adding a new timepoint with new timestamp should succeed
        point3 = TimeSeriesPoint(
            timestamp=3000.0,
            value=500.0,
            anomaly=Anomaly(anomaly_type="none", anomaly_score=1.0),
        )
        alert_data_accessor.save_timepoint(
            external_alert_id,
            point3,
            anomaly=MPTimeSeriesAnomalies(
                flags=["none"],
                scores=[0.8],
                matrix_profile_suss=np.array([[1.0, 10, -1, -1]]),
                matrix_profile_fixed=np.array([[1.0, 10, -1, -1]]),
                window_size=1,
                thresholds=[],
                original_flags=["none"],
                use_suss=[True],
                confidence_levels=[
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                ],
                algorithm_types=[
                    AlertAlgorithmType.NONE,
                    AlertAlgorithmType.NONE,
                    AlertAlgorithmType.NONE,
                    AlertAlgorithmType.NONE,
                ],
            ),
            anomaly_algo_data={
                "mp_suss": {"dist": 1.0, "idx": 10, "l_idx": -1, "r_idx": -1},
                "mp_fixed": {"dist": 1.0, "idx": 10, "l_idx": -1, "r_idx": -1},
            },
        )
        alert_from_db = alert_data_accessor.query(external_alert_id=external_alert_id)
        self.assertIsNotNone(alert_from_db, "Should retrieve the alert record")
        self.assertEqual(
            len(alert_from_db.timeseries.timestamps), 3, "Must have three data points in timeseries"
        )
        self.assertEqual(
            len(alert_from_db.timeseries.values), 3, "Must have three data points in timeseries"
        )
        self.assertEqual(alert_from_db.timeseries.timestamps[0], point1.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[0], point1.value)
        self.assertEqual(alert_from_db.timeseries.timestamps[1], point2.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[1], point2.value)
        self.assertEqual(alert_from_db.timeseries.timestamps[2], point3.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[2], point3.value)

        # Resaving an existing alert should update existing data, including overwritinf the entire time series
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=[point1],
            anomalies=MPTimeSeriesAnomalies(
                flags=["none"],
                scores=[1.0],
                matrix_profile_suss=np.array([[1.0, 10, -1, -1]]),
                matrix_profile_fixed=np.array([[1.0, 10, -1, -1]]),
                window_size=1,
                thresholds=[],
                original_flags=["none"],
                use_suss=[True],
                confidence_levels=[
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.MEDIUM,
                ],
                algorithm_types=[
                    AlertAlgorithmType.NONE,
                    AlertAlgorithmType.NONE,
                    AlertAlgorithmType.NONE,
                    AlertAlgorithmType.NONE,
                ],
            ),
            anomaly_algo_data={"window_size": 1},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        with Session() as session:
            self.assertEqual(
                session.query(DbDynamicAlert).count(), 1, "One and only one dynamic alert row saved"
            )
            self.assertEqual(
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=external_alert_id)
                .count(),
                1,
                f"One and only one dynamic alert with the external_alert_id {external_alert_id}",
            )
            # This time around we check the data by directly querying the table instead of using the DbAlertDataAccessor class
            db_dynamic_alert = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=external_alert_id).first()
            )
            self.assertIsNotNone(db_dynamic_alert, "Should retrieve the alert record")
            self.assertEqual(
                db_dynamic_alert.organization_id,
                organization_id,
                "Organization id should match",
            )
            self.assertEqual(db_dynamic_alert.project_id, project_id, "Project id should match")
            self.assertEqual(
                db_dynamic_alert.external_alert_id,
                external_alert_id,
                "external_alert_id id should match",
            )
            self.assertEqual(
                db_dynamic_alert.config["time_period"],
                config.time_period,
                "time_period in config should match",
            )

            self.assertEqual(
                db_dynamic_alert.config["sensitivity"],
                config.sensitivity,
                "time_period in config should match",
            )

            self.assertEqual(
                db_dynamic_alert.config["direction"],
                config.direction,
                "time_period in config should match",
            )

            self.assertEqual(
                db_dynamic_alert.config["expected_seasonality"],
                config.expected_seasonality,
                "time_period in config should match",
            )

            self.assertEqual(
                len(db_dynamic_alert.timeseries), 1, "Must have only one data point in timeseries"
            )
            self.assertEqual(db_dynamic_alert.timeseries[0].timestamp.timestamp(), point1.timestamp)
            self.assertAlmostEqual(db_dynamic_alert.timeseries[0].value, point1.value)

    def test_skip_fixed_window(self):
        """Test that matrix profiles are recalculated when missing from timeseries points"""
        # Create and save alert with timeseries points missing matrix profile data
        organization_id = 100
        project_id = 101
        external_alert_id = 10
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        points = [TimeSeriesPoint(timestamp=100000.0 * i, value=42.42 + i) for i in range(700)]

        anomalies = MPTimeSeriesAnomalies(
            flags=["none"] * 700,
            scores=[1.0] * 700,
            matrix_profile_suss=np.array([[1.0, 10, -1, -1]] * 700),
            matrix_profile_fixed=np.array([[1.0, 10, -1, -1]] * 700),
            window_size=3,
            thresholds=[],
            original_flags=["none"] * 700,
            use_suss=[True] * 700,
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
            ]
            * 700,
            algorithm_types=[AlertAlgorithmType.NONE] * 700,
        )

        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=points,
            anomalies=anomalies,
            anomaly_algo_data={"window_size": 3},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        with Session() as session:
            # Get the actual alert ID first
            db_alert = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=external_alert_id).one()
            )

            # Force the algo_data to be something other than the default (in this case what the old class represented)
            for ts in db_alert.timeseries:
                ts.anomaly_algo_data = {"dist": 1.0, "idx": 10, "l_idx": -1, "r_idx": -1}
            session.commit()

        # Query the alert to trigger recalculation
        alert_from_db = alert_data_accessor.query(external_alert_id=external_alert_id)

        self.assertIsNotNone(alert_from_db)
        self.assertEqual(len(alert_from_db.timeseries.timestamps), 700)
        self.assertEqual(alert_from_db.only_suss, True)

    def test_original_flags_padding(self):
        """Test that original_flags gets padded with 'none' even when there is missing original flag data"""
        organization_id = 100
        project_id = 101
        external_alert_id = 10
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )

        points = [TimeSeriesPoint(timestamp=100.0 * i, value=42.42) for i in range(5)]

        # Create anomalies with only 2 original flags - shorter than timeseries
        anomalies = MPTimeSeriesAnomalies(
            flags=["none"] * 5,
            scores=[1.0] * 5,
            matrix_profile_suss=np.array([[1.0, 10, -1, -1]] * 2),
            matrix_profile_fixed=np.array([[1.0, 10, -1, -1]] * 2),
            window_size=1,
            thresholds=[],
            original_flags=["none", "none"],
            use_suss=[True, True],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
            ],
        )
        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=points,
            anomalies=anomalies,
            anomaly_algo_data={"window_size": 3},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        alert_from_db = alert_data_accessor.query(external_alert_id=external_alert_id)
        self.assertIsNotNone(alert_from_db)

        self.assertEqual(
            len(alert_from_db.timeseries.timestamps),
            len(alert_from_db.anomalies.original_flags),
            "Timeseries and original_flags should be same length",
        )

        expected_flags = ["none"] * 3 + ["none", "none"]
        self.assertEqual(
            alert_from_db.anomalies.original_flags,
            expected_flags,
            "Original flags should be padded with 'none' at the start",
        )

        expected_use_suss = [True] * 3 + [True, True]
        self.assertEqual(
            alert_from_db.anomalies.use_suss,
            expected_use_suss,
            "Use suss should be padded with True at the start",
        )

    def test_queue_data_purge_flag(self):

        # Create and save alert
        organization_id = 100
        project_id = 101
        external_alert_id = 10
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        point1 = TimeSeriesPoint(timestamp=500.0, value=42.42)
        point2 = TimeSeriesPoint(timestamp=1000.0, value=500.0)
        anomalies = MPTimeSeriesAnomalies(
            flags=["none", "none"],
            scores=[1.0, 0.95],
            matrix_profile_suss=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            matrix_profile_fixed=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
            thresholds=[],
            original_flags=["none", "none"],
            use_suss=[True, True],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
            ],
        )
        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=[point1, point2],
            anomalies=anomalies,
            anomaly_algo_data={"window_size": 1},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        alert_data_accessor.queue_data_purge_flag(external_alert_id)

        with Session() as session:

            assert session.query(DbDynamicAlert).count() == 1

            dynamic_alert = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=external_alert_id).first()
            )
            assert dynamic_alert is not None
            assert dynamic_alert.data_purge_flag == TaskStatus.QUEUED
            assert (
                dynamic_alert.last_queued_at < datetime.now()
                and dynamic_alert.last_queued_at > datetime.now() - timedelta(minutes=1)
            )

        with self.assertRaises(Exception):
            alert_data_accessor.queue_data_purge_flag(999)

    def test_can_queue_cleanup_and_predict_task(self):

        # Create and save alert
        organization_id = 100
        project_id = 101
        external_alert_id = 10
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        point1 = TimeSeriesPoint(timestamp=500.0, value=42.42)
        point2 = TimeSeriesPoint(timestamp=1000.0, value=500.0)
        anomalies = MPTimeSeriesAnomalies(
            flags=["none", "none"],
            scores=[1.0, 0.95],
            matrix_profile_suss=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            matrix_profile_fixed=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
            thresholds=[],
            original_flags=["none", "none"],
            use_suss=[True, True],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
            ],
        )
        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=[point1, point2],
            anomalies=anomalies,
            anomaly_algo_data={"window_size": 1},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        assert alert_data_accessor.can_queue_cleanup_predict_task(external_alert_id)

        # Manually adjust dynamic_alert and assert accordingly
        with Session() as session:

            assert session.query(DbDynamicAlert).count() == 1
            dynamic_alert = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=external_alert_id).first()
            )

            dynamic_alert.data_purge_flag = TaskStatus.PROCESSING
            session.commit()
            assert not alert_data_accessor.can_queue_cleanup_predict_task(external_alert_id)

            dynamic_alert.last_queued_at = datetime.now()
            session.commit()
            assert not alert_data_accessor.can_queue_cleanup_predict_task(external_alert_id)
            assert not alert_data_accessor.can_queue_cleanup_predict_task(
                external_alert_id, apply_time_threshold=False
            )

        with self.assertRaises(Exception):
            alert_data_accessor.can_queue_cleanup_predict_task(999)

    def test_delete_alert_data(self):
        # Create and save alert
        organization_id = 100
        project_id = 101
        external_alert_id = 10
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        point1 = TimeSeriesPoint(timestamp=500.0, value=42.42)
        point2 = TimeSeriesPoint(timestamp=1000.0, value=500.0)
        anomalies = MPTimeSeriesAnomalies(
            flags=["none", "none"],
            scores=[1.0, 0.95],
            matrix_profile_suss=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            matrix_profile_fixed=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
            thresholds=[],
            original_flags=["none", "none"],
            use_suss=[True, True],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
            ],
        )
        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=[point1, point2],
            anomalies=anomalies,
            anomaly_algo_data={"window_size": 1},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        # Add some prophet predictions to the alert manually
        with Session() as session:
            alert = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=external_alert_id).first()
            )
            prophet_predictions = [
                DbProphetAlertTimeSeries(
                    dynamic_alert_id=alert.id,
                    timestamp=datetime.now(),
                    yhat=1.0,
                    yhat_lower=0.5,
                    yhat_upper=1.5,
                ),
                DbProphetAlertTimeSeries(
                    dynamic_alert_id=alert.id,
                    timestamp=datetime.now() + timedelta(hours=1),
                    yhat=2.0,
                    yhat_lower=1.5,
                    yhat_upper=2.5,
                ),
            ]
            alert.prophet_predictions = prophet_predictions
            session.commit()

        alert_data_accessor.delete_alert_data(external_alert_id)

        with Session() as session:
            dynamic_alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )

            assert dynamic_alert is None
            assert session.query(DbDynamicAlertTimeSeriesHistory).count() == 2
            assert session.query(DbProphetAlertTimeSeriesHistory).count() == 2

        with self.assertRaises(Exception):
            alert_data_accessor.delete_alert_data(999)

    def test_reset_cleanup_and_predict_task(self):
        # Create and save alert
        organization_id = 100
        project_id = 101
        external_alert_id = 10
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        point1 = TimeSeriesPoint(timestamp=500.0, value=42.42)
        point2 = TimeSeriesPoint(timestamp=1000.0, value=500.0)
        anomalies = MPTimeSeriesAnomalies(
            flags=["none", "none"],
            scores=[1.0, 0.95],
            matrix_profile_suss=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            matrix_profile_fixed=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
            thresholds=[],
            original_flags=["none", "none"],
            use_suss=[True, True],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
            ],
        )
        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=[point1, point2],
            anomalies=anomalies,
            anomaly_algo_data={"window_size": 1},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )

        alert_data_accessor.reset_cleanup_predict_task(external_alert_id)

        with Session() as session:
            dynamic_alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )

            assert dynamic_alert is not None
            assert dynamic_alert.last_queued_at is None
            assert dynamic_alert.data_purge_flag == TaskStatus.NOT_QUEUED

        with self.assertRaises(Exception):
            alert_data_accessor.reset_cleanup_predict_task(999)

    def test_combine_anomalies(self):
        suss_thresholds = [
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=10.0, upper=10.0, lower=10.0),
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=11.0, upper=10.0, lower=10.0),
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=12.0, upper=20.0, lower=20.0),
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=13.0, upper=20.0, lower=20.0),
        ]
        anomalies_suss = MPTimeSeriesAnomaliesSingleWindow(
            flags=["none", "anomaly_higher_confidence", "anomaly_higher_confidence", "none"],
            scores=[1.0, 0.95, 0.95, 0.95],
            matrix_profile=np.array(
                [[1.0, 10, -1, -1], [1.5, 15, -1, -1], [1.0, 10, -1, -1], [1.5, 15, -1, -1]]
            ),
            window_size=30,
            thresholds=[suss_thresholds],
            original_flags=[
                "none",
                "anomaly_higher_confidence",
                "anomaly_higher_confidence",
                "none",
            ],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
            ],
        )
        fixed_thresholds = [
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=10.0, upper=1.0, lower=1.0),
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=11.0, upper=2.0, lower=2.0),
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=12.0, upper=3.0, lower=3.0),
            Threshold(type=ThresholdType.MP_DIST_IQR, timestamp=13.0, upper=4.0, lower=4.0),
        ]

        anomalies_fixed = MPTimeSeriesAnomaliesSingleWindow(
            flags=["none", "none", "none", "none"],
            scores=[1.0, 0.95, 3, 0.95],
            matrix_profile=np.array(
                [[1.0, 10, -1, -1], [12, 15, -1, -1], [19.0, 10, -1, -1], [20, 15, -1, -1]]
            ),
            window_size=10,
            thresholds=[fixed_thresholds],
            original_flags=["none", "none", "none", "none"],
            confidence_levels=[
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.MEDIUM,
            ],
            algorithm_types=[
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
                AlertAlgorithmType.NONE,
            ],
        )

        accessor = DbAlertDataAccessor()
        combined_anomalies = accessor.combine_anomalies(
            anomalies_suss, anomalies_fixed, [True, True, False, True]
        )

        assert combined_anomalies.flags == ["none", "anomaly_higher_confidence", "none", "none"]
        assert combined_anomalies.scores == [1.0, 0.95, 3, 0.95]
        assert np.array_equal(
            combined_anomalies.matrix_profile_suss,
            np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1], [1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
        )
        assert np.array_equal(
            combined_anomalies.matrix_profile_fixed,
            np.array([[1.0, 10, -1, -1], [12, 15, -1, -1], [19.0, 10, -1, -1], [20, 15, -1, -1]]),
        )
        assert combined_anomalies.window_size == 30
        assert len(combined_anomalies.thresholds[0]) == 4
        assert combined_anomalies.thresholds[0][0] == suss_thresholds[0]
        assert combined_anomalies.thresholds[0][1] == suss_thresholds[1]
        assert combined_anomalies.thresholds[0][2] == fixed_thresholds[2]
        assert combined_anomalies.thresholds[0][3] == suss_thresholds[3]
        assert combined_anomalies.original_flags == [
            "none",
            "anomaly_higher_confidence",
            "none",
            "none",
        ]
        assert combined_anomalies.use_suss == [True, True, False, True]

    @patch("seer.anomaly_detection.accessors.stumpy.mparray.mparray")
    def test_hydrate_alert_sets_flags_from_algo_data(self, mock_mparray):
        mock_mparray.return_value = np.array([])

        db_alert = DbDynamicAlert(
            organization_id=1,
            project_id=2,
            external_alert_id=3,
            config={
                "time_period": 15,
                "sensitivity": "medium",
                "direction": "both",
                "expected_seasonality": "auto",
            },
            anomaly_algo_data={"window_size": 10},
            prophet_predictions=[],
            data_purge_flag="not_queued",
            last_queued_at=None,
        )

        ts1 = DbDynamicAlertTimeSeries(
            timestamp=datetime.now(),
            value=1.0,
            anomaly_type="none",
            anomaly_score=0.0,
            anomaly_algo_data=None,
        )

        ts2 = DbDynamicAlertTimeSeries(
            timestamp=datetime.now(),
            value=2.0,
            anomaly_type="none",
            anomaly_score=0.0,
            anomaly_algo_data=None,
        )

        ts3 = DbDynamicAlertTimeSeries(
            timestamp=datetime.now(),
            value=3.0,
            anomaly_type="spike",
            anomaly_score=0.8,
            anomaly_algo_data={
                "mp_suss": {"dist": 0.2, "idx": 2, "l_idx": 1, "r_idx": 3},
                "original_flag": "none",
                "use_suss": True,
                "confidence_level": ConfidenceLevel.HIGH,
                "algorithm_type": AlertAlgorithmType.NONE,
            },
        )

        db_alert.timeseries = [ts1, ts2, ts3]

        mock_algo_config = MagicMock()

        accessor = DbAlertDataAccessor()
        result = accessor._hydrate_alert(db_alert, algo_config=mock_algo_config)

        assert result.anomalies.original_flags[0] == "none"
        assert result.anomalies.original_flags[1] == "none"
        assert result.anomalies.use_suss[0]
        assert result.anomalies.use_suss[1]
        assert result.anomalies.confidence_levels[0] == ConfidenceLevel.MEDIUM
        assert result.anomalies.confidence_levels[1] == ConfidenceLevel.MEDIUM

        assert result.anomalies.use_suss[2]
        assert result.anomalies.confidence_levels[2] == ConfidenceLevel.HIGH
