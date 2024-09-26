import unittest
from datetime import datetime, timedelta

import numpy as np

from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.models import Anomaly, MPTimeSeriesAnomalies
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.db import DbDynamicAlert, Session, TaskStatus


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
            matrix_profile=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
            thresholds=[0.0, 0.0],
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
        self.assertEqual(alert_from_db.timeseries.timestamps[0](), point1.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[0], point1.value)
        self.assertEqual(alert_from_db.timeseries.timestamps[1](), point2.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[1], point2.value)

        # Verify updating an existing alert
        organization_id = 1001
        project_id = 1002
        config = AnomalyDetectionConfig(
            time_period=30, sensitivity="low", direction="up", expected_seasonality="auto"
        )

        # Adding a new timepoint with an existing timestamp should fail
        with self.assertRaises(Exception):
            alert_data_accessor.save_timepoint(
                external_alert_id, point1, anomaly_algo_data={"dummy": 10}
            )

        # Adding a new timepoint with new timestamp should succeed
        point3 = TimeSeriesPoint(
            timestamp=3000.0,
            value=500.0,
            anomaly_algo_data={"dummy": 10},
            anomaly=Anomaly(anomaly_type="none", anomaly_score=1.0),
        )
        alert_data_accessor.save_timepoint(
            external_alert_id,
            point3,
            anomaly=MPTimeSeriesAnomalies(
                flags=["none"],
                scores=[0.8],
                matrix_profile=np.array([[1.0, 10, -1, -1]]),
                window_size=1,
                thresholds=[0.0],
            ),
            anomaly_algo_data={"dummy": 10},
        )
        alert_from_db = alert_data_accessor.query(external_alert_id=external_alert_id)
        self.assertIsNotNone(alert_from_db, "Should retrieve the alert record")
        self.assertEqual(
            len(alert_from_db.timeseries.timestamps), 3, "Must have three data points in timeseries"
        )
        self.assertEqual(
            len(alert_from_db.timeseries.values), 3, "Must have three data points in timeseries"
        )
        self.assertEqual(alert_from_db.timeseries.timestamps[0](), point1.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[0], point1.value)
        self.assertEqual(alert_from_db.timeseries.timestamps[1](), point2.timestamp)
        self.assertEqual(alert_from_db.timeseries.values[1], point2.value)
        self.assertEqual(alert_from_db.timeseries.timestamps[2](), point3.timestamp)
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
                matrix_profile=np.array([[1.0, 10, -1, -1]]),
                window_size=1,
                thresholds=[0.0],
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
            matrix_profile=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
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

    def test_can_queue_cleanup_task(self):

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
            matrix_profile=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
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

        assert alert_data_accessor.can_queue_cleanup_task(external_alert_id)

        # Manually adjust dynamic_alert and assert accordingly
        with Session() as session:

            assert session.query(DbDynamicAlert).count() == 1
            dynamic_alert = (
                session.query(DbDynamicAlert).filter_by(external_alert_id=external_alert_id).first()
            )

            dynamic_alert.data_purge_flag = TaskStatus.PROCESSING
            session.commit()
            assert not alert_data_accessor.can_queue_cleanup_task(external_alert_id)

            dynamic_alert.last_queued_at = datetime.now()
            session.commit()
            assert not alert_data_accessor.can_queue_cleanup_task(external_alert_id)

        with self.assertRaises(Exception):
            alert_data_accessor.can_queue_cleanup_task(999)

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
            matrix_profile=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
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

        alert_data_accessor.delete_alert_data(external_alert_id)

        with Session() as session:
            dynamic_alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )

            assert dynamic_alert is None

        with self.assertRaises(Exception):
            alert_data_accessor.delete_alert_data(999)

    def test_reset_cleanup_task(self):
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
            matrix_profile=np.array([[1.0, 10, -1, -1], [1.5, 15, -1, -1]]),
            window_size=1,
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

        alert_data_accessor.reset_cleanup_task(external_alert_id)

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
            alert_data_accessor.reset_cleanup_task(999)
