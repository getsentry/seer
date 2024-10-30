import random
import unittest
from datetime import datetime, timedelta

import numpy as np

from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.detectors.anomaly_detectors import MPBatchAnomalyDetector
from seer.anomaly_detection.models import MPTimeSeriesAnomalies
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.anomaly_detection.models.timeseries import TimeSeries
from seer.anomaly_detection.tasks import cleanup_timeseries
from seer.db import DbDynamicAlert, Session, TaskStatus


class TestCleanupTasks(unittest.TestCase):
    def _save_alert(self, num_old_points: int, num_new_points: int):
        # Helper function to save an alert with a given number of old and new points
        external_alert_id = random.randint(1, 100)
        organization_id = 100
        project_id = 101
        cur_ts = datetime.now().timestamp()
        past_ts = (datetime.now() - timedelta(days=100)).timestamp()

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        points_old = [
            TimeSeriesPoint(timestamp=past_ts + (i * 1000), value=random.randint(1, 100))
            for i in range(num_old_points)
        ]
        points_new = [
            TimeSeriesPoint(timestamp=cur_ts + (i * 1000), value=random.randint(1, 100))
            for i in range(num_new_points)
        ]
        points = [*points_old, *points_new]

        ts = TimeSeries(
            timestamps=np.array([point.timestamp for point in points]),
            values=np.array([point.value for point in points]),
        )
        if len(ts.values) == 0:
            anomalies = MPTimeSeriesAnomalies(
                flags=np.array([]),
                scores=np.array([]),
                matrix_profile=np.array([]),
                window_size=0,
                thresholds=np.array([]),
            )
        else:
            anomalies = MPBatchAnomalyDetector().detect(ts, config)

        alert_data_accessor = DbAlertDataAccessor()
        alert_data_accessor.save_alert(
            organization_id=organization_id,
            project_id=project_id,
            external_alert_id=external_alert_id,
            config=config,
            timeseries=points,
            anomalies=anomalies,
            anomaly_algo_data={"window_size": anomalies.window_size},
            data_purge_flag=TaskStatus.NOT_QUEUED,
        )
        with Session() as session:
            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )
            assert alert is not None
            assert len(alert.timeseries) == len(points)

        return external_alert_id, config, points, anomalies

    def test_cleanup_invalid_alert_id(self):
        with self.assertRaises(ValueError, msg="Alert with id 100 not found"):
            cleanup_timeseries(100, datetime.now().timestamp())

    def test_cleanup_timeseries_no_points(self):
        # Save alert with no points
        external_alert_id, config, _, _ = self._save_alert(0, 0)
        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries(external_alert_id, date_threshold)

    def test_only_old_points_deleted(self):
        # Create and save alert with 1000 points (all old)
        external_alert_id, config, points, anomalies = self._save_alert(1000, 0)
        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries(external_alert_id, date_threshold)

        # Confirm if points are being deleted and matrix profile recalculated after cleanup task is called
        with Session() as session:
            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )
            assert alert is not None
            assert alert.data_purge_flag == TaskStatus.NOT_QUEUED
            assert len(alert.timeseries) == 0
            assert "window_size" in alert.anomaly_algo_data
            assert alert.anomaly_algo_data["window_size"] == 0

    def test_cleanup_timeseries(self):

        # Create and save alert with 2000 points (1000 old, 1000 new)
        external_alert_id, config, points, anomalies = self._save_alert(1000, 1000)
        points_new = points[1000:]
        ts_new = TimeSeries(
            timestamps=np.array([point.timestamp for point in points_new]),
            values=np.array([point.value for point in points_new]),
        )
        anomalies_new = MPBatchAnomalyDetector().detect(ts_new, config)

        old_timeseries_points = []
        with Session() as session:
            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )
            assert alert is not None
            assert len(alert.timeseries) == 2000
            old_timeseries_points = alert.timeseries

        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries(external_alert_id, date_threshold)

        # Confirm if points are being deleted and matrix profile recalculated after cleanup task is called
        with Session() as session:
            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )
            assert alert is not None
            assert alert.data_purge_flag == TaskStatus.NOT_QUEUED
            assert len(alert.timeseries) == 1000
            assert "window_size" in alert.anomaly_algo_data
            assert alert.anomaly_algo_data["window_size"] == anomalies_new.window_size
            new_timeseries_points = alert.timeseries
            assert old_timeseries_points != new_timeseries_points

            for old, new, algo_data in zip(
                old_timeseries_points[1000:],
                new_timeseries_points,
                anomalies_new.get_anomaly_algo_data(len(points_new)),
            ):
                assert new.timestamp.timestamp() > date_threshold
                assert old.timestamp == new.timestamp
                assert old.value == new.value
                assert new.anomaly_algo_data == algo_data

        # Fails due to invalid alert_id
        with self.assertRaises(Exception):
            cleanup_timeseries(999, date_threshold)
