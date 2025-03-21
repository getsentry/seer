import random
import unittest
from datetime import datetime, timedelta

import numpy as np

from seer.anomaly_detection.accessors import DbAlertDataAccessor
from seer.anomaly_detection.detectors.anomaly_detectors import MPBatchAnomalyDetector
from seer.anomaly_detection.models import MPTimeSeriesAnomalies
from seer.anomaly_detection.models.external import AnomalyDetectionConfig, TimeSeriesPoint
from seer.anomaly_detection.models.timeseries import TimeSeries
from seer.anomaly_detection.tasks import (
    cleanup_disabled_alerts,
    cleanup_old_timeseries_and_prophet_history,
    cleanup_timeseries_and_predict,
)
from seer.db import (
    DbDynamicAlert,
    DbDynamicAlertTimeSeries,
    DbDynamicAlertTimeSeriesHistory,
    DbProphetAlertTimeSeries,
    DbProphetAlertTimeSeriesHistory,
    Session,
    TaskStatus,
)


class TestCleanupTasks(unittest.TestCase):
    def _save_alert(
        self,
        external_alert_id: int,
        num_old_points: int,
        num_new_points: int,
    ):
        # Helper function to save an alert with a given number of old and new points
        organization_id = 100
        project_id = 101
        cur_ts = datetime.now().replace(second=0, microsecond=0).timestamp()
        past_ts = (
            (datetime.now() - timedelta(days=200)).replace(second=0, microsecond=0).timestamp()
        )

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="high", direction="both", expected_seasonality="auto"
        )
        points_old = [
            TimeSeriesPoint(timestamp=past_ts + (i * 3600), value=random.randint(1, 100))
            for i in range(num_old_points)
        ]
        points_new = [
            TimeSeriesPoint(timestamp=cur_ts + (i * 3600), value=random.randint(1, 100))
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
                matrix_profile_suss=np.array([]),
                matrix_profile_fixed=np.array([]),
                window_size=0,
                thresholds=np.array([]),
                original_flags=np.array([]),
                use_suss=np.array([]),
                confidence_levels=[],
                algorithm_types=[],
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

        return external_alert_id, config, points, anomalies, cur_ts

    def _save_prophet_predictions(
        self,
        external_alert_id: int,
        num_points_old: int,
        num_points_new: int,
        first_timestamp: float = None,
    ):

        cur_ts = first_timestamp if first_timestamp else datetime.now().timestamp()
        past_ts = (datetime.fromtimestamp(first_timestamp) - timedelta(days=200)).timestamp()

        # Strip the timestamp to the minute
        cur_ts = int(cur_ts / 60) * 60
        past_ts = int(past_ts / 60) * 60

        timestamps_old = [past_ts + (i * 3600) for i in range(num_points_old)]
        timestamps_new = [cur_ts + (i * 3600) for i in range(num_points_new)]
        timestamps = np.array([*timestamps_old, *timestamps_new])
        yhats = np.array([random.randint(1, 100) for _ in range(num_points_old + num_points_new)])
        yhats_lower = yhats - 10
        yhats_upper = yhats + 10

        prophet_predictions = []

        with Session() as session:

            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )

            assert alert is not None

            for timestamp, yhat, yhat_lower, yhat_upper in zip(
                timestamps, yhats, yhats_lower, yhats_upper
            ):

                prediction = DbProphetAlertTimeSeries(
                    dynamic_alert_id=alert.id,
                    timestamp=datetime.fromtimestamp(timestamp).replace(second=0, microsecond=0),
                    yhat=float(yhat),
                    yhat_lower=float(yhat_lower),
                    yhat_upper=float(yhat_upper),
                )
                session.add(prediction)
                prophet_predictions.append(prediction)

            session.commit()

        return external_alert_id, prophet_predictions

    def test_cleanup_invalid_alert_id(self):
        with self.assertRaises(ValueError, msg="Alert with id 100 not found"):
            cleanup_timeseries_and_predict(100, datetime.now().timestamp())

    def test_cleanup_timeseries_no_points(self):
        # Save alert with no points
        external_alert_id, config, _, _, _ = self._save_alert(0, 0, 0)
        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries_and_predict(external_alert_id, date_threshold)

    def test_only_old_points_deleted(self):
        # Create and save alert with 1000 points (all old)
        external_alert_id, config, points, anomalies, _ = self._save_alert(0, 1000, 0)
        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries_and_predict(external_alert_id, date_threshold)

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
            assert len(alert.prophet_predictions) == 0

    def test_cleanup_timeseries(
        self,
    ):

        # Create and save alert with 2000 points (1000 old, 1000 new)
        external_alert_id, config, points, anomalies, _ = self._save_alert(0, 1000, 1000)

        first_timestamp = points[0].timestamp

        external_alert_id, prophet_predictions = self._save_prophet_predictions(
            0, 1000, 0, first_timestamp
        )

        points_new = points[1000:]
        ts_new = TimeSeries(
            timestamps=np.array([point.timestamp for point in points_new]),
            values=np.array([point.value for point in points_new]),
        )
        anomalies_new = MPBatchAnomalyDetector().detect(
            ts_new, config
        )  # TODO: This test will need to be updated when the combined anomalies are implemented

        old_timeseries_points = []
        with Session() as session:
            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )
            assert alert is not None
            assert len(alert.timeseries) == 2000
            assert len(alert.prophet_predictions) == 1000
            old_timeseries_points = alert.timeseries

        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries_and_predict(external_alert_id, date_threshold)

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
                if new.anomaly_algo_data is not None and algo_data is None:
                    assert new.anomaly_algo_data["mp_suss"] is None
                    assert (
                        "dist" in new.anomaly_algo_data["mp_fixed"]
                        and "idx" in new.anomaly_algo_data["mp_fixed"]
                        and "l_idx" in new.anomaly_algo_data["mp_fixed"]
                        and "r_idx" in new.anomaly_algo_data["mp_fixed"]
                    )

                if new.anomaly_algo_data is not None and algo_data is not None:
                    assert (
                        new.anomaly_algo_data["mp_suss"]["dist"] == algo_data["dist"]
                        and new.anomaly_algo_data["mp_suss"]["idx"] == algo_data["idx"]
                        and new.anomaly_algo_data["mp_suss"]["l_idx"] == algo_data["l_idx"]
                        and new.anomaly_algo_data["mp_suss"]["r_idx"] == algo_data["r_idx"]
                    )
                    # Commenting fixed window logic for now as we are not using it
                    # assert (
                    #     "dist" in new.anomaly_algo_data["mp_fixed"]
                    #     and "idx" in new.anomaly_algo_data["mp_fixed"]
                    #     and "l_idx" in new.anomaly_algo_data["mp_fixed"]
                    #     and "r_idx" in new.anomaly_algo_data["mp_fixed"]
                    # )

        # Fails due to invalid alert_id
        with self.assertRaises(Exception):
            cleanup_timeseries_and_predict(999, date_threshold)

    def test_make_prophet_predictions(self):
        # Create and save alert with 6 points (3 old, 3 new)
        external_alert_id, config, points, anomalies, cur_timestamp = self._save_alert(777, 4800, 1)
        first_timestamp = points[0].timestamp
        external_alert_id, prophet_predictions = self._save_prophet_predictions(
            777, 0, 6, first_timestamp
        )

        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries_and_predict(external_alert_id, date_threshold)

        with Session() as session:
            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id)
                .one_or_none()
            )
            assert alert is not None

            new_predictions = [
                prediction.timestamp
                for prediction in alert.prophet_predictions
                if prediction.timestamp > datetime.fromtimestamp(cur_timestamp)
            ]

            assert len(new_predictions) == (36 * (60 // config.time_period))

    def test_cleanup_disabled_alerts(self):
        # Create and save alerts with old points
        external_alert_id1, _, _, _, _ = self._save_alert(1, 1000, 0)
        external_alert_id2, _, _, _, _ = self._save_alert(2, 500, 0)
        external_alert_id3, _, _, _, _ = self._save_alert(3, 0, 500)
        external_alert_id4, _, _, _, _ = self._save_alert(4, 0, 500)

        # Set last_queued_at to be over 28 days ago for alerts 1 and 2
        with Session() as session:
            for alert_id in [external_alert_id1, external_alert_id2]:
                alert = (
                    session.query(DbDynamicAlert)
                    .filter(DbDynamicAlert.external_alert_id == alert_id)
                    .one_or_none()
                )
                assert alert is not None
                alert.last_queued_at = datetime.now() - timedelta(days=29)

            for alert_id in [external_alert_id3]:
                alert = (
                    session.query(DbDynamicAlert)
                    .filter(DbDynamicAlert.external_alert_id == alert_id)
                    .one_or_none()
                )
                assert alert is not None
                alert.created_at = datetime.now() - timedelta(days=29)
                alert.last_queued_at = None

            session.commit()

        cleanup_disabled_alerts()

        with Session() as session:
            for alert_id in [external_alert_id1, external_alert_id2, external_alert_id3]:
                alert = (
                    session.query(DbDynamicAlert)
                    .filter(DbDynamicAlert.external_alert_id == alert_id)
                    .one_or_none()
                )
                assert alert is None

                timeseries = (
                    session.query(DbDynamicAlertTimeSeries)
                    .filter(DbDynamicAlertTimeSeries.dynamic_alert_id == alert_id)
                    .all()
                )
                assert len(timeseries) == 0

                prophet_predictions_history = (
                    session.query(DbProphetAlertTimeSeriesHistory)
                    .filter(DbProphetAlertTimeSeriesHistory.alert_id == alert_id)
                    .all()
                )
                assert len(prophet_predictions_history) == 0

        # Confirm that alert 4 and its respective timeseries are not deleted
        with Session() as session:
            alert = (
                session.query(DbDynamicAlert)
                .filter(DbDynamicAlert.external_alert_id == external_alert_id4)
                .one_or_none()
            )

            assert alert is not None
            assert len(alert.timeseries) == 500

    def test_cleanup_old_timeseries_history(self):
        # Create and save alert with 1000 points (all old)
        external_alert_id, _, _, _, _ = self._save_alert(0, 1000, 0)
        date_threshold = (datetime.now() - timedelta(days=28)).timestamp()
        cleanup_timeseries_and_predict(external_alert_id, date_threshold)

        # Confirm the historical table is populated with 1000 points

        with Session() as session:
            history = (
                session.query(DbDynamicAlertTimeSeriesHistory)
                .filter(DbDynamicAlertTimeSeriesHistory.alert_id == external_alert_id)
                .all()
            )
            assert len(history) == 1000

        # Timeseries History should be deleted as these points have been added over 90 days ago

        with Session() as session:

            cleanup_old_timeseries_and_prophet_history()

            history = (
                session.query(DbDynamicAlertTimeSeriesHistory)
                .filter(DbDynamicAlertTimeSeriesHistory.alert_id == external_alert_id)
                .all()
            )

            assert len(history) == 0

            prophet_predictions_history = (
                session.query(DbProphetAlertTimeSeriesHistory)
                .filter(DbProphetAlertTimeSeriesHistory.alert_id == external_alert_id)
                .all()
            )
            assert len(prophet_predictions_history) == 0
