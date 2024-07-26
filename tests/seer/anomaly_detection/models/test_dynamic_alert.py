import unittest

from seer.anomaly_detection.models import DynamicAlert
from seer.anomaly_detection.models.external import TimeSeriesPoint
from seer.db import DbDynamicAlert, Session


class TestDynamicAlert(unittest.TestCase):
    def test_save(self):
        dynamic_alert = DynamicAlert(
            organization_id=100,
            project_id=101,
            external_alert_id=10,
            config={"param1": True, "param2": "up"},
        )
        point1 = TimeSeriesPoint(timestamp=500.0, value=42.42)
        point2 = TimeSeriesPoint(timestamp=1000.0, value=500.0)

        # Verify saving
        dynamic_alert.save([point1, point2])
        with Session() as session:
            self.assertEqual(
                session.query(DbDynamicAlert).count(), 1, "One and only one dynamic alert row saved"
            )
            self.assertEqual(
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=dynamic_alert.external_alert_id)
                .count(),
                1,
                f"One and only one dynamic alert with the external_alert_id {dynamic_alert.external_alert_id}",
            )
            db_dynamic_alert = (
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=dynamic_alert.external_alert_id)
                .first()
            )
            self.assertIsNotNone(db_dynamic_alert, "Should retrieve the alert record")
            self.assertEqual(
                db_dynamic_alert.organization_id,
                dynamic_alert.organization_id,
                "Organization id should match",
            )
            self.assertEqual(
                db_dynamic_alert.project_id, dynamic_alert.project_id, "Project id should match"
            )
            self.assertEqual(
                db_dynamic_alert.external_alert_id,
                dynamic_alert.external_alert_id,
                "external_alert_id id should match",
            )
            self.assertEqual(
                db_dynamic_alert.config, dynamic_alert.config, "config id should match"
            )
            self.assertEqual(
                len(db_dynamic_alert.timeseries), 2, "Must have two data points in timeseries"
            )
            self.assertEqual(db_dynamic_alert.timeseries[0].timestamp.timestamp(), point1.timestamp)
            self.assertEqual(db_dynamic_alert.timeseries[0].value, point1.value)
            self.assertEqual(db_dynamic_alert.timeseries[1].timestamp.timestamp(), point2.timestamp)
            self.assertEqual(db_dynamic_alert.timeseries[1].value, point2.value)

        # Verify updating
        dynamic_alert.organization_id = 1001
        dynamic_alert.project_id = 1002
        dynamic_alert.config = {"param1": False}
        point1.value = 4242.42
        dynamic_alert.save([point1])
        with Session() as session:
            self.assertEqual(
                session.query(DbDynamicAlert).count(), 1, "One and only one dynamic alert row saved"
            )
            self.assertEqual(
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=dynamic_alert.external_alert_id)
                .count(),
                1,
                f"One and only one dynamic alert with the external_alert_id {dynamic_alert.external_alert_id}",
            )
            db_dynamic_alert = (
                session.query(DbDynamicAlert)
                .filter_by(external_alert_id=dynamic_alert.external_alert_id)
                .first()
            )
            self.assertIsNotNone(db_dynamic_alert, "Should retrieve the alert record")
            self.assertEqual(
                db_dynamic_alert.organization_id,
                dynamic_alert.organization_id,
                "Organization id should match",
            )
            self.assertEqual(
                db_dynamic_alert.project_id, dynamic_alert.project_id, "Project id should match"
            )
            self.assertEqual(
                db_dynamic_alert.external_alert_id,
                dynamic_alert.external_alert_id,
                "external_alert_id id should match",
            )
            self.assertEqual(
                db_dynamic_alert.config, dynamic_alert.config, "config id should match"
            )
            self.assertEqual(
                len(db_dynamic_alert.timeseries), 1, "Must have only one data point in timeseries"
            )
            self.assertEqual(db_dynamic_alert.timeseries[0].timestamp.timestamp(), point1.timestamp)
            self.assertAlmostEqual(db_dynamic_alert.timeseries[0].value, 4242.42)
