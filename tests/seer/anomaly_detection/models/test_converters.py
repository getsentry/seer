import unittest

from seer.anomaly_detection.models import converters
from seer.anomaly_detection.models.external import (
    AlertInSeer,
    AnomalyDetectionConfig,
    StoreDataRequest,
)


class TestConverters(unittest.TestCase):
    def test_store_request_to_dynamic_alert(self):
        """
        Tests the logic for converting a DataStoreRequest object to DynamicAlert object
        """
        data_store_request = StoreDataRequest(
            organization_id=100,
            project_id=101,
            config=AnomalyDetectionConfig(
                time_period=30,
                sensitivity="medium",
                direction="both",
                expected_seasonality="auto",
            ),
            alert=AlertInSeer(id=201),
            timeseries=[],
        )

        dynamic_alert = converters.store_request_to_dynamic_alert(data_store_request)
        assert dynamic_alert
        assert dynamic_alert.organization_id == data_store_request.organization_id
        assert dynamic_alert.project_id == data_store_request.project_id
        assert dynamic_alert.config == {
            "time_period": 30,
            "sensitivity": "medium",
            "direction": "both",
            "expected_seasonality": "auto",
        }
        assert dynamic_alert.external_alert_id == data_store_request.alert.id
