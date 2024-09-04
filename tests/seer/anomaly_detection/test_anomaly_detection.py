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
)
from tests.seer.anomaly_detection.timeseries.timeseries import context

# from typing import List


class TestAnomalyDetection(unittest.TestCase):

    def test_detect_anomalies(self):

        # Set up respective params for detection types (batch, online, combo)

        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        # Batch Detection
        contextBatchTimeseries = context

        requestBatch = DetectAnomaliesRequest(
            organization_id=1, project_id=1, config=config, context=contextBatchTimeseries
        )

        responseBatch = AnomalyDetection().detect_anomalies(request=requestBatch)

        # response is a timeseries (DetectAnomaliesResponse)
        self.assertIsInstance(responseBatch, DetectAnomaliesResponse)

        # assert that specific values of response are expected?

        # Online Detection
        contextOnline = AlertInSeer(id=1000, cur_window=TimeSeriesPoint(timestamp=501, value=0.5))

        requestOnline = DetectAnomaliesRequest(
            organization_id=1, project_id=1, config=config, context=contextOnline
        )

        responseOnline = AnomalyDetection().detect_anomalies(request=requestOnline)

        # assert that the response is a timeseries (DetectAnomaliesResponse)
        self.assertIsInstance(responseOnline, DetectAnomaliesResponse)
        # self.assertIsInstance(responseOnline.timeseries, List[TimeSeriesPoint])

        # TODO: Combo detection

    def test_store_data(self):

        # Set up request and accessor
        organization_id = 1
        project_id = 1
        alert = AlertInSeer(id=1)
        # TODO: Consider looping through config options? Would that be too expensive/brittle?
        config = AnomalyDetectionConfig(
            time_period=15, sensitivity="low", direction="both", expected_seasonality="auto"
        )

        # Load in sample timeseries
        ts = context

        request = StoreDataRequest(
            organization_id=organization_id,
            project_id=project_id,
            alert=alert,
            config=config,
            timeseries=ts,
        )
        alert_data_accessor = DbAlertDataAccessor()

        response = AnomalyDetection().store_data(
            request=request, alert_data_accessor=alert_data_accessor
        )

        # Successful
        self.assertEqual(
            response,
            StoreDataResponse(success=True),
            "Store Data Response should be successful",
        )
