import json
import time
import unittest
from typing import cast
from unittest import mock

import httpx
import pytest
from celery import Celery
from celery.apps.worker import Worker
from johen import generate
from johen.pytest import parametrize
from openai import APITimeoutError
from sqlalchemy import text

from seer.anomaly_detection.models.external import (
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
from seer.app import app
from seer.automation.assisted_query.models import (
    Chart,
    CreateCacheRequest,
    CreateCacheResponse,
    TranslateRequest,
    TranslateResponse,
)
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixEvaluationRequest,
    AutofixRequest,
)
from seer.automation.autofix.runs import create_initial_autofix_run
from seer.automation.autofixability import AutofixabilityModel
from seer.automation.state import LocalMemoryState
from seer.automation.summarize.models import (
    GetFixabilityScoreRequest,
    SpanInsight,
    SummarizeIssueRequest,
    SummarizeIssueResponse,
    SummarizeIssueScores,
    SummarizeTraceRequest,
    SummarizeTraceResponse,
)
from seer.configuration import AppConfig, provide_test_defaults
from seer.db import DbGroupingRecord, DbSmokeTest, ProcessRequest, Session
from seer.dependency_injection import Module, resolve
from seer.exceptions import ClientError, ServerError
from seer.grouping.grouping import CreateGroupingRecordData, CreateGroupingRecordsRequest
from seer.inference_models import dummy_deferred, reset_loading_state, start_loading
from seer.smoke_test import smoke_test


@pytest.fixture(autouse=True)
def mock_severity_score():
    # Create a mock instance with a dummy severity_score method
    with mock.patch("seer.inference_models.SeverityInference") as mock_severity_inference:
        mock_instance = mock_severity_inference.return_value
        mock_instance.severity_score.return_value = [0, 1]
        yield


class TestSeer(unittest.TestCase):
    def get_sample_data(self):
        trend_data = {
            "data": [
                [1681934400, [{"count": 680.0}]],
                [1681938000, [{"count": 742.0}]],
                [1681941600, [{"count": 631.0}]],
                [1681945200, [{"count": 567.0}]],
                [1681948800, [{"count": 538.0}]],
                [1681952400, [{"count": 619.0}]],
                [1681956000, [{"count": 577.0}]],
                [1681959600, [{"count": 551.5}]],
                [1681963200, [{"count": 589.0}]],
                [1681966800, [{"count": 531.5}]],
                [1681970400, [{"count": 562.0}]],
                [1681974000, [{"count": 528.0}]],
                [1681977600, [{"count": 587.0}]],
                [1681981200, [{"count": 569.0}]],
                [1681984800, [{"count": 615.0}]],
                [1681988400, [{"count": 611.0}]],
                [1681992000, [{"count": 630.0}]],
                [1681995600, [{"count": 637.0}]],
                [1681999200, [{"count": 661.0}]],
                [1682002800, [{"count": 671.0}]],
                [1682006400, [{"count": 638.0}]],
                [1682010000, [{"count": 670.0}]],
                [1682013600, [{"count": 666.5}]],
                [1682017200, [{"count": 650.0}]],
                [1682020800, [{"count": 641.0}]],
                [1682024400, [{"count": 682.0}]],
                [1682028000, [{"count": 627.0}]],
                [1682031600, [{"count": 611.5}]],
                [1682035200, [{"count": 557.0}]],
                [1682038800, [{"count": 526.0}]],
                [1682042400, [{"count": 538.0}]],
                [1682046000, [{"count": 598.0}]],
                [1682049600, [{"count": 563.0}]],
                [1682053200, [{"count": 588.0}]],
                [1682056800, [{"count": 610.5}]],
                [1682060400, [{"count": 576.0}]],
                [1682064000, [{"count": 598.0}]],
                [1682067600, [{"count": 560.5}]],
                [1682071200, [{"count": 623.0}]],
                [1682074800, [{"count": 557.0}]],
                [1682078400, [{"count": 883.5}]],
                [1682082000, [{"count": 972.0}]],
                [1682085600, [{"count": 844.5}]],
                [1682089200, [{"count": 929.0}]],
                [1682092800, [{"count": 1071.0}]],
                [1682096400, [{"count": 1090.0}]],
                [1682100000, [{"count": 883.0}]],
                [1682103600, [{"count": 913.0}]],
                [1682107200, [{"count": 850.0}]],
                [1682110800, [{"count": 911.5}]],
                [1682114400, [{"count": 814.0}]],
                [1682118000, [{"count": 786.0}]],
                [1682121600, [{"count": 660.5}]],
                [1682125200, [{"count": 605.5}]],
                [1682128800, [{"count": 551.0}]],
                [1682132400, [{"count": 430.0}]],
                [1682136000, [{"count": 635.0}]],
                [1682139600, [{"count": 569.0}]],
                [1682143200, [{"count": 491.5}]],
                [1682146800, [{"count": 536.0}]],
                [1682150400, [{"count": 533.0}]],
                [1682154000, [{"count": 393.0}]],
                [1682157600, [{"count": 534.0}]],
                [1682161200, [{"count": 498.0}]],
                [1682164800, [{"count": 645.5}]],
                [1682168400, [{"count": 521.0}]],
                [1682172000, [{"count": 485.5}]],
                [1682175600, [{"count": 668.0}]],
                [1682179200, [{"count": 654.0}]],
                [1682182800, [{"count": 520.5}]],
                [1682186400, [{"count": 619.5}]],
                [1682190000, [{"count": 549.5}]],
                [1682193600, [{"count": 560.0}]],
                [1682197200, [{"count": 550.5}]],
                [1682200800, [{"count": 604.5}]],
                [1682204400, [{"count": 623.0}]],
                [1682208000, [{"count": 561.0}]],
                [1682211600, [{"count": 598.0}]],
                [1682215200, [{"count": 743.5}]],
                [1682218800, [{"count": 658.0}]],
                [1682222400, [{"count": 704.0}]],
                [1682226000, [{"count": 606.0}]],
                [1682229600, [{"count": 508.0}]],
                [1682233200, [{"count": 486.0}]],
                [1682236800, [{"count": 554.0}]],
                [1682240400, [{"count": 543.0}]],
                [1682244000, [{"count": 435.0}]],
                [1682247600, [{"count": 561.5}]],
                [1682251200, [{"count": 518.0}]],
                [1682254800, [{"count": 661.0}]],
                [1682258400, [{"count": 514.5}]],
                [1682262000, [{"count": 581.5}]],
                [1682265600, [{"count": 503.0}]],
                [1682269200, [{"count": 598.0}]],
                [1682272800, [{"count": 520.5}]],
                [1682276400, [{"count": 494.0}]],
                [1682280000, [{"count": 785.0}]],
                [1682283600, [{"count": 383.0}]],
                [1682287200, [{"count": 457.0}]],
                [1682290800, [{"count": 464.0}]],
                [1682294400, [{"count": 559.0}]],
                [1682298000, [{"count": 489.5}]],
                [1682301600, [{"count": 746.0}]],
                [1682305200, [{"count": 609.0}]],
                [1682308800, [{"count": 587.0}]],
                [1682312400, [{"count": 1263.5}]],
                [1682316000, [{"count": 744.5}]],
                [1682319600, [{"count": 805.5}]],
                [1682323200, [{"count": 987.0}]],
                [1682326800, [{"count": 869.0}]],
                [1682330400, [{"count": 779.5}]],
                [1682334000, [{"count": 880.5}]],
                [1682337600, [{"count": 929.5}]],
                [1682341200, [{"count": 862.0}]],
                [1682344800, [{"count": 884.0}]],
                [1682348400, [{"count": 895.0}]],
                [1682352000, [{"count": 939.0}]],
                [1682355600, [{"count": 1183.0}]],
                [1682359200, [{"count": 922.0}]],
                [1682362800, [{"count": 953.0}]],
                [1682366400, [{"count": 1373.5}]],
                [1682370000, [{"count": 963.0}]],
                [1682373600, [{"count": 719.5}]],
                [1682377200, [{"count": 1024.5}]],
                [1682380800, [{"count": 940.0}]],
                [1682384400, [{"count": 630.0}]],
                [1682388000, [{"count": 943.0}]],
                [1682391600, [{"count": 796.5}]],
                [1682395200, [{"count": 695.5}]],
                [1682398800, [{"count": 965.5}]],
                [1682402400, [{"count": 921.5}]],
                [1682406000, [{"count": 896.0}]],
                [1682409600, [{"count": 962.0}]],
                [1682413200, [{"count": 1099.0}]],
                [1682416800, [{"count": 837.0}]],
                [1682420400, [{"count": 915.0}]],
                [1682424000, [{"count": 978.5}]],
                [1682427600, [{"count": 1051.5}]],
                [1682431200, [{"count": 1125.0}]],
                [1682434800, [{"count": 838.5}]],
                [1682438400, [{"count": 936.0}]],
                [1682442000, [{"count": 1170.0}]],
                [1682445600, [{"count": 1057.5}]],
                [1682449200, [{"count": 1097.0}]],
                [1682452800, [{"count": 1034.0}]],
                [1682456400, [{"count": 1219.0}]],
                [1682460000, [{"count": 936.0}]],
                [1682463600, [{"count": 911.0}]],
                [1682467200, [{"count": 841.0}]],
                [1682470800, [{"count": 790.0}]],
                [1682474400, [{"count": 1015.0}]],
                [1682478000, [{"count": 651.5}]],
                [1682481600, [{"count": 839.0}]],
                [1682485200, [{"count": 820.0}]],
                [1682488800, [{"count": 783.0}]],
                [1682492400, [{"count": 853.0}]],
                [1682496000, [{"count": 811.0}]],
                [1682499600, [{"count": 971.0}]],
                [1682503200, [{"count": 931.0}]],
                [1682506800, [{"count": 1028.0}]],
                [1682510400, [{"count": 828.0}]],
                [1682514000, [{"count": 817.0}]],
                [1682517600, [{"count": 971.0}]],
                [1682521200, [{"count": 1235.0}]],
                [1682524800, [{"count": 1080.0}]],
                [1682528400, [{"count": 974.0}]],
                [1682532000, [{"count": 1016.0}]],
                [1682535600, [{"count": 938.0}]],
                [1682539200, [{"count": 738.5}]],
                [1682542800, [{"count": 924.0}]],
                [1682546400, [{"count": 900.0}]],
                [1682550000, [{"count": 958.0}]],
                [1682553600, [{"count": 974.0}]],
                [1682557200, [{"count": 756.0}]],
                [1682560800, [{"count": 912.0}]],
                [1682564400, [{"count": 924.0}]],
                [1682568000, [{"count": 822.0}]],
                [1682571600, [{"count": 776.0}]],
                [1682575200, [{"count": 979.0}]],
                [1682578800, [{"count": 606.0}]],
                [1682582400, [{"count": 1109.5}]],
                [1682586000, [{"count": 884.5}]],
                [1682589600, [{"count": 833.0}]],
                [1682593200, [{"count": 897.0}]],
                [1682596800, [{"count": 844.0}]],
                [1682600400, [{"count": 1014.0}]],
            ],
            "request_start": 1681934400,
            "request_end": 1683144000,
            "data_start": 1681934400,
            "data_end": 1683144000,
        }
        input_data = {
            "data": {"sentry,/api/0/organizations/{organization_slug}/issues/": trend_data},
            "sort": "-trend_percentage()",
        }
        return input_data

    def test_empty_txns_dataset(self):
        response = app.test_client().post(
            "/trends/breakpoint-detector",
            data=json.dumps({"data": {}, "sort": "trend_percentage()"}),
            content_type="application/json",
        )

        actual_output = json.loads(response.get_data(as_text=True))

        expected_output = {"data": []}

        assert actual_output == expected_output

    def test_breakpoint_output(self):
        input_data = self.get_sample_data()

        response = app.test_client().post(
            "/trends/breakpoint-detector",
            data=json.dumps(input_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        request_start = input_data["data"][
            "sentry,/api/0/organizations/{organization_slug}/issues/"
        ]["request_start"]
        request_end = input_data["data"]["sentry,/api/0/organizations/{organization_slug}/issues/"][
            "request_end"
        ]

        expected_output = {
            "data": [
                {
                    "project": "sentry",
                    "transaction": "/api/0/organizations/{organization_slug}/issues/",
                    "data_start": 1681934400,
                    "data_end": 1683144000,
                    "aggregate_range_1": 619.2740384615385,
                    "aggregate_range_2": 921.5060975609756,
                    "unweighted_t_value": -14.710852035668289,
                    "unweighted_p_value": round(7.386632642605361e-32, 10),
                    "trend_percentage": 1.4880425148295766,
                    "trend_difference": 302.23205909943715,
                    "breakpoint": 1682308800,
                    "change": "regression",
                    "absolute_percentage_change": 1.4880425148295766,
                    "request_start": request_start,
                    "request_end": request_end,
                }
            ]
        }

        actual_output = json.loads(response.get_data(as_text=True))

        assert actual_output == expected_output

    def test_no_data_after_request_start(self):
        mid = 5  # needs to be greater than 3 because that's the minimum time series length
        input_data = {
            "data": {
                "sentry,/api/0/organizations/{organization_slug}/issues/": {
                    "data": [[ts, [{"count": 1 if ts <= mid else 0}]] for ts in range(2 * mid)],
                    "request_start": mid,
                    "request_end": 2 * mid,
                    "data_start": 0,
                    "data_end": 2 * mid,
                },
            },
            "sort": "-trend_percentage()",
        }

        response = app.test_client().post(
            "/trends/breakpoint-detector",
            data=json.dumps(input_data),
            content_type="application/json",
        )
        output = json.loads(response.get_data(as_text=True))
        assert output == {"data": []}

    def test_similarity_grouping_record_endpoint_valid(self):
        """Test the similarity grouping record endpoint"""
        hashes = [str(i) * 32 for i in range(5)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                )
                for i in range(5)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(5)],
        )

        response = app.test_client().post(
            "/v0/issues/similar-issues/grouping-record",
            data=record_requests.json(),
            content_type="application/json",
        )
        output = json.loads(response.get_data(as_text=True))
        assert output == {"success": True, "groups_with_neighbor": {}}
        with Session() as session:
            records = session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes))
            for i in range(5):
                assert records[i] is not None

    def test_similarity_grouping_record_endpoint_invalid(self):
        """
        Test the similarity grouping record endpoint is unsuccessful when input lists are of
        different lengths
        """
        hashes = [str(i) * 32 for i in range(5, 7)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                )
                for i in range(2)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(3)],
        )

        response = app.test_client().post(
            "/v0/issues/similar-issues/grouping-record",
            data=record_requests.json(),
            content_type="application/json",
        )
        output = json.loads(response.get_data(as_text=True))
        assert output == {"success": False, "groups_with_neighbor": {}}
        with Session() as session:
            assert (
                session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes)).first()
                is None
            )

    @mock.patch("seer.app.run_autofix_evaluation")
    def test_autofix_evaluation_start_endpoint(self, mock_run_autofix_evaluation):
        # Prepare test data
        test_data = AutofixEvaluationRequest(
            dataset_name="test_dataset",
            run_name="test_run",
            run_description="Test run description",
            run_type="full",
            test=False,
            random_for_test=False,
            run_on_item_id=None,
            n_runs_per_item=1,
        )

        # Make a POST request to the endpoint
        response = app.test_client().post(
            "/v1/automation/autofix/evaluations/start",
            data=test_data.json(),
            content_type="application/json",
        )

        # Assert that the response is correct
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.get_data(as_text=True))
        self.assertEqual(response_data, {"started": True, "run_id": -1})

        # Assert that run_autofix_evaluation was called with the correct arguments
        mock_run_autofix_evaluation.assert_called_once_with(test_data)

    @mock.patch("seer.app.run_autofix_evaluation")
    def test_autofix_evaluation_start_endpoint_test_mode(self, mock_run_autofix_evaluation):
        # Prepare test data with test mode enabled
        test_data = AutofixEvaluationRequest(
            dataset_name="test_dataset",
            run_name="test_run",
            run_description="Test run description",
            run_type="root_cause",
            test=True,
            random_for_test=False,
            run_on_item_id=None,
            n_runs_per_item=1,
        )

        # Make a POST request to the endpoint
        response = app.test_client().post(
            "/v1/automation/autofix/evaluations/start",
            data=test_data.json(),
            content_type="application/json",
        )

        # Assert that the response is correct
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.get_data(as_text=True))
        self.assertEqual(response_data, {"started": True, "run_id": -1})

        # Assert that run_autofix_evaluation was called with the correct arguments
        mock_run_autofix_evaluation.assert_called_once_with(test_data)

    @mock.patch("seer.app.run_autofix_evaluation")
    def test_autofix_evaluation_start_endpoint_with_run_on_item_id(
        self, mock_run_autofix_evaluation
    ):
        # Prepare test data with run_on_item_id
        test_data = AutofixEvaluationRequest(
            dataset_name="test_dataset",
            run_name="test_run",
            run_description="Test run description",
            run_type="full",
            test=False,
            random_for_test=False,
            run_on_item_id="specific_item_id",
            n_runs_per_item=1,
        )

        # Make a POST request to the endpoint
        response = app.test_client().post(
            "/v1/automation/autofix/evaluations/start",
            data=test_data.json(),
            content_type="application/json",
        )

        # Assert that the response is correct
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.get_data(as_text=True))
        self.assertEqual(response_data, {"started": True, "run_id": -1})

        # Assert that run_autofix_evaluation was called with the correct arguments
        mock_run_autofix_evaluation.assert_called_once_with(test_data)

    @mock.patch("seer.app.run_summarize_issue")
    def test_summarize_issue_endpoint_timeout(self, mock_run_summarize_issue):
        """Test that summarize_issue_endpoint handles APITimeoutError correctly"""
        mock_run_summarize_issue.side_effect = APITimeoutError(
            request=httpx.Request(
                method="POST", url="http://localhost/v1/automation/summarize/issue"
            )
        )
        test_data = next(generate(SummarizeIssueRequest))

        response = app.test_client().post(
            "/v1/automation/summarize/issue",
            data=test_data.json(),
            content_type="application/json",
        )

        assert response.status_code == 504  # GatewayTimeout
        mock_run_summarize_issue.assert_called_once_with(test_data)

    @mock.patch("seer.app.run_summarize_issue")
    def test_summarize_issue_endpoint_internal_error(self, mock_run_summarize_issue):
        """Test that summarize_issue_endpoint handles general exceptions correctly"""
        mock_run_summarize_issue.side_effect = Exception("Test error")
        test_data = next(generate(SummarizeIssueRequest))

        response = app.test_client().post(
            "/v1/automation/summarize/issue",
            data=test_data.json(),
            content_type="application/json",
        )

        assert response.status_code == 500  # InternalServerError
        mock_run_summarize_issue.assert_called_once_with(test_data)

    @mock.patch("seer.app.run_fixability_score")
    def test_get_fixability_score_endpoint_success(self, mock_run_fixability_score):
        """Test a successful run of get_fixability_score endpoint"""
        # Create a mock response
        mock_response = SummarizeIssueResponse(
            group_id=123,
            headline="Test Error",
            whats_wrong="Something is broken",
            trace="No related issues",
            possible_cause="Bad code",
            scores=SummarizeIssueScores(
                possible_cause_confidence=0.8,
                possible_cause_novelty=0.6,
                fixability_score=0.75,
                fixability_score_version=3,
                is_fixable=True,
            ),
        )
        mock_run_fixability_score.return_value = mock_response

        # Create test data
        test_data = GetFixabilityScoreRequest(group_id=123)

        # Make the request
        response = app.test_client().post(
            "/v1/automation/summarize/fixability",
            data=test_data.json(),
            content_type="application/json",
        )

        # Assertions
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data["group_id"] == 123
        assert response_data["scores"]["fixability_score"] == 0.75
        assert response_data["scores"]["fixability_score_version"] == 3
        assert response_data["scores"]["is_fixable"] is True

        expected_test_data, autofixability_model = mock_run_fixability_score.call_args[0]
        assert expected_test_data == test_data
        assert isinstance(autofixability_model, AutofixabilityModel)

    @mock.patch("seer.app.run_fixability_score")
    def test_get_fixability_score_endpoint_error(self, mock_run_fixability_score):
        """Test that get_fixability_score_endpoint handles exceptions correctly"""
        mock_run_fixability_score.side_effect = ValueError("No issue summary found")
        test_data = GetFixabilityScoreRequest(group_id=123)

        response = app.test_client().post(
            "/v1/automation/summarize/fixability",
            data=test_data.json(),
            content_type="application/json",
        )

        assert response.status_code == 500  # InternalServerError
        expected_test_data, autofixability_model = mock_run_fixability_score.call_args[0]
        assert expected_test_data == test_data
        assert isinstance(autofixability_model, AutofixabilityModel)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.detect_anomalies")
    def test_detect_anomalies_endpoint_success(self, mock_detect_anomalies):
        """Test a successful run of detect_anomalies end point"""
        mock_detect_anomalies.return_value = DetectAnomaliesResponse(success=True)
        test_data = next(generate(DetectAnomaliesRequest))

        response = app.test_client().post(
            "/v1/anomaly-detection/detect",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["success"] is True
        mock_detect_anomalies.assert_called_once_with(test_data)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.detect_anomalies")
    def test_combo_detect_anomalies_endpoint_success(self, mock_detect_anomalies):
        """Test a successful run of detect_anomalies end point"""
        mock_detect_anomalies.return_value = DetectAnomaliesResponse(success=True)
        test_data = DetectAnomaliesRequest(
            organization_id=1,
            project_id=1,
            config=AnomalyDetectionConfig(
                time_period=60,
                sensitivity="medium",
                direction="both",
                expected_seasonality="auto",
            ),
            context=TimeSeriesWithHistory(
                history=[TimeSeriesPoint(timestamp=1, value=1)],
                current=[TimeSeriesPoint(timestamp=2, value=2)],
            ),
        )

        response = app.test_client().post(
            "/v1/anomaly-detection/detect",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["success"] is True
        mock_detect_anomalies.assert_called_once_with(test_data)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.detect_anomalies")
    def test_detect_anomalies_endpoint_client_error(self, mock_detect_anomalies):
        """Test that detect_anomalies endpoint handles client errors correctly"""
        mock_detect_anomalies.side_effect = ClientError("Test error")
        test_data = next(generate(DetectAnomaliesRequest))

        response = app.test_client().post(
            "/v1/anomaly-detection/detect",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["message"] == "Test error"
        assert response.json["success"] is False
        mock_detect_anomalies.assert_called_once_with(test_data)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.detect_anomalies")
    def test_detect_anomalies_endpoint_server_error(self, mock_detect_anomalies):
        """Test that detect_anomalies endpoint handles server errors correctly"""
        mock_detect_anomalies.side_effect = ServerError("Test server error")
        test_data = next(generate(DetectAnomaliesRequest))

        with pytest.raises(ServerError, match="Test server error"):
            app.test_client().post(
                "/v1/anomaly-detection/detect",
                data=test_data.model_dump_json(),
                content_type="application/json",
            )

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.store_data")
    def test_store_data_endpoint_success(self, mock_store_data):
        """Test a successful run of store_data end point"""
        mock_store_data.return_value = StoreDataResponse(success=True)
        test_data = next(generate(StoreDataRequest))

        response = app.test_client().post(
            "/v1/anomaly-detection/store",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["success"] is True
        mock_store_data.assert_called_once_with(test_data)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.store_data")
    def test_store_data_endpoint_client_error(self, mock_store_data):
        """Test that store_data endpoint handles client errors correctly"""
        mock_store_data.side_effect = ClientError("Test error")
        test_data = next(generate(StoreDataRequest))

        response = app.test_client().post(
            "/v1/anomaly-detection/store",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["message"] == "Test error"
        assert response.json["success"] is False
        mock_store_data.assert_called_once_with(test_data)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.store_data")
    def test_store_data_endpoint_server_error(self, mock_store_data):
        """Test that store_data endpoint handles server errors correctly"""
        mock_store_data.side_effect = ServerError("Test server error")
        test_data = next(generate(StoreDataRequest))

        with pytest.raises(ServerError, match="Test server error"):
            app.test_client().post(
                "/v1/anomaly-detection/store",
                data=test_data.model_dump_json(),
                content_type="application/json",
            )

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.delete_alert_data")
    def test_delete_alert_data_endpoint_success(self, mock_delete_alert_data):
        """Test a successful run of delete_alert_data end point"""
        mock_delete_alert_data.return_value = DeleteAlertDataResponse(success=True)
        test_data = next(generate(DeleteAlertDataRequest))
        test_data.project_id = 1

        response = app.test_client().post(
            "/v1/anomaly-detection/delete-alert-data",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["success"] is True
        mock_delete_alert_data.assert_called_once_with(test_data)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.delete_alert_data")
    def test_delete_alert_data_endpoint_client_error(self, mock_delete_alert_data):
        """Test that delete_alert_data endpoint handles client errors correctly"""
        mock_delete_alert_data.side_effect = ClientError("Test error")
        test_data = next(generate(DeleteAlertDataRequest))

        response = app.test_client().post(
            "/v1/anomaly-detection/delete-alert-data",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["message"] == "Test error"
        assert response.json["success"] is False
        mock_delete_alert_data.assert_called_once_with(test_data)

    @mock.patch("seer.anomaly_detection.anomaly_detection.AnomalyDetection.delete_alert_data")
    def test_delete_alert_data_endpoint_server_error(self, mock_delete_alert_data):
        """Test that delete_alert_data endpoint handles server errors correctly"""
        mock_delete_alert_data.side_effect = ServerError("Test server error")
        test_data = next(generate(DeleteAlertDataRequest))

        with pytest.raises(ServerError, match="Test server error"):
            app.test_client().post(
                "/v1/anomaly-detection/delete-alert-data",
                data=test_data.model_dump_json(),
                content_type="application/json",
            )

    @mock.patch("seer.app.summarize_trace")
    def test_summarize_trace_endpoint_success(self, mock_summarize_trace):
        """Test a successful run of summarize_trace end point"""
        mock_summarize_trace.return_value = SummarizeTraceResponse(
            trace_id="test_trace_id",
            summary="Test summary",
            key_observations="Test key observations",
            performance_characteristics="Test performance characteristics",
            suggested_investigations=[
                SpanInsight(explanation="test suggested investigation 1", span_id="1", span_op="1")
            ],
        )
        test_data = next(generate(SummarizeTraceRequest))

        response = app.test_client().post(
            "/v1/automation/summarize/trace",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["trace_id"] == "test_trace_id"
        assert response.json["summary"] == "Test summary"
        assert response.json["key_observations"] == "Test key observations"
        assert response.json["performance_characteristics"] == "Test performance characteristics"
        assert response.json["suggested_investigations"] == [
            {
                "explanation": "test suggested investigation 1",
                "span_id": "1",
                "span_op": "1",
            }
        ]

        mock_summarize_trace.assert_called_once_with(test_data)

    @mock.patch("seer.app.summarize_trace")
    def test_summarize_trace_endpoint_error(self, mock_summarize_trace):
        """Test that summarize_trace endpoint handles exceptions correctly"""

        mock_summarize_trace.side_effect = APITimeoutError(
            request=httpx.Request(
                method="POST", url="http://localhost/v1/automation/summarize/trace"
            ),
        )
        test_data = next(generate(SummarizeTraceRequest))

        response = app.test_client().post(
            "/v1/automation/summarize/trace",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )
        assert response.status_code == 504
        mock_summarize_trace.side_effect = Exception("Test error")
        test_data = next(generate(SummarizeTraceRequest))

        response = app.test_client().post(
            "/v1/automation/summarize/trace",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )
        assert response.status_code == 500

    @mock.patch("seer.app.create_cache")
    def test_create_cache_endpoint_success(self, mock_create_cache):
        """Test a successful run of create_cache end point"""
        mock_create_cache.return_value = CreateCacheResponse(
            success=True, message="Cache created", cache_name="test-cache"
        )
        test_data = next(generate(CreateCacheRequest))

        response = app.test_client().post(
            "/v1/assisted-query/create-cache",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["success"] is True
        assert response.json["message"] == "Cache created"
        assert response.json["cache_name"] == "test-cache"

        mock_create_cache.assert_called_once_with(test_data)

    @mock.patch("seer.app.create_cache")
    def test_create_cache_endpoint_error(self, mock_create_cache):
        """Test that create_cache endpoint handles exceptions correctly"""
        mock_create_cache.side_effect = Exception("Test error")
        test_data = next(generate(CreateCacheRequest))

        response = app.test_client().post(
            "/v1/assisted-query/create-cache",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )
        assert response.status_code == 500

    @mock.patch("seer.app.create_cache")
    def test_create_cache_endpoint_client_error(self, mock_create_cache):
        """Test that create_cache endpoint handles client errors correctly"""

        mock_create_cache.side_effect = APITimeoutError(
            request=httpx.Request(
                method="POST", url="http://localhost/v1/assisted-query/create-cache"
            ),
        )
        test_data = next(generate(CreateCacheRequest))

        response = app.test_client().post(
            "/v1/assisted-query/create-cache",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )
        assert response.status_code == 504

    @mock.patch("seer.app.translate_query")
    def test_translate_query_endpoint_success(self, mock_translate_query):
        """Test a successful run of translate_query end point"""
        mock_translate_query.return_value = TranslateResponse(
            query="Test query",
            stats_period="Test stats period",
            group_by=["Test group by"],
            visualization=[
                Chart(
                    chart_type=1,
                    y_axes=["Test y-axis 1", "Test y-axis 2"],
                )
            ],
            sort="Test sort",
        )
        test_data = next(generate(TranslateRequest))

        response = app.test_client().post(
            "/v1/assisted-query/translate",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["query"] == "Test query"
        assert response.json["stats_period"] == "Test stats period"
        assert response.json["group_by"] == ["Test group by"]
        assert response.json["visualization"] == [
            {
                "chart_type": 1,
                "y_axes": ["Test y-axis 1", "Test y-axis 2"],
            }
        ]
        assert response.json["sort"] == "Test sort"

        mock_translate_query.assert_called_once_with(test_data)

    @mock.patch("seer.app.translate_query")
    def test_translate_query_endpoint_error(self, mock_translate_query):
        """Test that translate_query endpoint handles exceptions correctly"""
        mock_translate_query.side_effect = Exception("Test error")
        test_data = next(generate(TranslateRequest))

        response = app.test_client().post(
            "/v1/assisted-query/translate",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )
        assert response.status_code == 500
        mock_translate_query.assert_called_once_with(test_data)

        mock_translate_query.side_effect = APITimeoutError(
            request=httpx.Request(
                method="POST", url="http://localhost/v1/assisted-query/translate"
            ),
        )
        test_data = next(generate(TranslateRequest))

        response = app.test_client().post(
            "/v1/assisted-query/translate",
            data=test_data.model_dump_json(),
            content_type="application/json",
        )
        assert response.status_code == 504


@parametrize(count=1)
def test_prepared_statements_disabled(
    requests: tuple[
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
        ProcessRequest,
    ],
):
    with Session() as session:
        # This would cause postgresql to issue prepared statements.  Remove logic from bootup connect args to validate.
        for i, request in enumerate(requests):
            request.name += str(i)
            session.add(request)
            session.flush()
        assert session.execute(text("select count(*) from pg_prepared_statements")).scalar() == 0


smokeless_module = Module()


@smokeless_module.provider
def smokeless_config() -> AppConfig:
    test_config = provide_test_defaults()
    test_config.SMOKE_CHECK = False
    return test_config


def test_smoke_test(celery_app: Celery, celery_worker: Worker):
    celery_app.task(smoke_test)
    celery_worker.reload()
    app_config = resolve(AppConfig)
    with Session() as session:
        assert (
            session.query(DbSmokeTest)
            .filter(DbSmokeTest.request_id == app_config.smoke_test_id)
            .first()
            is None
        )
    response = app.test_client().get("/health/live")
    assert response.status_code == 200
    response = app.test_client().get("/health/ready")
    assert response.status_code == 503

    start_loading().join()

    for i in range(10):
        response = app.test_client().get("/health/ready")
        if response.status_code == 200:
            with Session() as session:
                assert (
                    session.query(DbSmokeTest)
                    .filter(DbSmokeTest.request_id == app_config.smoke_test_id)
                    .first()
                )
            break
        time.sleep(1)
    else:
        assert False, "Timed out, did not complete smoke check"


def test_async_loading():
    with smokeless_module:
        reset_loading_state()
        response = app.test_client().get("/health/live")
        assert response.status_code == 200
        response = app.test_client().get("/health/ready")
        assert response.status_code == 503

        with dummy_deferred(lambda: time.sleep(1)):
            start_loading()
            response = app.test_client().get("/health/live")
            assert response.status_code == 200
            response = app.test_client().get("/health/ready")
            assert response.status_code == 503

            time.sleep(2)
            response = app.test_client().get("/health/live")
            assert response.status_code == 200
            response = app.test_client().get("/health/ready")
            assert response.status_code == 200

        def failed_loader():
            time.sleep(1)
            raise Exception("Dummy loading failure!")

        reset_loading_state()

        with dummy_deferred(failed_loader):
            start_loading()
            response = app.test_client().get("/health/live")
            assert response.status_code == 200
            response = app.test_client().get("/health/ready")
            assert response.status_code == 503

            time.sleep(2)
            response = app.test_client().get("/health/live")
            assert response.status_code == 500
            response = app.test_client().get("/health/ready")
            assert response.status_code == 500


class TestGetAutofixState:
    @mock.patch("seer.app.get_autofix_state")
    def test_get_autofix_state_endpoint_with_group_id(self, mock_get_autofix_state):
        state = next(generate(AutofixContinuation))
        mock_get_autofix_state.return_value = LocalMemoryState(state)

        response = app.test_client().post(
            "/v1/automation/autofix/state",
            data=json.dumps({"group_id": 400, "check_repo_access": False}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.get_data(as_text=True))
        assert data["group_id"] == state.request.issue.id
        assert data["run_id"] == state.run_id
        assert data["state"] == state.model_dump(mode="json")

        mock_get_autofix_state.assert_called_once_with(group_id=400, run_id=None)

    @mock.patch("seer.app.get_autofix_state")
    def test_get_autofix_state_endpoint_with_run_id(self, mock_get_autofix_state):
        state = next(generate(AutofixContinuation))
        mock_get_autofix_state.return_value = LocalMemoryState(state)

        response = app.test_client().post(
            "/v1/automation/autofix/state",
            data=json.dumps({"run_id": 500, "check_repo_access": False}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.get_data(as_text=True))
        assert data["group_id"] == state.request.issue.id
        assert data["run_id"] == state.run_id
        assert data["state"] == state.model_dump(mode="json")

        mock_get_autofix_state.assert_called_once_with(group_id=None, run_id=500)

    @mock.patch("seer.app.get_autofix_state")
    def test_get_autofix_state_endpoint_no_state_found(self, mock_get_autofix_state):
        mock_get_autofix_state.return_value = None

        response = app.test_client().post(
            "/v1/automation/autofix/state",
            data=json.dumps({"group_id": 999, "check_repo_access": False}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.get_data(as_text=True))
        assert data == {"group_id": None, "run_id": None, "state": None}

    @mock.patch("seer.automation.autofix.runs.set_repo_branches_and_commits")
    @mock.patch("seer.app.update_repo_access_and_properties")
    @mock.patch("seer.app.get_autofix_state")
    def test_get_autofix_state_endpoint_with_check_repo_access(
        self,
        mock_get_autofix_state,
        mock_update_repo_access_and_properties,
        mock_set_repo_branches_and_commits,
    ):

        state_obj = create_initial_autofix_run(next(generate(AutofixRequest)))

        state = state_obj.get()
        request = cast(AutofixRequest, state.request)

        mock_get_autofix_state.return_value = state_obj

        response = app.test_client().post(
            "/v1/automation/autofix/state",
            data=json.dumps({"group_id": request.issue.id, "check_repo_access": True}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.get_data(as_text=True))
        assert data["group_id"] == state.request.issue.id
        assert data["run_id"] == state.run_id

        mock_get_autofix_state.assert_called_once_with(group_id=1, run_id=None)
        mock_update_repo_access_and_properties.assert_called_once()

    @mock.patch("seer.app.get_autofix_state_from_pr_id")
    def test_get_autofix_state_from_pr_endpoint(self, mock_get_autofix_state_from_pr_id):
        state = next(generate(AutofixContinuation))
        mock_get_autofix_state_from_pr_id.return_value = mock.Mock(get=lambda: state)

        response = app.test_client().post(
            "/v1/automation/autofix/state/pr",
            data=json.dumps({"provider": "github", "pr_id": 123}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.get_data(as_text=True))
        assert data["group_id"] == state.request.issue.id
        assert data["run_id"] == state.run_id
        assert data["state"] == state.model_dump(mode="json")

        mock_get_autofix_state_from_pr_id.assert_called_once_with("github", 123)

    @mock.patch("seer.app.get_autofix_state_from_pr_id")
    def test_get_autofix_state_from_pr_endpoint_no_state_found(
        self, mock_get_autofix_state_from_pr_id
    ):
        mock_get_autofix_state_from_pr_id.return_value = None

        response = app.test_client().post(
            "/v1/automation/autofix/state/pr",
            data=json.dumps({"provider": "github", "pr_id": 999}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.get_data(as_text=True))
        assert data == {"group_id": None, "run_id": None, "state": None}
