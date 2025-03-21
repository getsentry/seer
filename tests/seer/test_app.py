import unittest
from unittest.mock import MagicMock, patch

from seer.app import codecov_request_endpoint
from seer.automation.codegen.models import CodecovTaskRequest, CodegenPrReviewRequest, CodegenUnitTestsRequest
from seer.automation.models import RepoDefinition


class TestCodecovRequestEndpoint(unittest.TestCase):
    @patch("seer.app.codegen_pr_review_endpoint")
    def test_pr_review_request_type(self, mock_pr_review):
        # Setup
        mock_pr_review.return_value = {"status": "success"}
        data = CodegenPrReviewRequest(
            pr_id=123, 
            repo=RepoDefinition(name="repo", owner="owner", provider="github", external_id="ext123")
        )
        request = CodecovTaskRequest(request_type="pr-review", data=data, external_owner_id="owner123")
        
        # Execute
        result = codecov_request_endpoint(request)
        
        # Verify
        mock_pr_review.assert_called_once_with(data)
        self.assertEqual(result, {"status": "success"})
    
    @patch("seer.app.codegen_unittest")
    def test_unit_tests_request_type(self, mock_unittest):
        # Setup
        mock_unittest.return_value = {"status": "success"}
        data = CodegenUnitTestsRequest(
            pr_id=123, 
            repo=RepoDefinition(name="repo", owner="owner", provider="github", external_id="ext123")
        )
        request = CodecovTaskRequest(request_type="unit-tests", data=data, external_owner_id="owner123")
        
        # Execute
        result = codecov_request_endpoint(request)
        
        # Verify
        mock_unittest.assert_called_once_with(data, is_codecov_request=True)
        self.assertEqual(result, {"status": "success"})
