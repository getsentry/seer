import unittest
from unittest.mock import MagicMock, patch

from seer.automation.codegen.models import CodegenUnitTestsRequest
from seer.automation.codegen.tasks import (
    codegen_unittest,
    create_initial_unittest_run,
    create_subsequent_unittest_run,
)
from seer.automation.models import RepoDefinition
from seer.db import DbState


class TestCodegenUnittest(unittest.TestCase):
    @patch("seer.automation.codegen.tasks.create_initial_unittest_run")
    @patch("seer.automation.codegen.tasks.UnittestStep.get_signature")
    def test_codegen_unittest_default_params(self, mock_get_signature, mock_create_initial):
        # Setup
        mock_apply_async = MagicMock()
        mock_get_signature.return_value.apply_async = mock_apply_async
        
        mock_state = MagicMock(spec=DbState)
        mock_state.get.return_value.run_id = 1
        mock_create_initial.return_value = mock_state
        
        mock_request = MagicMock(spec=CodegenUnitTestsRequest)
        mock_request.pr_id = 123
        mock_request.repo = RepoDefinition(
            name="repo", owner="owner", provider="github", external_id="ext123"
        )
        
        mock_app_config = MagicMock()
        mock_app_config.CELERY_WORKER_QUEUE = "worker_queue"
        
        # Execute
        codegen_unittest(mock_request, mock_app_config)
        
        # Verify
        mock_create_initial.assert_called_once_with(mock_request)
        mock_get_signature.assert_called_once()
        unittest_request_arg = mock_get_signature.call_args[0][0]
        self.assertEqual(unittest_request_arg.run_id, 1)
        self.assertEqual(unittest_request_arg.pr_id, 123)
        self.assertEqual(unittest_request_arg.repo_definition, mock_request.repo)
        self.assertFalse(unittest_request_arg.is_codecov_request)
        mock_get_signature.assert_called_once_with(
            unittest_request_arg, queue=mock_app_config.CELERY_WORKER_QUEUE
        )
        mock_apply_async.assert_called_once()

    @patch("seer.automation.codegen.tasks.create_initial_unittest_run")
    @patch("seer.automation.codegen.tasks.UnittestStep.get_signature")
    def test_codegen_unittest_with_codecov_request(self, mock_get_signature, mock_create_initial):
        # Setup
        mock_apply_async = MagicMock()
        mock_get_signature.return_value.apply_async = mock_apply_async
        
        mock_state = MagicMock(spec=DbState)
        mock_state.get.return_value.run_id = 1
        mock_create_initial.return_value = mock_state
        
        mock_request = MagicMock(spec=CodegenUnitTestsRequest)
        mock_request.pr_id = 123
        mock_request.repo = RepoDefinition(
            name="repo", owner="owner", provider="github", external_id="ext123"
        )
        
        mock_app_config = MagicMock()
        mock_app_config.CELERY_WORKER_QUEUE = "worker_queue"
        
        # Execute
        codegen_unittest(mock_request, mock_app_config, is_codecov_request=True)
        
        # Verify
        mock_create_initial.assert_called_once_with(mock_request)
        mock_get_signature.assert_called_once()
        unittest_request_arg = mock_get_signature.call_args[0][0]
        self.assertEqual(unittest_request_arg.run_id, 1)
        self.assertEqual(unittest_request_arg.pr_id, 123)
        self.assertEqual(unittest_request_arg.repo_definition, mock_request.repo)
        self.assertTrue(unittest_request_arg.is_codecov_request)
        mock_get_signature.assert_called_once_with(
            unittest_request_arg, queue=mock_app_config.CELERY_WORKER_QUEUE
        )
        mock_apply_async.assert_called_once()
