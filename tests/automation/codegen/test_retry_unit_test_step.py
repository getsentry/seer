import unittest
from unittest.mock import MagicMock, patch

from seer.automation.codegen.retry_unit_test_github_pr_creator import RetryUnitTestGithubPrUpdater
from seer.automation.codegen.retry_unittest_step import RetryUnittestStep
from seer.automation.models import RepoDefinition
from seer.automation.state import DbStateRunTypes


class DummyPreviousContext:
    pass


class TestRetryUnittestStep(unittest.TestCase):
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    def setUp(self, _):
        self.mock_pr = MagicMock()
        self.mock_pr.html_url = "http://example.com/pr/123"
        self.mock_pr.url = "http://api.github.com/pr/123"
        self.repo_client = MagicMock()
        self.repo_client.repo.get_pull.return_value = self.mock_pr
        self.mock_previous_context = DummyPreviousContext()
        self.mock_previous_context.original_pr_url = "http://original.com/pr/111"
        self.mock_previous_context.iterations = 1
        self.context = MagicMock()
        self.context.get_repo_client.return_value = self.repo_client
        self.context.get_previous_run_context.return_value = self.mock_previous_context
        self.context.event_manager = MagicMock()
        self.request_data = {
            "run_id": 1,
            "pr_id": 123,
            "repo_definition": RepoDefinition(
                name="repo1", owner="owner1", provider="github", external_id="id"
            ),
            "codecov_status": {"conclusion": "success"},
        }

    def _build_step(self, extra_request=None):
        req = self.request_data.copy()
        if extra_request:
            req.update(extra_request)
        step = RetryUnittestStep(request=req, type=DbStateRunTypes.UNIT_TESTS_RETRY)
        step.context = self.context
        return step

    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_invoke_success_status(self, _):
        step = self._build_step()
        step.invoke()
        self.repo_client.post_unit_test_reference_to_original_pr_codecov_app.assert_called_once_with(
            self.mock_previous_context.original_pr_url, self.mock_pr.html_url
        )
        self.context.event_manager.mark_running.assert_called_once()
        self.context.event_manager.mark_completed.assert_called_once()

    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_invoke_failed_status_with_no_generated_tests(self, _):
        extra = {"codecov_status": {"conclusion": "failure"}}
        step = self._build_step(extra)
        with patch.object(step, "_generate_unit_tests", return_value=None) as mock_generate:
            step.invoke()
            mock_generate.assert_called_once_with(
                self.repo_client, self.mock_pr, self.mock_previous_context
            )
            self.repo_client.post_unit_test_reference_to_original_pr_codecov_app.assert_called_once_with(
                self.mock_previous_context.original_pr_url, self.mock_pr.html_url
            )

    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_max_iterations_reached(self, _):
        self.mock_previous_context.iterations = RetryUnittestStep.MAX_ITERATIONS
        extra = {"codecov_status": {"conclusion": "failure"}}
        step = self._build_step(extra)
        with patch.object(step, "_generate_unit_tests") as mock_generate:
            step.invoke()
            mock_generate.assert_not_called()
            self.repo_client.post_unit_test_reference_to_original_pr_codecov_app.assert_not_called()

    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_get_previous_run_context_failure(self, _):
        self.context.get_previous_run_context.return_value = None
        extra = {"codecov_status": {"conclusion": "failure"}}
        step = self._build_step(extra)
        with self.assertRaises(RuntimeError):
            step.invoke()
        self.context.event_manager.mark_completed.assert_called_once()

    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_generate_unit_tests_exception(self, _):
        self.repo_client.get_pr_diff_content.side_effect = Exception("diff error")
        step = self._build_step()
        result = step._generate_unit_tests(
            self.repo_client, self.mock_pr, self.mock_previous_context
        )
        self.assertIsNone(result)


class TestRetryUnitTestGithubPrUpdater(unittest.TestCase):
    @patch("seer.automation.codegen.retry_unit_test_github_pr_creator.Session")
    def test_update_github_pull_request_success(self, mock_session_cls):
        fc1 = MagicMock()
        fc1.commit_message = "msg1"
        fc2 = MagicMock()
        fc2.commit_message = "msg2"
        file_changes = [fc1, fc2]
        mock_pr = MagicMock()
        mock_pr.head.sha = "head_sha"
        mock_pr.number = 123
        mock_pr.html_url = "http://example.com/pr/123"
        repo_client = MagicMock()
        repo_client.push_new_commit_to_pr.return_value = "new_commit"
        mock_previous_context = DummyPreviousContext()
        mock_previous_context.iterations = 1
        updater = RetryUnitTestGithubPrUpdater(
            file_changes_payload=file_changes,
            pr=mock_pr,
            repo_client=repo_client,
            previous_context=mock_previous_context,
        )
        merged_context = type("MergedContext", (), {})()
        merged_context.iterations = mock_previous_context.iterations
        session_instance = MagicMock()
        session_instance.merge.return_value = merged_context
        mock_session = MagicMock()
        mock_session.__enter__.return_value = session_instance
        mock_session_cls.return_value = mock_session
        updater.update_github_pull_request()
        repo_client.push_new_commit_to_pr.assert_called_once()
        session_instance.merge.assert_called_once_with(mock_previous_context)
        self.assertEqual(merged_context.iterations, mock_previous_context.iterations + 1)
        session_instance.commit.assert_called_once()

    def test_update_github_pull_request_failure(self):
        fc = MagicMock()
        fc.commit_message = "msg"
        file_changes = [fc]
        mock_pr = MagicMock()
        mock_pr.head.sha = "head_sha"
        mock_pr.number = 123
        mock_pr.html_url = "http://example.com/pr/123"
        repo_client = MagicMock()
        repo_client.push_new_commit_to_pr.return_value = None
        mock_previous_context = DummyPreviousContext()
        mock_previous_context.iterations = 1
        updater = RetryUnitTestGithubPrUpdater(
            file_changes_payload=file_changes,
            pr=mock_pr,
            repo_client=repo_client,
            previous_context=mock_previous_context,
        )
        with patch(
            "seer.automation.codegen.retry_unit_test_github_pr_creator.logger"
        ) as mock_logger:
            updater.update_github_pull_request()
            mock_logger.warning.assert_called_once_with("Failed to push new commit to PR")
