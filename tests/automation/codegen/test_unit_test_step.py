import unittest
from unittest.mock import MagicMock, patch

from github.PullRequest import PullRequest

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.models import CodeUnitTestRequest
from seer.automation.codegen.unit_test_github_pr_creator import GeneratedTestsPullRequestCreator
from seer.automation.codegen.unittest_step import UnittestStep, UnittestStepRequest
from seer.automation.models import FileChange, RepoDefinition


class TestUnittestStep(unittest.TestCase):
    @patch(
        "seer.automation.codegen.unit_test_github_pr_creator.GeneratedTestsPullRequestCreator.create_github_pull_request"
    )
    @patch("seer.automation.codegen.unit_test_coding_component.UnitTestCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_invoke_happy_path(
        self,
        _,
        mock_pipeline_step,
        mock_invoke_unit_test_component,
        mock_create_pr,
    ):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_diff_content = "diff content"
        mock_latest_commit_sha = "latest_commit_sha"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content
        mock_repo_client.get_pr_head_sha.return_value = mock_latest_commit_sha

        request_data = {
            "run_id": 1,
            "pr_id": 123,
            "repo_definition": RepoDefinition(
                name="repo1", owner="owner1", provider="github", external_id="123123"
            ),
        }
        request = UnittestStepRequest(**request_data)
        step = UnittestStep(request=request)
        step.context = mock_context

        step.invoke()

        mock_context.get_repo_client.assert_called_once()
        mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
        mock_repo_client.get_pr_diff_content.assert_called_once_with(mock_pr.url)
        mock_repo_client.get_pr_head_sha.assert_called_once_with(mock_pr.url)

        actual_request = mock_invoke_unit_test_component.call_args[0][0]

        assert isinstance(actual_request, CodeUnitTestRequest)
        assert actual_request.diff == mock_diff_content
        assert actual_request.codecov_client_params == {
            "repo_name": request.repo_definition.name,
            "pullid": request.pr_id,
            "owner_username": request.repo_definition.owner,
            "head_sha": mock_latest_commit_sha,
        }

    @patch("seer.automation.codegen.unit_test_github_pr_creator.Session")
    @patch("time.time", return_value=1234567890)
    def test_create_github_pull_request_success(self, _, mock_session):
        file_changes_payload = [MagicMock(spec=FileChange), MagicMock(spec=FileChange)]
        pr = MagicMock(spec=PullRequest)
        pr.head.ref = "head_sha"
        pr.head.sha = "head_sha"
        pr.number = 123
        pr.base.ref = "main"
        pr.html_url = "http://example.com/pr/123"
        repo_client = MagicMock(spec=RepoClient)
        repo_client.repo = MagicMock()
        repo_client.repo.owner = MagicMock()
        repo_client.repo.owner.login = "owner_login"
        repo_client.repo.name = "repo1"

        new_pr = MagicMock(spec=PullRequest)
        new_pr.number = 456
        new_pr.html_url = "http://example.com/pr/456"
        repo_client.create_branch_from_changes.return_value = "branch_ref"
        repo_client.create_pr_from_branch.return_value = new_pr

        creator = GeneratedTestsPullRequestCreator(
            file_changes_payload=file_changes_payload,
            pr=pr,
            repo_client=repo_client,
            unit_test_run_id=2,
        )

        branch_name = "ai_tests_for_pr123_1234567890"
        pr_title = "Add Tests for PR#123"
        file_changes_payload[0].commit_message = "commit message 1"
        file_changes_payload[1].commit_message = "commit message 2"

        session_instance = MagicMock()
        mock_session.return_value.__enter__.return_value = session_instance

        creator.create_github_pull_request()

        repo_client.create_branch_from_changes.assert_called_once_with(
            pr_title=pr_title,
            file_changes=file_changes_payload,
            branch_name=branch_name,
            from_base_sha=True,
        )
        repo_client.create_pr_from_branch.assert_called_once_with(
            branch="branch_ref",
            title=pr_title,
            description="This PR adds tests for #123\n\n### Commits:\n- commit message 1\n- commit message 2",
            provided_base="head_sha",
        )
        self.assertEqual(repo_client.base_commit_sha, pr.head.sha)

        session_instance.add.assert_called_once()
        mapping = session_instance.add.call_args[0][0]
        self.assertEqual(mapping.provider, "github")
        self.assertEqual(mapping.owner, "owner_login")
        self.assertEqual(mapping.repo, "repo1")
        self.assertEqual(mapping.pr_id, new_pr.number)
        self.assertEqual(mapping.iterations, 0)
        self.assertEqual(mapping.run_id, 2)
        self.assertEqual(mapping.original_pr_url, pr.html_url)
        session_instance.commit.assert_called_once()

    @patch("seer.automation.codegen.unit_test_github_pr_creator.Session")
    @patch("seer.automation.codegen.unit_test_github_pr_creator.logger")
    def test_create_github_pull_request_failure(self, mock_logger, mock_session):
        file_changes_payload = [MagicMock(spec=FileChange), MagicMock(spec=FileChange)]
        pr = MagicMock(spec=PullRequest)
        pr.head.sha = "head_sha"
        pr.number = 123
        pr.base.ref = "main"
        pr.html_url = "http://example.com/pr/123"
        repo_client = MagicMock(spec=RepoClient)

        creator = GeneratedTestsPullRequestCreator(
            file_changes_payload=file_changes_payload,
            pr=pr,
            repo_client=repo_client,
            unit_test_run_id=1,
        )

        file_changes_payload[0].commit_message = "commit message 1"
        file_changes_payload[1].commit_message = "commit message 2"

        repo_client.create_branch_from_changes.return_value = None

        creator.create_github_pull_request()

        mock_logger.warning.assert_called_once_with("Failed to create branch from changes")
        repo_client.create_pr_from_branch.assert_not_called()
        mock_session.assert_not_called()

    @patch("seer.automation.codegen.unit_test_coding_component.UnitTestCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_post_unit_test_not_generated_on_exception(
        self, mock_instantiate_context, _, mock_invoke_unit_test_component
    ):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_pr.url = "http://example.com/pr/123"
        mock_pr.html_url = "http://example.com/pr/123"
        mock_diff_content = "diff content"
        mock_latest_commit_sha = "latest_commit_sha"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content
        mock_repo_client.get_pr_head_sha.return_value = mock_latest_commit_sha

        mock_invoke_unit_test_component.side_effect = ValueError()

        request_data = {
            "run_id": 1,
            "pr_id": 123,
            "repo_definition": RepoDefinition(
                name="repo1", owner="owner1", provider="github", external_id="123123"
            ),
        }
        request = UnittestStepRequest(**request_data)
        step = UnittestStep(request=request)
        step.context = mock_context

        step.invoke()

        mock_context.get_repo_client.assert_called_once()
        mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
        mock_repo_client.get_pr_diff_content.assert_called_once_with(mock_pr.url)
        mock_repo_client.get_pr_head_sha.assert_called_once_with(mock_pr.url)

        mock_repo_client.post_unit_test_not_generated_message_to_original_pr.assert_called_once_with(
            mock_pr.html_url
        )

    @patch("seer.automation.codegen.unit_test_coding_component.UnitTestCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_post_unit_test_not_generated_when_no_output(
        self, mock_instantiate_context, _, mock_invoke_unit_test_component
    ):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_pr.url = "http://example.com/pr/123"
        mock_pr.html_url = "http://example.com/pr/123"
        mock_diff_content = "diff content"
        mock_latest_commit_sha = "latest_commit_sha"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content
        mock_repo_client.get_pr_head_sha.return_value = mock_latest_commit_sha

        mock_invoke_unit_test_component.return_value = None

        request_data = {
            "run_id": 1,
            "pr_id": 123,
            "repo_definition": RepoDefinition(
                name="repo1", owner="owner1", provider="github", external_id="123123"
            ),
        }
        request = UnittestStepRequest(**request_data)
        step = UnittestStep(request=request)
        step.context = mock_context

        step.invoke()

        mock_context.get_repo_client.assert_called_once()
        mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
        mock_repo_client.get_pr_diff_content.assert_called_once_with(mock_pr.url)
        mock_repo_client.get_pr_head_sha.assert_called_once_with(mock_pr.url)

        mock_repo_client.post_unit_test_not_generated_message_to_original_pr.assert_called_once_with(
            mock_pr.html_url
        )
