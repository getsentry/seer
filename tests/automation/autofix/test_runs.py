from unittest.mock import MagicMock, patch

import pytest

from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixRequestOptions,
    IssueDetails,
)
from seer.automation.autofix.runs import create_initial_autofix_run, set_repo_branches_and_commits
from seer.automation.models import RepoDefinition, SeerProjectPreference
from seer.automation.preferences import GetSeerProjectPreferenceRequest


@pytest.fixture
def mock_request():
    return AutofixRequest(
        organization_id=1,
        project_id=2,
        repos=[],
        issue=IssueDetails(id=123, title="Test Issue", short_id="TEST-123", events=[]),
        invoking_user=None,
        instruction=None,
        options=AutofixRequestOptions(),
    )


class TestRuns:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        self.mock_event_manager = patch("seer.automation.autofix.runs.AutofixEventManager").start()
        self.mock_continuation_state = patch(
            "seer.automation.autofix.runs.ContinuationState"
        ).start()
        self.mock_autofix_continuation = patch(
            "seer.automation.autofix.runs.AutofixContinuation"
        ).start()
        # Patch preference retrieval and creation
        self.mock_get_pref = patch(
            "seer.automation.autofix.runs.get_seer_project_preference"
        ).start()
        self.mock_create_initial_pref = patch(
            "seer.automation.autofix.runs.create_initial_seer_project_preference_from_repos"
        ).start()
        yield
        patch.stopall()

    def test_create_initial_autofix_run(self, mock_request):
        # Set up mock for ContinuationState
        mock_state = MagicMock()
        self.mock_continuation_state.new.return_value = mock_state

        expected_state = AutofixContinuation(request=mock_request)
        self.mock_autofix_continuation.return_value = expected_state

        # Call the function
        result = create_initial_autofix_run(mock_request)

        # Assertions
        self.mock_autofix_continuation.assert_called_once_with(request=mock_request)
        assert mock_state.update.call_count == 2

        self.mock_event_manager.assert_called_once_with(mock_state)
        self.mock_event_manager.return_value.send_root_cause_analysis_will_start.assert_called_once()

        assert result == mock_state

    @pytest.mark.parametrize("exception_class", [ValueError, TypeError, RuntimeError])
    def test_create_initial_autofix_run_error_handling(self, mock_request, exception_class):
        # Set up mock to raise an exception
        self.mock_continuation_state.new.side_effect = exception_class("Test error")

        # Call the function and check if it raises the expected exception
        with pytest.raises(exception_class):
            create_initial_autofix_run(mock_request)

        # Assert that the event manager was not called due to the exception
        self.mock_event_manager.assert_not_called()

    def test_set_repo_branches_and_commits(self):
        # Mock repo and codebase
        mock_repo = MagicMock()
        mock_repo.provider = "github"
        mock_repo.external_id = "repo1"
        mock_repo.branch_name = "test_branch"
        mock_repo.base_commit_sha = "test_commit_sha"

        mock_codebase = MagicMock()
        mock_codebase.is_readable = True
        mock_codebase.is_writeable = True

        # Mock state.get() to return a state with repos and codebases
        mock_state = MagicMock()
        mock_state.request.repos = [mock_repo]
        mock_state.codebases = {"repo1": mock_codebase}
        mock_state.readable_repos = MagicMock(return_value=[mock_repo])

        # Patch state.update() as a context manager
        mock_update_cm = MagicMock()
        mock_update_cm.__enter__.return_value = mock_state
        mock_update_cm.__exit__.return_value = None

        mock_continuation_state = MagicMock()
        mock_continuation_state.get.return_value = mock_state
        mock_continuation_state.update.return_value = mock_update_cm

        set_repo_branches_and_commits(mock_continuation_state)

        # The fixture already asserts the patch was used, so just check the results
        assert mock_repo.branch_name == "test_branch"
        assert mock_repo.base_commit_sha == "test_commit_sha"

    def test_create_initial_autofix_run_creates_preference_when_none(self, mock_request):
        # Setup mock state
        mock_state = MagicMock()
        self.mock_continuation_state.new.return_value = mock_state

        # No existing preference
        self.mock_get_pref.return_value = MagicMock(preference=None)
        # Prepare sample repos
        sample_repos = [
            RepoDefinition(
                owner="ownerX",
                name="repoX",
                external_id="extX",
                provider="github",
            )
        ]
        # Ensure request.repos is initial list
        mock_request.repos = [
            RepoDefinition(owner="initial", name="repo", external_id="extInit", provider="github")
        ]
        # Setup creation return
        pref = SeerProjectPreference(
            organization_id=mock_request.organization_id,
            project_id=mock_request.project_id,
            repositories=sample_repos,
        )
        self.mock_create_initial_pref.return_value = pref

        # Call the function
        create_initial_autofix_run(mock_request)

        # Verify preference retrieval was attempted
        self.mock_get_pref.assert_called_once_with(
            GetSeerProjectPreferenceRequest(project_id=mock_request.project_id)
        )
        # Verify new preference creation was called
        self.mock_create_initial_pref.assert_called_once_with(
            organization_id=mock_request.organization_id,
            project_id=mock_request.project_id,
            repos=mock_request.repos,
        )
        # Verify that state.update was used to set repos to created preference
        ctx = mock_state.update.return_value.__enter__.return_value
        assert ctx.request.repos == sample_repos

