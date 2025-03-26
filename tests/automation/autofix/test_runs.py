from unittest.mock import MagicMock, patch

import pytest

from seer.automation.autofix.models import AutofixContinuation, CodebaseState
from seer.automation.autofix.runs import update_repo_access
from seer.automation.autofix.state import ContinuationState
from seer.automation.models import RepoDefinition


class TestUpdateRepoAccess:
    @patch("seer.automation.autofix.runs.RepoClient")
    def test_update_repo_access_with_missing_codebase(self, mock_repo_client):
        # Mock RepoClient methods
        mock_repo_client.check_repo_read_access.return_value = True
        mock_repo_client.check_repo_write_access.return_value = True

        # Create a test repo definition with an external ID
        test_repo = RepoDefinition(
            provider="github",
            owner="test-owner",
            name="test-repo",
            external_id="123456789",  # This ID is not in the codebases dict yet
            branch_name=None,
            instructions=None,
            base_commit_sha=None,
            provider_raw="integrations:github"
        )

        # Create a mock continuation with the test repo but without a codebase entry
        mock_continuation = AutofixContinuation(
            request=MagicMock(repos=[test_repo]),
            codebases={},  # Empty codebases dict - the repo's external_id is not here
        )

        # Create a mock state that will return our mock continuation
        mock_state = MagicMock(spec=ContinuationState)
        mock_state.update.return_value.__enter__.return_value = mock_continuation
        
        # Call the function
        update_repo_access(mock_state)
        
        # Verify that the repo was added to codebases
        assert test_repo.external_id in mock_continuation.codebases
        assert isinstance(mock_continuation.codebases[test_repo.external_id], CodebaseState)
        
        # Verify that is_readable and is_writeable were set correctly
        assert mock_continuation.codebases[test_repo.external_id].is_readable
        assert mock_continuation.codebases[test_repo.external_id].is_writeable
