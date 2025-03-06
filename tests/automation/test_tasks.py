from unittest.mock import MagicMock, call, patch

from seer.automation.codegen.models import CodegenBaseRequest, CodegenStatus
from seer.automation.codegen.state import CodegenContinuation
from seer.automation.state import DbState, DbStateRunTypes
from seer.automation.tasks import check_github_reactions, check_pr_reactions
from seer.db import DbRunState


def test_check_github_reactions_spawns_tasks():
    mock_states = [
        DbRunState(id=1, type=DbStateRunTypes.PR_REVIEW, group_id=123),
        DbRunState(id=2, type=DbStateRunTypes.PR_REVIEW, group_id=456),
    ]

    with (
        patch("seer.db.Session") as mock_session,
        patch("seer.automation.tasks.check_pr_reactions.delay") as mock_delay,
    ):
        mock_session.return_value.__enter__.return_value.query.return_value.filter.return_value.all.return_value = (
            mock_states
        )

        # Call the task
        check_github_reactions()

        # Assert individual tasks were spawned
        assert mock_delay.call_count == 2
        mock_delay.assert_has_calls([call(1), call(2)])


def test_check_pr_reactions():
    mock_db_state = DbRunState(
        id=1,
        type=DbStateRunTypes.PR_REVIEW,
        group_id=123,
        value={
            "status": CodegenStatus.COMPLETED,
            "request": CodegenBaseRequest(
                pr_id=123,
                # Add other required fields
            ).model_dump(),
            "signals": [],
        },
    )

    with (
        patch("seer.db.Session") as mock_session,
        patch("seer.automation.state.DbState.get") as mock_get,
    ):
        mock_session.return_value.__enter__.return_value.query.return_value.get.return_value = (
            mock_db_state
        )
        mock_get.return_value = CodegenContinuation(
            status=CodegenStatus.COMPLETED,
            # Add other required fields
        )

        # Call the task
        check_pr_reactions(1)

        # Add assertions for your GitHub API calls and DB operations
