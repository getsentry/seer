from seer.dependency_injection import resolve
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from seer.automation.codegen.models import CodegenContinuation, CodegenStatus
from seer.automation.codegen.pr_closed_step import (
    CommentAnalyzer,
    PrClosedStep,
    PrClosedStepRequest,
)
from seer.automation.codegen.state import CodegenContinuationState
from seer.automation.models import RepoDefinition
from seer.automation.state import DbStateRunTypes
from seer.configuration import AppConfig
from seer.db import DbReviewCommentEmbedding, Session

bot_id = "41898282"

@pytest.fixture(autouse=True)
def setup_github_app_id():
    app_config = resolve(AppConfig)
    app_config.GITHUB_CODECOV_PR_REVIEW_APP_ID = bot_id
    yield

@pytest.fixture
def repo_definition():
    """Create test repo definition"""
    return RepoDefinition(
        name="test-repo",
        owner="test-org",
        provider="github",
        external_id="12345",
        base_commit_sha="test_sha",
        provider_raw="github",
    )


@pytest.fixture
def mock_pr():
    """Create mock PR"""
    pr = MagicMock()
    pr.number = 123
    pr.base.repo.name = "test-repo"
    pr.base.repo.owner.login = "test-org"
    pr.get_review_comments.return_value = []
    return pr


@pytest.fixture
def mock_repo_client(mock_pr):
    """Create mock repo client"""
    client = MagicMock()
    client.repo = mock_pr.base.repo
    client.repo.get_pull.return_value = mock_pr
    return client


@pytest.fixture
def pr_closed_request(repo_definition):
    """Create PR closed request"""
    return PrClosedStepRequest(run_id=789, step_id=123, pr_id=123, repo_definition=repo_definition)


@pytest.fixture
def state(pr_closed_request):
    """Create test state"""
    now = datetime.now(timezone.utc)

    state = CodegenContinuationState.new(
        CodegenContinuation(
            request={
                "repo": pr_closed_request.repo_definition.model_dump(),
                "pr_id": pr_closed_request.pr_id,
            },
            status=CodegenStatus.PENDING,
            run_id=pr_closed_request.run_id,
            last_triggered_at=now,
            updated_at=now,
            completed_at=None,
            file_changes=[],
            signals=[],
            relevant_warning_results=[],
        ),
        group_id=pr_closed_request.pr_id,
        t=DbStateRunTypes.PR_CLOSED,
    )

    with state.update() as cur:
        cur.status = CodegenStatus.PENDING
        cur.run_id = pr_closed_request.run_id
        cur.mark_triggered()

    return state


class TestPrClosedStep(unittest.TestCase):
    def setUp(self):
        self.request_data = {
            "run_id": 789,
            "step_id": 123,
            "pr_id": 123,
            "repo_definition": RepoDefinition(
                name="test-repo",
                owner="test-org",
                provider="github",
                external_id="12345",
                base_commit_sha="test_sha",
                provider_raw="github",
            ).model_dump(),
        }
        self.request = PrClosedStepRequest(**self.request_data)

    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context")
    def test_invoke_no_comments(self, mock_instantiate_context, _):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_pr.get_review_comments.return_value = []

        step = PrClosedStep(request=self.request_data, type=DbStateRunTypes.PR_CLOSED)
        step.context = mock_context
        step.invoke()

        mock_context.get_repo_client.assert_called_once()
        mock_repo_client.repo.get_pull.assert_called_once_with(self.request.pr_id)
        mock_pr.get_review_comments.assert_called_once()
        mock_context.event_manager.mark_completed.assert_called_once()

    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context")
    @patch("seer.automation.agent.embeddings.GoogleProviderEmbeddings.model")
    def test_invoke_with_bot_comment(self, mock_model, mock_instantiate_context, _):
        mock_model.return_value.encode.return_value = [[0.1] * 768]

        mock_pr = MagicMock()
        mock_pr.number = 123
        mock_pr.base.repo.name = "test-repo"
        mock_pr.base.repo.owner.login = "test-org"

        mock_repo_client = MagicMock()
        mock_repo_client.repo = mock_pr.base.repo
        mock_repo_client.repo.get_pull.return_value = mock_pr

        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client

        now = datetime.now(timezone.utc)
        comment = MagicMock(spec=[])
        comment.id = 456
        comment.user = MagicMock(spec=[])
        comment.user.id = bot_id
        comment.body = "wrap this in try-catch"

        reaction = MagicMock(spec=[])
        reaction.content = "+1"
        comment.get_reactions = MagicMock(return_value=[reaction])

        comment.html_url = "https://github.com/org/repo/pull/123#discussion_r456"
        comment.path = "src/file.py"
        comment.position = 42
        comment.created_at = now
        comment.updated_at = now

        comment.raw_data = {
            "id": 456,
            "url": "https://github.com/org/repo/pull/123#discussion_r456",
            "html_url": "https://github.com/org/repo/pull/123#discussion_r456",
            "path": "src/file.py",
            "position": 42,
            "user": {"id": bot_id},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "body": "wrap this in try-catch",
            "reactions": [{"content": "+1"}],
        }
        mock_pr.get_review_comments.return_value = [comment]

        step = PrClosedStep(request=self.request_data, type=DbStateRunTypes.PR_CLOSED)
        step.context = mock_context
        step.invoke()

        with Session() as session:
            record = session.query(DbReviewCommentEmbedding).first()
            self.assertIsNotNone(record)
            self.assertEqual(record.provider, "github")
            self.assertEqual(record.owner, "test-org")
            self.assertEqual(record.repo, "test-repo")
            self.assertEqual(record.pr_id, 123)
            self.assertEqual(record.body, "wrap this in try-catch")
            self.assertTrue(record.is_good_pattern)

        step.context.event_manager.mark_completed.assert_called_once()

    def test_comment_analyzer(self):
        analyzer = CommentAnalyzer()

        comment = MagicMock()
        comment.get_reactions.return_value = [
            MagicMock(content="+1"),
            MagicMock(content="+1"),
            MagicMock(content="-1"),
        ]

        is_good, is_bad = analyzer.analyze_reactions(comment)
        self.assertTrue(is_good)
        self.assertFalse(is_bad)
