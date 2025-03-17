import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from github.GithubException import UnknownObjectException
from johen import generate

from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    ChangesStep,
    CodebaseChange,
    CodebaseState,
    StepType,
)
from seer.automation.autofix.state import ContinuationState
from seer.automation.models import (
    EventDetails,
    ExceptionDetails,
    FileChange,
    IssueDetails,
    RepoDefinition,
    SentryEventData,
    Stacktrace,
    StacktraceFrame,
    ThreadDetails,
)
from seer.automation.state import DbStateRunTypes
from seer.automation.summarize.issue import IssueSummary, IssueSummaryWithScores
from seer.automation.summarize.models import SummarizeIssueScores
from seer.db import DbIssueSummary, DbPrIdToAutofixRunIdMapping, DbRunMemory, Session


class TestAutofixContext(unittest.TestCase):
    def setUp(self):
        self.mock_repo_client = MagicMock()
        error_event = next(generate(SentryEventData))
        self.state = ContinuationState.new(
            AutofixContinuation(
                request=AutofixRequest(
                    organization_id=1,
                    project_id=1,
                    repos=[],
                    issue=IssueDetails(id=0, title="", events=[error_event]),
                )
            ),
            t=DbStateRunTypes.AUTOFIX,
        )
        self.autofix_context = AutofixContext(
            self.state,
            MagicMock(),
        )

    def test_process_event_paths(self):
        mock_event = EventDetails(
            title="title",
            exceptions=[
                ExceptionDetails(stacktrace=None, type="type"),
                ExceptionDetails(
                    type="type",
                    stacktrace=Stacktrace(
                        frames=[
                            StacktraceFrame(
                                filename="test_file.py",
                                in_app=True,
                                repo_name=None,
                                context=[],
                                abs_path="path",
                                line_no=1,
                                col_no=1,
                            )
                        ]
                    ),
                ),
            ],
            threads=[
                ThreadDetails(
                    id=1,
                    stacktrace=Stacktrace(
                        frames=[
                            StacktraceFrame(
                                filename="another_file.py",
                                in_app=True,
                                repo_name=None,
                                context=[],
                                abs_path="path",
                                line_no=1,
                                col_no=1,
                            )
                        ]
                    ),
                ),
                ThreadDetails(id=1, stacktrace=None),
            ],
        )

        self.autofix_context._process_stacktrace_paths = MagicMock()
        self.autofix_context.process_event_paths(mock_event)

        self.assertEqual(self.autofix_context._process_stacktrace_paths.call_count, 2)
        self.autofix_context._process_stacktrace_paths.assert_any_call(
            mock_event.exceptions[1].stacktrace
        )
        self.autofix_context._process_stacktrace_paths.assert_any_call(
            mock_event.threads[0].stacktrace
        )

    def test_get_issue_summary(self):
        with Session() as session:
            original = IssueSummaryWithScores(
                title="title",
                whats_wrong="whats wrong",
                session_related_issues="session_related_issues",
                possible_cause="possible cause",
                scores=SummarizeIssueScores(
                    possible_cause_confidence=0.5,
                    possible_cause_novelty=0.8,
                    is_fixable=True,
                    fixability_score=0.9,
                    fixability_score_version=1,
                ),
            )
            session.add(original.to_db_state(0))
            session.commit()

        instance = self.autofix_context

        result = instance.get_issue_summary()

        self.assertIsNotNone(result)
        self.assertIsInstance(result, IssueSummaryWithScores)
        if result:
            self.assertEqual(result.title, "title")
            self.assertEqual(result.whats_wrong, "whats wrong")
            self.assertEqual(result.session_related_issues, "session_related_issues")
            self.assertEqual(result.possible_cause, "possible cause")
            self.assertEqual(result.scores.possible_cause_confidence, 0.5)
            self.assertEqual(result.scores.possible_cause_novelty, 0.8)
            self.assertEqual(result.scores.is_fixable, True)
            self.assertEqual(result.scores.fixability_score, 0.9)
            self.assertEqual(result.scores.fixability_score_version, 1)

        with Session() as session:
            invalid_summary_data = {"bad data": "uh oh"}
            db_issue_summary = DbIssueSummary(group_id=0, summary=invalid_summary_data)
            session.merge(db_issue_summary)
            session.commit()

        result = instance.get_issue_summary()
        self.assertIsNone(result)

    def test_store_memory(self):
        instance = self.autofix_context
        key = "test_key"
        memory = [Message(role="user", content="Test message")]

        instance.store_memory(key, memory)

        with Session() as session:
            db_memory = (
                session.query(DbRunMemory).where(DbRunMemory.run_id == self.state.id).first()
            )
            self.assertIsNotNone(db_memory)
            if db_memory:
                self.assertEqual(
                    db_memory.value,
                    {
                        "run_id": instance.run_id,
                        "memory": {
                            "test_key": [
                                {
                                    "role": "user",
                                    "content": "Test message",
                                    "tool_call_id": None,
                                    "tool_calls": None,
                                    "tool_call_function": None,
                                    "thinking_content": None,
                                    "thinking_signature": None,
                                }
                            ]
                        },
                    },
                )

    def test_get_memory_existing(self):
        instance = self.autofix_context
        key = "test_key"
        memory = [Message(role="user", content="Test message")]

        instance.store_memory(key, memory)

        result = instance.get_memory(key)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, "user")
        self.assertEqual(result[0].content, "Test message")

    def test_get_memory_non_existing(self):
        instance = self.autofix_context
        result = instance.get_memory("non_existing_key")

        self.assertEqual(result, [])

    def test_store_and_get_memory(self):
        instance = self.autofix_context
        key = "test_key"
        memory = [Message(role="user", content="Test message")]

        # Store memory
        instance.store_memory(key, memory)

        # Get memory
        result = instance.get_memory(key)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, "user")
        self.assertEqual(result[0].content, "Test message")

    def test_store_memory_multiple_times(self):
        instance = self.autofix_context
        key = "test_key"
        memory1 = [Message(role="user", content="Test message 1")]
        memory2 = [Message(role="assistant", content="Test message 2")]

        # Store memory twice
        instance.store_memory(key, memory1)
        instance.store_memory(key, memory2)

        # Get memory
        result = instance.get_memory(key)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, "assistant")
        self.assertEqual(result[0].content, "Test message 2")

    def test_store_multiple_keys(self):
        instance = self.autofix_context
        key1 = "test_key_1"
        key2 = "test_key_2"
        memory1 = [Message(role="user", content="Test message 1")]
        memory2 = [Message(role="assistant", content="Test message 2")]

        # Store memory for two different keys
        instance.store_memory(key1, memory1)
        instance.store_memory(key2, memory2)

        # Get memory for both keys
        result1 = instance.get_memory(key1)
        result2 = instance.get_memory(key2)

        self.assertEqual(len(result1), 1)
        self.assertEqual(result1[0].role, "user")
        self.assertEqual(result1[0].content, "Test message 1")

        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0].role, "assistant")
        self.assertEqual(result2[0].content, "Test message 2")

    @patch("seer.automation.autofix.autofix_context.RepoClient")
    def test_process_stacktrace_paths_unknown_object_exception(self, mock_RepoClient):
        mock_RepoClient.from_repo_definition.side_effect = UnknownObjectException(status=404)
        mock_RepoClient.supported_providers = ["github"]
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="test_file.py",
                    in_app=True,
                    repo_name=None,
                    context=[],
                    abs_path="path",
                    line_no=1,
                    col_no=1,
                )
            ]
        )
        test_repo = RepoDefinition(provider="github", owner="test", name="repo", external_id="1")
        self.autofix_context.repos = [test_repo]

        self.autofix_context._process_stacktrace_paths(stacktrace)

        self.autofix_context.event_manager.on_error.assert_called_once_with(
            error_msg="Autofix does not have access to the `test/repo` repo. Please give permission through the Sentry GitHub integration, or remove the repo from your code mappings.",
            should_completely_error=True,
        )

    @patch("seer.automation.autofix.autofix_context.RepoClient")
    def test_process_stacktrace_paths_ignores_unsupported_providers(self, mock_RepoClient):
        mock_RepoClient.from_repo_definition.side_effect = UnknownObjectException(status=404)
        mock_RepoClient.supported_providers = ["github"]
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="test_file.py",
                    in_app=True,
                    repo_name=None,
                    context=[],
                    abs_path="path",
                    line_no=1,
                    col_no=1,
                )
            ]
        )
        test_repo = RepoDefinition(provider="bitbucket", owner="test", name="repo", external_id="1")
        self.autofix_context.repos = [test_repo]

        self.autofix_context._process_stacktrace_paths(stacktrace)

        self.autofix_context.event_manager.on_error.assert_not_called()


class TestAutofixContextPrCommit(unittest.TestCase):
    def setUp(self):
        error_event = next(generate(SentryEventData))
        self.state = ContinuationState.new(
            AutofixContinuation(
                request=AutofixRequest(
                    organization_id=1,
                    project_id=1,
                    repos=[],
                    issue=IssueDetails(id=0, title="", events=[error_event], short_id="ISSUE_1"),
                ),
            ),
            t=DbStateRunTypes.AUTOFIX,
        )
        self.autofix_context = AutofixContext(self.state, MagicMock())
        self.autofix_context.get_org_slug = MagicMock(return_value="slug")

    @patch("seer.automation.autofix.autofix_context.RepoClient")
    def test_commit_changes(self, mock_RepoClient):
        mock_repo_client = MagicMock()
        mock_branch_ref = MagicMock(ref="test_branch")
        mock_repo_client.create_branch_from_changes.return_value = mock_branch_ref
        mock_pr = MagicMock(number=1, html_url="http://test.com", id=123)
        mock_repo_client.create_pr_from_branch.return_value = mock_pr
        mock_repo_client.provider = "github"
        mock_repo_client.get_branch_ref.return_value = mock_branch_ref

        mock_RepoClient.from_repo_definition.return_value = mock_repo_client

        with self.state.update() as cur:
            cur.codebases = {
                "1": CodebaseState(
                    repo_external_id="1",
                    file_changes=[
                        FileChange(
                            path="test.py",
                            reference_snippet="test",
                            change_type="edit",
                            new_snippet="test2",
                            description="test",
                        )
                    ],
                )
            }
            cur.steps = [
                ChangesStep(
                    key="changes",
                    title="changes_title",
                    type=StepType.CHANGES,
                    status=AutofixStatus.PROCESSING,
                    index=0,
                    changes=[
                        CodebaseChange(
                            repo_external_id="1",
                            repo_name="test",
                            title="This is the title",
                            description="This is the description",
                        )
                    ],
                )
            ]

        self.autofix_context.repos = [
            RepoDefinition(
                provider="github",
                owner="getsentry",
                name="name",
                external_id="1",
            )
        ]

        self.autofix_context.commit_changes(make_pr=True)

        mock_repo_client.create_pr_from_branch.assert_called_once_with(
            mock_branch_ref,
            "This is the title",
            f"👋 Hi there! This PR was automatically generated by Autofix 🤖\n\n\nFixes [ISSUE_1](https://sentry.io/organizations/slug/issues/0/)\n\nThis is the description\n\nIf you have any questions or feedback for the Sentry team about this fix, please email [autofix@sentry.io](mailto:autofix@sentry.io) with the Run ID: {self.state.id}.",
        )

        with Session() as session:
            pr_mapping = session.query(DbPrIdToAutofixRunIdMapping).filter_by(pr_id=123).first()
            self.assertIsNotNone(pr_mapping)

            if pr_mapping:
                cur = self.state.get()
                self.assertEqual(pr_mapping.run_id, cur.run_id)

        state = self.autofix_context.state.get()
        changes_step = cast(ChangesStep, state.find_step(key="changes"))
        self.assertIsNotNone(changes_step)
        self.assertGreater(len(changes_step.changes), 0)

        self.assertIsNotNone(changes_step.changes[0].pull_request)
        if changes_step.changes[0].pull_request:
            self.assertEqual(changes_step.changes[0].pull_request.pr_number, 1)

    @patch("seer.automation.autofix.autofix_context.RepoClient")
    def test_commit_changes_with_draft_branch_name(self, mock_RepoClient):
        """Test that draft_branch_name is used when provided."""
        mock_repo_client = MagicMock()
        mock_branch_ref = MagicMock(ref="draft_test_branch")
        mock_repo_client.create_branch_from_changes.return_value = mock_branch_ref
        mock_repo_client.provider = "github"

        mock_RepoClient.from_repo_definition.return_value = mock_repo_client

        with self.state.update() as cur:
            cur.codebases = {
                "1": CodebaseState(
                    repo_external_id="1",
                    file_changes=[
                        FileChange(
                            path="test.py",
                            reference_snippet="test",
                            change_type="edit",
                            new_snippet="test2",
                            description="test",
                        )
                    ],
                )
            }
            cur.steps = [
                ChangesStep(
                    key="changes",
                    title="changes_title",
                    type=StepType.CHANGES,
                    status=AutofixStatus.PROCESSING,
                    index=0,
                    changes=[
                        CodebaseChange(
                            repo_external_id="1",
                            repo_name="test",
                            title="This is the title",
                            description="This is the description",
                            draft_branch_name="draft/custom-branch-name",
                        )
                    ],
                )
            ]

        self.autofix_context.repos = [
            RepoDefinition(
                provider="github",
                owner="getsentry",
                name="name",
                external_id="1",
            )
        ]

        self.autofix_context.commit_changes()

        # Verify that draft_branch_name was used
        mock_repo_client.create_branch_from_changes.assert_called_once_with(
            pr_title="This is the title",
            file_patches=[],
            branch_name="draft/custom-branch-name",
        )

        # Verify the branch name was updated in the state
        state = self.autofix_context.state.get()
        changes_step = cast(ChangesStep, state.find_step(key="changes"))
        self.assertEqual(changes_step.changes[0].branch_name, "draft_test_branch")

    @patch("seer.automation.autofix.autofix_context.RepoClient")
    def test_commit_changes_skip_branch_creation(self, mock_RepoClient):
        """Test that branch creation is skipped when branch_name already exists."""
        mock_repo_client = MagicMock()
        mock_branch_ref = MagicMock(ref="existing_branch")
        mock_repo_client.get_branch_ref.return_value = mock_branch_ref
        mock_pr = MagicMock(number=1, html_url="http://test.com", id=123)
        mock_repo_client.create_pr_from_branch.return_value = mock_pr
        mock_repo_client.provider = "github"

        mock_RepoClient.from_repo_definition.return_value = mock_repo_client

        with self.state.update() as cur:
            cur.codebases = {
                "1": CodebaseState(
                    repo_external_id="1",
                    file_changes=[
                        FileChange(
                            path="test.py",
                            reference_snippet="test",
                            change_type="edit",
                            new_snippet="test2",
                            description="test",
                        )
                    ],
                )
            }
            cur.steps = [
                ChangesStep(
                    key="changes",
                    title="changes_title",
                    type=StepType.CHANGES,
                    status=AutofixStatus.PROCESSING,
                    index=0,
                    changes=[
                        CodebaseChange(
                            repo_external_id="1",
                            repo_name="test",
                            title="This is the title",
                            description="This is the description",
                            branch_name="existing_branch",  # Branch already exists
                        )
                    ],
                )
            ]

        self.autofix_context.repos = [
            RepoDefinition(
                provider="github",
                owner="getsentry",
                name="name",
                external_id="1",
            )
        ]

        self.autofix_context.commit_changes(make_pr=True)

        # Verify that create_branch_from_changes was NOT called
        mock_repo_client.create_branch_from_changes.assert_not_called()

        # Verify that get_branch_ref was called with the existing branch name
        mock_repo_client.get_branch_ref.assert_called_with("existing_branch")

        # Verify the PR was created with the existing branch
        mock_repo_client.create_pr_from_branch.assert_called_once()
        self.assertEqual(mock_repo_client.create_pr_from_branch.call_args[0][0], mock_branch_ref)

        # Verify the PR details were stored in the state
        state = self.autofix_context.state.get()
        changes_step = cast(ChangesStep, state.find_step(key="changes"))
        self.assertIsNotNone(changes_step.changes[0].pull_request)
        self.assertEqual(changes_step.changes[0].pull_request.pr_number, 1)
        self.assertEqual(changes_step.changes[0].pull_request.pr_url, "http://test.com")
        self.assertEqual(changes_step.changes[0].pull_request.pr_id, 123)


if __name__ == "__main__":
    unittest.main()
