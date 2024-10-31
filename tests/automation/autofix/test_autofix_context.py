import unittest
from unittest.mock import MagicMock, patch

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
from seer.automation.state import DbStateRunTypes, LocalMemoryState
from seer.automation.summarize.issue import IssueSummary
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

    @patch("seer.automation.autofix.autofix_context.AutofixEventManager")
    def test_migrate_step_keys_called_in_init(self, mock_AutofixEventManager):
        mock_event_manager = MagicMock()
        mock_AutofixEventManager.return_value = mock_event_manager

        error_event = next(generate(SentryEventData))
        state = LocalMemoryState(
            AutofixContinuation(
                request=AutofixRequest(
                    organization_id=1,
                    project_id=1,
                    repos=[],
                    issue=IssueDetails(id=0, title="", events=[error_event]),
                )
            )
        )

        AutofixContext(state, mock_event_manager)

        mock_event_manager.migrate_step_keys.assert_called_once()

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
            valid_summary_data = {
                "bulleted_summary_of_the_issue_based_on_your_step_by_step_reasoning": "summary",
                "reason_step_by_step": [],
                "five_to_ten_word_headline": "headline",
            }
            db_issue_summary = DbIssueSummary(group_id=0, summary=valid_summary_data)
            session.add(db_issue_summary)
            session.commit()

        instance = self.autofix_context

        result = instance.get_issue_summary()

        self.assertIsNotNone(result)
        self.assertIsInstance(result, IssueSummary)
        self.assertEqual(
            result.bulleted_summary_of_the_issue_based_on_your_step_by_step_reasoning, "summary"
        )
        self.assertEqual(result.five_to_ten_word_headline, "headline")

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
        mock_repo_client.create_branch_from_changes.return_value = "test_branch"
        mock_pr = MagicMock(number=1, html_url="http://test.com", id=123)
        mock_repo_client.create_pr_from_branch.return_value = mock_pr
        mock_repo_client.provider = "github"

        mock_RepoClient.from_repo_definition.return_value = mock_repo_client

        with self.state.update() as cur:
            cur.codebases = {
                "1": CodebaseState(
                    repo_external_id="1",
                    namespace_id=1,
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

        self.autofix_context.commit_changes()

        mock_repo_client.create_pr_from_branch.assert_called_once_with(
            "test_branch",
            "🤖 This is the title",
            f"👋 Hi there! This PR was automatically generated by Autofix 🤖\n\n\nFixes [ISSUE_1](https://sentry.io/organizations/slug/issues/0/)\n\nThis is the description\n\nIf you have any questions or feedback for the Sentry team about this fix, please email [autofix@sentry.io](mailto:autofix@sentry.io) with the Run ID: {self.state.id}.",
        )

        with Session() as session:
            pr_mapping = session.query(DbPrIdToAutofixRunIdMapping).filter_by(pr_id=123).first()
            self.assertIsNotNone(pr_mapping)

            if pr_mapping:
                cur = self.state.get()
                self.assertEqual(pr_mapping.run_id, cur.run_id)


if __name__ == "__main__":
    unittest.main()
