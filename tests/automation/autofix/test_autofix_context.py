import unittest
from unittest.mock import MagicMock, patch

from johen import generate

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
from seer.automation.state import LocalMemoryState
from seer.db import DbPrIdToAutofixRunIdMapping, Session


class TestAutofixContext(unittest.TestCase):
    def setUp(self):
        self.mock_repo_client = MagicMock()
        error_event = next(generate(SentryEventData))
        self.state = LocalMemoryState(
            AutofixContinuation(
                request=AutofixRequest(
                    organization_id=1,
                    project_id=1,
                    repos=[],
                    issue=IssueDetails(id=0, title="", events=[error_event]),
                )
            )
        )
        self.autofix_context = AutofixContext(
            self.state,
            MagicMock(),
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

        AutofixContext(state, MagicMock(), mock_event_manager)

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
            )
        )
        self.autofix_context = AutofixContext(self.state, MagicMock(), MagicMock())
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
                    id="changes",
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
            f"👋 Hi there! This PR was automatically generated by Autofix 🤖\n\n\nFixes [ISSUE_1](https://sentry.io/organizations/slug/issues/0/)\n\nThis is the description\n\nIf you have any questions or feedback for the Sentry team about this fix, please email [autofix@sentry.io](mailto:autofix@sentry.io) with the Run ID (see below).\n\n### 🤓 Stats for the nerds:\nRun ID: **{self.state.id}**\nPrompt tokens: **0**\nCompletion tokens: **0**\nTotal tokens: **0**",
        )

        with Session() as session:
            pr_mapping = session.query(DbPrIdToAutofixRunIdMapping).filter_by(pr_id=123).first()
            self.assertIsNotNone(pr_mapping)

            if pr_mapping:
                cur = self.state.get()
                self.assertEqual(pr_mapping.run_id, cur.run_id)


if __name__ == "__main__":
    unittest.main()
