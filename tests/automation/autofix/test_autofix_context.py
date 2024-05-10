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
from seer.automation.codebase.models import QueryResultDocumentChunk, RepositoryInfo
from seer.automation.models import FileChange, IssueDetails, SentryEventData
from seer.automation.state import LocalMemoryState
from seer.db import DbPrIdToAutofixRunIdMapping, Session


class TestAutofixContext(unittest.TestCase):
    def setUp(self):
        self.mock_codebase_index = MagicMock()
        self.mock_repo_client = MagicMock()
        self.mock_codebase_index.repo_client = self.mock_repo_client
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
        self.autofix_context.get_codebase = MagicMock(return_value=self.mock_codebase_index)

    def test_multi_codebase_query(self):
        chunks: list[QueryResultDocumentChunk] = []
        for _ in range(8):
            chunks.append(next(generate(QueryResultDocumentChunk)))

        self.autofix_context.codebases = {
            1: MagicMock(query=MagicMock(return_value=chunks[:3])),
            2: MagicMock(query=MagicMock(return_value=chunks[3:])),
        }

        sorted_chunks = sorted(chunks, key=lambda x: x.distance)
        result_chunks = self.autofix_context.query_all_codebases("test", top_k=8)

        self.assertEqual(result_chunks, sorted_chunks)


class TestAutofixContextPrCommit(unittest.TestCase):
    def setUp(self):
        error_event = next(generate(SentryEventData))
        self.state = LocalMemoryState(
            AutofixContinuation(
                request=AutofixRequest(
                    organization_id=1,
                    project_id=1,
                    repos=[],
                    issue=IssueDetails(id=0, title="", events=[error_event], short_id="ISSUE_1"),
                )
            )
        )
        self.autofix_context = AutofixContext(
            self.state, MagicMock(), MagicMock(), skip_loading_codebase=True
        )
        self.autofix_context._get_org_slug = MagicMock(return_value="slug")

    @patch(
        "seer.automation.autofix.autofix_context.CodebaseIndex.get_repo_info_from_db",
        return_value=RepositoryInfo(
            id=1,
            organization=1,
            project=1,
            provider="github",
            external_slug="getsentry/slug",
            external_id="1",
        ),
    )
    @patch("seer.automation.autofix.autofix_context.RepoClient")
    def test_commit_changes(self, mock_RepoClient, mock_get_repo_info_from_db):
        mock_repo_client = MagicMock()
        mock_repo_client.create_branch_from_changes.return_value = "test_branch"
        mock_pr = MagicMock(number=1, html_url="http://test.com", id=123)
        mock_repo_client.create_pr_from_branch.return_value = mock_pr

        mock_RepoClient.from_repo_info.return_value = mock_repo_client

        with self.state.update() as cur:
            cur.codebases = {
                1: CodebaseState(
                    repo_id=1,
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
                    status=AutofixStatus.PENDING,
                    index=0,
                    changes=[
                        CodebaseChange(
                            repo_id=1,
                            repo_name="test",
                            title="This is the title",
                            description="This is the description",
                        )
                    ],
                )
            ]

        self.autofix_context.commit_changes()

        mock_repo_client.create_pr_from_branch.assert_called_once_with(
            "test_branch",
            "🤖 This is the title",
            "👋 Hi there! This PR was automatically generated 🤖\n\n\nFixes [ISSUE_1](https://sentry.io/organizations/slug/issues/0/)\n\nThis is the description\n\n### 📣 Instructions for the reviewer which is you, yes **you**:\n- **If these changes were incorrect, please close this PR and comment explaining why.**\n- **If these changes were incomplete, please continue working on this PR then merge it.**\n- **If you are feeling confident in my changes, please merge this PR.**\n\nThis will greatly help us improve the autofix system. Thank you! 🙏\n\nIf there are any questions, please reach out to the [AI/ML Team](https://github.com/orgs/getsentry/teams/machine-learning-ai) on [#proj-autofix](https://sentry.slack.com/archives/C06904P7Z6E)\n\n### 🤓 Stats for the nerds:\nPrompt tokens: **0**\nCompletion tokens: **0**\nTotal tokens: **0**",
        )

        with Session() as session:
            pr_mapping = session.query(DbPrIdToAutofixRunIdMapping).filter_by(pr_id=123).first()
            self.assertIsNotNone(pr_mapping)

            if pr_mapping:
                cur = self.state.get()
                self.assertEqual(pr_mapping.run_id, cur.run_id)


if __name__ == "__main__":
    unittest.main()
