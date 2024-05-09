import unittest
from unittest.mock import MagicMock

from johen import generate

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.models import AutofixContinuation, AutofixRequest
from seer.automation.codebase.models import QueryResultDocumentChunk
from seer.automation.models import (
    EventDetails,
    ExceptionDetails,
    IssueDetails,
    SentryEventData,
    Stacktrace,
    StacktraceFrame,
)
from seer.automation.state import LocalMemoryState


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


if __name__ == "__main__":
    unittest.main()
