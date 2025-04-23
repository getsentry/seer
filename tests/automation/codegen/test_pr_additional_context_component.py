import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from seer.automation.codegen.models import (
    CodeFetchIssuesOutput,
    PrAdditionalContextComponent,
    PrAdditionalContextOutput,
    PrAdditionalContextRequest,
    PrFile,
)
from seer.automation.codegen.pr_additional_context_component import (
    PrAdditionalContextComponent as PrAdditionalContextComponent,
)
from seer.automation.models import EventDetails, IssueDetails, Profile, SentryEventData, TraceEvent
from seer.dependency_injection import resolve


class TestPrAdditionalContextComponent(unittest.TestCase):
    def setUp(self):
        self.component = PrAdditionalContextComponent(MagicMock())
        self.mock_rpc_client = MagicMock()
        self.component.rpc_client = self.mock_rpc_client

        # Sample test data
        self.organization_id = 123
        self.pr_files = [
            PrFile(filename="file1.py", content="def test(): pass", patch="@@ -1,1 +1,1 @@"),
            PrFile(filename="file2.py", content="class Test: pass", patch="@@ -1,1 +1,1 @@"),
        ]

        # Load fixture data
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")

        with open(os.path.join(fixtures_dir, "issues", "issue1.json")) as f:
            issue_data = json.load(f)

        with open(os.path.join(fixtures_dir, "profiles", "profile1.json")) as f:
            self.mock_profile_data = json.load(f)

        with open(os.path.join(fixtures_dir, "traces", "trace1.json")) as f:
            self.mock_error_data = json.load(f)

        # Mock issues using fixture data
        self.mock_issues = [
            IssueDetails(
                id="issue1",
                title=issue_data["events"][0]["title"],
                events=[
                    EventDetails(
                        event_id=issue_data["events"][0]["event_id"],
                        organization_id=issue_data["events"][0]["organization_id"],
                        project_id=issue_data["events"][0]["project_id"],
                        profile_id=issue_data["events"][0]["profile_id"],
                        trace_id=issue_data["events"][0]["trace_id"],
                    )
                ],
            ),
            IssueDetails(
                id="issue2",
                title="Test Issue 2",
                events=[
                    EventDetails(
                        event_id="event2",
                        organization_id=self.organization_id,
                        project_id=456,
                        profile_id="profile2",
                        trace_id="trace2",
                    )
                ],
            ),
        ]

    @patch("seer.automation.codegen.pr_additional_context_component.FetchIssuesComponent")
    def test_invoke(self, mock_fetch_issues_component):
        # Setup
        mock_fetch_issues_instance = MagicMock()
        mock_fetch_issues_component.return_value = mock_fetch_issues_instance
        mock_fetch_issues_instance.invoke.return_value = CodeFetchIssuesOutput(
            filename_to_issues={"file1.py": self.mock_issues}
        )

        # Mock profile and trace fetching
        self.mock_rpc_client.call.side_effect = [
            self.mock_profile_data,  # First call for profile
            self.mock_error_data,  # Second call for trace
        ]

        # Execute
        request = PrAdditionalContextRequest(
            organization_id=self.organization_id,
            pr_files=self.pr_files,
        )
        result = self.component.invoke(request)

        # Assert
        self.assertIsInstance(result, PrAdditionalContextOutput)
        self.assertEqual(len(result.filename_to_additional_context), 1)
        self.assertIn("file1.py", result.filename_to_additional_context)

        additional_context = result.filename_to_additional_context["file1.py"]
        self.assertIsInstance(additional_context, PrAdditionalContextComponent)
        self.assertIsNotNone(additional_context.profiles)
        self.assertIsNotNone(additional_context.traces)

        # Verify RPC calls
        self.assertEqual(self.mock_rpc_client.call.call_count, 2)
