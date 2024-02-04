import unittest
from unittest.mock import MagicMock, patch

from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.models import (
    AutofixRequest,
    IssueDetails,
    PlanningOutput,
    ProblemDiscoveryOutput,
    ProblemDiscoveryResult,
    SentryEvent,
)


class TestAutoFixRunFlow(unittest.TestCase):
    @patch("seer.automation.autofix.autofix.AutofixEventManager")
    @patch("seer.automation.autofix.autofix.RepoClient")
    @patch("seer.automation.autofix.autofix.ContextManager")
    def test_problem_discovery_short_circuit(
        self,
        mock_context_manager,
        mock_repo_client,
        mock_event_manager,
    ):
        autofix = Autofix(
            request=AutofixRequest(
                additional_context="",
                base_commit_sha="",
                issue=IssueDetails(
                    id=1,
                    title="",
                    events=[
                        SentryEvent(
                            entries=[
                                {
                                    "type": "exception",
                                    "data": {
                                        "values": [
                                            {
                                                "stacktrace": {
                                                    "frames": [
                                                        {
                                                            "function": "test",
                                                            "filename": "file2.py",
                                                            "lineNo": 10,
                                                            "colNo": 0,
                                                            "context": [],
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    },
                                }
                            ]
                        )
                    ],
                ),
            ),
            rpc_client=MagicMock(),
        )

        autofix.run_problem_discovery_agent = MagicMock()
        autofix.run_problem_discovery_agent.return_value = ProblemDiscoveryOutput(
            actionability_score=0.5,
            description="",
            reasoning="",
        )

        autofix.run()

        autofix.run_problem_discovery_agent.assert_called_once()
        autofix.event_manager.send_problem_discovery_result.assert_called_once_with(
            ProblemDiscoveryResult(
                status="CANCELLED",
                description="",
                reasoning="",
            )
        )
        autofix.context_manager.load_codebase.assert_not_called()

    @patch("seer.automation.autofix.autofix.AutofixEventManager")
    @patch("seer.automation.autofix.autofix.RepoClient")
    @patch("seer.automation.autofix.autofix.ContextManager")
    def test_stacktrace_files_short_circuit(
        self,
        mock_context_manager,
        mock_repo_client,
        mock_event_manager,
    ):
        autofix = Autofix(
            request=AutofixRequest(
                additional_context="",
                base_commit_sha="",
                issue=IssueDetails(
                    id=1,
                    title="",
                    events=[
                        SentryEvent(
                            entries=[
                                {
                                    "type": "exception",
                                    "data": {
                                        "values": [
                                            {
                                                "stacktrace": {
                                                    "frames": [
                                                        {
                                                            "function": "test",
                                                            "filename": "file2.py",
                                                            "lineNo": 10,
                                                            "colNo": 0,
                                                            "context": [],
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    },
                                }
                            ]
                        )
                    ],
                ),
            ),
            rpc_client=MagicMock(),
        )

        autofix.context_manager.diff_contains_stacktrace_files = MagicMock()
        autofix.context_manager.diff_contains_stacktrace_files.return_value = False
        autofix.run_problem_discovery_agent = MagicMock()
        autofix.run_problem_discovery_agent.return_value = ProblemDiscoveryOutput(
            actionability_score=1.0,
            description="",
            reasoning="",
        )
        autofix.run_planning_agent = MagicMock()
        autofix.run_planning_agent.return_value = PlanningOutput(title="", description="", steps=[])
        autofix.run_execution_agent = MagicMock()

        autofix.run()

        autofix.run_problem_discovery_agent.assert_called_once()
        autofix.event_manager.send_problem_discovery_result.assert_called_once_with(
            ProblemDiscoveryResult(
                status="CONTINUE",
                description="",
                reasoning="",
            )
        )
        autofix.context_manager.load_codebase.assert_called_once()

        assert autofix.context_manager.codebase_context is not None
        autofix.context_manager.codebase_context.update_codebase_index.assert_not_called()

        autofix.run_planning_agent.assert_called_once()
        autofix.run_execution_agent.assert_not_called()

    @patch("seer.automation.autofix.autofix.AutofixEventManager")
    @patch("seer.automation.autofix.autofix.RepoClient")
    @patch("seer.automation.autofix.autofix.ContextManager")
    def test_stacktrace_not_short_circuit(
        self,
        mock_context_manager,
        mock_repo_client,
        mock_event_manager,
    ):
        autofix = Autofix(
            request=AutofixRequest(
                additional_context="",
                base_commit_sha="",
                issue=IssueDetails(
                    id=1,
                    title="",
                    events=[
                        SentryEvent(
                            entries=[
                                {
                                    "type": "exception",
                                    "data": {
                                        "values": [
                                            {
                                                "stacktrace": {
                                                    "frames": [
                                                        {
                                                            "function": "test",
                                                            "filename": "file2.py",
                                                            "lineNo": 10,
                                                            "colNo": 0,
                                                            "context": [],
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    },
                                }
                            ]
                        )
                    ],
                ),
            ),
            rpc_client=MagicMock(),
        )

        autofix.context_manager.diff_contains_stacktrace_files = MagicMock()
        autofix.context_manager.diff_contains_stacktrace_files.return_value = True
        autofix.run_problem_discovery_agent = MagicMock()
        autofix.run_problem_discovery_agent.return_value = ProblemDiscoveryOutput(
            actionability_score=1.0,
            description="",
            reasoning="",
        )
        autofix.run_planning_agent = MagicMock()
        autofix.run_planning_agent.return_value = PlanningOutput(title="", description="", steps=[])
        autofix.run_execution_agent = MagicMock()

        autofix.run()

        autofix.run_problem_discovery_agent.assert_called_once()
        autofix.event_manager.send_problem_discovery_result.assert_called_once_with(
            ProblemDiscoveryResult(
                status="CONTINUE",
                description="",
                reasoning="",
            )
        )
        autofix.context_manager.load_codebase.assert_called_once()

        assert autofix.context_manager.codebase_context is not None
        autofix.context_manager.codebase_context.update_codebase_index.assert_called_once()

        autofix.run_planning_agent.assert_called_once()
        autofix.run_execution_agent.assert_not_called()
