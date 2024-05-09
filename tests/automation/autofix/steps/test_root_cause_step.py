import unittest
from unittest.mock import MagicMock, patch

from johen import generate

from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisOutput
from seer.automation.autofix.models import AutofixContinuation, AutofixRequest
from seer.automation.autofix.steps.root_cause_step import RootCauseStep
from seer.automation.models import IssueDetails, SentryEventData


class TestRootCauseStep(unittest.TestCase):
    @patch("seer.automation.autofix.steps.root_cause_step.RootCauseAnalysisComponent")
    def test_happy_path(self, mock_root_cause_component):
        mock_context = MagicMock()
        RootCauseStep._instantiate_context = MagicMock(return_value=mock_context)

        error_event = next(generate(SentryEventData))

        mock_context.event_manager = MagicMock()
        mock_context.has_missing_codebase_indexes.return_value = False
        mock_context.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[error_event]),
            )
        )

        mock_root_cause_output = next(generate(RootCauseAnalysisOutput))
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output

        step = RootCauseStep({"run_id": 1, "step_id": 1})

        step.invoke()

        step.context.event_manager.send_codebase_indexing_complete_if_exists.assert_called()
        step.context.event_manager.send_root_cause_analysis_start.assert_called()
        step.context.process_event_paths.assert_called()
        step.context.event_manager.send_root_cause_analysis_result.assert_called()
