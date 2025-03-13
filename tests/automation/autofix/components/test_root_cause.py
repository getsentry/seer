from unittest.mock import MagicMock, patch

import pytest

from seer.automation.agent.client import LlmClient
from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Usage,
)
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPrompt,
    RelevantCodeFile,
    RootCauseAnalysisItemPrompt,
    RootCauseAnalysisRequest,
    TimelineEvent,
)
from seer.automation.models import EventDetails
from seer.dependency_injection import Module


class TestRootCauseComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=AutofixContext, event_manager=MagicMock(add_log=MagicMock()))
        mock_context.state = MagicMock()
        mock_context.skip_loading_codebase = True
        return RootCauseAnalysisComponent(mock_context)

    @pytest.fixture
    def mock_agent(self):
        with patch("seer.automation.autofix.components.root_cause.component.AutofixAgent") as mock:
            yield mock

    def test_root_cause_simple_response_parsing(self, component, mock_agent):
        mock_agent.return_value.run.side_effect = [
            "Anything really",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    root_cause_reproduction=[
                        TimelineEvent(
                            title="Missing Null Check",
                            code_snippet_and_analysis="The root cause of the issue is ...",
                            timeline_item_type="internal_code",
                            relevant_code_file=RelevantCodeFile(
                                file_path="test.py",
                                repo_name="owner/repo",
                            ),
                            is_most_important_event=True,
                        )
                    ],
                    summary="The root cause of the issue is ...",
                )
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 1
        assert output.causes[0].root_cause_reproduction is not None
        assert len(output.causes[0].root_cause_reproduction) == 1
        assert output.causes[0].root_cause_reproduction[0].title == "Missing Null Check"
        assert (
            output.causes[0].root_cause_reproduction[0].code_snippet_and_analysis
            == "The root cause of the issue is ..."
        )
        assert output.causes[0].root_cause_reproduction[0].timeline_item_type == "internal_code"
        assert output.causes[0].root_cause_reproduction[0].relevant_code_file is not None

    def test_no_root_causes_response(self, component, mock_agent):
        mock_agent.return_value.run.return_value = "<NO_ROOT_CAUSES> this is too hard, I give up"

        output = component.invoke(MagicMock())

        assert output.causes == []
        assert output.termination_reason == "this is too hard, I give up"
        # Ensure that the formatter is not called when <NO_ROOT_CAUSES> is returned
        assert mock_agent.return_value.run.call_count == 2

    def test_agent_run_returns_none(self, component, mock_agent):
        mock_agent.return_value.run.return_value = None

        output = component.invoke(MagicMock())

        assert output.causes == []
        assert output.termination_reason == "Something went wrong when Autofix was running."
        # Ensure that the formatter is not called
        assert mock_agent.return_value.run.call_count == 1

    def test_root_cause(self, component, mock_agent):
        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    root_cause_reproduction=[
                        TimelineEvent(
                            title="Test Cause",
                            code_snippet_and_analysis="The root cause of the issue is ...",
                            timeline_item_type="internal_code",
                            relevant_code_file=RelevantCodeFile(
                                file_path="test.py",
                                repo_name="owner/repo",
                            ),
                            is_most_important_event=True,
                        )
                    ],
                    summary="The root cause of the issue is ...",
                )
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            component.invoke(RootCauseAnalysisRequest(event_details=MagicMock(spec=EventDetails)))

        # Verify agent was created with tools
        mock_agent.assert_called_once()
        tools_arg = mock_agent.call_args[1]["tools"]
        assert tools_arg is not None
