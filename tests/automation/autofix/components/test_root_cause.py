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
    RootCauseAnalysisItemPrompt,
)
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
            "Reproduction steps",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    title="Missing Null Check",
                    description="The root cause of the issue is ...",
                    reproduction_instructions="Steps to reproduce",
                    relevant_code=None,
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
        assert output.causes[0].title == "Missing Null Check"
        assert output.causes[0].description == "The root cause of the issue is ..."
        assert output.causes[0].reproduction == "Steps to reproduce"
        assert output.causes[0].code_context is None

    def test_no_root_causes_response(self, component, mock_agent):
        mock_agent.return_value.run.return_value = "<NO_ROOT_CAUSES> this is too hard, I give up"

        output = component.invoke(MagicMock())

        assert output.causes == []
        assert output.termination_reason == "this is too hard, I give up"
        # Ensure that the second run (reproduction) and the formatter are not called when <NO_ROOT_CAUSES> is returned
        assert mock_agent.return_value.run.call_count == 1
