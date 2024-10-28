from unittest.mock import MagicMock

import pytest

from seer.automation.agent.client import LlmClient
from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Usage,
)
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.is_obvious import (
    IsObviousComponent,
    IsObviousOutput,
    IsObviousRequest,
)
from seer.automation.models import EventDetails
from seer.dependency_injection import Module


class TestIsObviousComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=AutofixContext)
        mock_context.state = MagicMock()
        return IsObviousComponent(mock_context)

    def test_invoke_returns_true(self, component):
        mock_llm_client = MagicMock(spec=LlmClient)
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=IsObviousOutput(is_root_cause_clear=True),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            result = component.invoke(IsObviousRequest(event_details=MagicMock(spec=EventDetails)))

        assert result is not None
        assert result.is_root_cause_clear is True

    def test_invoke_returns_false(self, component):
        mock_llm_client = MagicMock(spec=LlmClient)
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=IsObviousOutput(is_root_cause_clear=False),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            result = component.invoke(IsObviousRequest(event_details=MagicMock(spec=EventDetails)))

        assert result is not None
        assert result.is_root_cause_clear is False

    def test_invoke_handles_none_response(self, component):
        mock_llm_client = MagicMock(spec=LlmClient)
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=None,
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            result = component.invoke(IsObviousRequest(event_details=MagicMock(spec=EventDetails)))

        assert result is not None
        assert result.is_root_cause_clear is False

    def test_invoke_formats_prompt_correctly(self, component):
        mock_llm_client = MagicMock(spec=LlmClient)
        mock_event_details = MagicMock(spec=EventDetails)
        mock_event_details.__str__.return_value = "Test event details"

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            component.invoke(IsObviousRequest(event_details=mock_event_details))

        # Verify the prompt was formatted with the event details
        mock_llm_client.generate_structured.assert_called_once()
        prompt_arg = mock_llm_client.generate_structured.call_args[1]["prompt"]
        assert "Test event details" in prompt_arg
