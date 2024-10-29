from unittest.mock import MagicMock

import pytest

from seer.automation.agent.client import LlmClient
from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Message,
    Usage,
)
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.is_fix_obvious import (
    IsFixObviousComponent,
    IsFixObviousOutput,
    IsFixObviousRequest,
)
from seer.automation.models import EventDetails
from seer.dependency_injection import Module


class TestIsFixObviousComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=AutofixContext)
        mock_context.state = MagicMock()
        return IsFixObviousComponent(mock_context)

    def test_invoke_returns_true(self, component):
        mock_llm_client = MagicMock(spec=LlmClient)
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=IsFixObviousOutput(is_fix_clear=True),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            result = component.invoke(
                IsFixObviousRequest(
                    event_details=MagicMock(spec=EventDetails),
                    task_str="Fix the null pointer exception",
                    fix_instruction="Add null checks",
                    memory=[
                        Message(role="assistant", content="File content"),
                        Message(role="user", content="def example():\n    return None"),
                    ],
                )
            )

        assert result is not None
        assert result.is_fix_clear is True

        # Verify the prompt was formatted with all required information
        mock_llm_client.generate_structured.assert_called_once()
        prompt_arg = mock_llm_client.generate_structured.call_args[1]["prompt"]
        assert "Fix the null pointer exception" in prompt_arg
        assert "Add null checks" in prompt_arg

    def test_invoke_returns_false(self, component):
        mock_llm_client = MagicMock(spec=LlmClient)
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=IsFixObviousOutput(is_fix_clear=False),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            result = component.invoke(
                IsFixObviousRequest(
                    event_details=MagicMock(spec=EventDetails),
                    task_str="Complex task requiring investigation",
                    fix_instruction=None,
                    memory=[],
                )
            )

        assert result is not None
        assert result.is_fix_clear is False

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
            result = component.invoke(
                IsFixObviousRequest(
                    event_details=MagicMock(spec=EventDetails),
                    task_str="Some task",
                    fix_instruction=None,
                    memory=[],
                )
            )

        assert result is not None
        assert result.is_fix_clear is False
