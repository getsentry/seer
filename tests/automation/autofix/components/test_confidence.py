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
from seer.automation.autofix.components.confidence import (
    ConfidenceComponent,
    ConfidenceOutput,
    ConfidencePrompts,
    ConfidenceRequest,
)
from seer.dependency_injection import Module


class TestConfidenceComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=AutofixContext, event_manager=MagicMock(add_log=MagicMock()))
        mock_context.state = MagicMock()
        mock_context.skip_loading_codebase = True
        return ConfidenceComponent(mock_context)

    def test_confidence_successful_response(self, component):
        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=ConfidenceOutput(
                question="Need more information about the error handling.",
                output_confidence_score=0.75,
                proceed_confidence_score=0.85,
            ),
            metadata=LlmResponseMetadata(
                model="gemini-2.0-flash-001",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        request = ConfidenceRequest(
            run_memory=[Message(role="user", content="Test message")],
            step_goal_description="analyzing the root cause",
            next_step_goal_description="implementing a fix",
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(request)

        assert output is not None
        assert output.question == "Need more information about the error handling."
        assert output.output_confidence_score == 0.75
        assert output.proceed_confidence_score == 0.85

        mock_llm_client.generate_structured.assert_called_once()
        call_args = mock_llm_client.generate_structured.call_args[1]
        assert call_args["prompt"] == ConfidencePrompts.format_default_msg(
            step_goal_description="analyzing the root cause",
            next_step_goal_description="implementing a fix",
        )
        assert call_args["system_prompt"] == ConfidencePrompts.format_system_msg()
        assert call_args["messages"] == [Message(role="user", content="Test message")]

    def test_confidence_null_comment(self, component):
        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=ConfidenceOutput(
                question=None,
                output_confidence_score=0.95,
                proceed_confidence_score=0.90,
            ),
            metadata=LlmResponseMetadata(
                model="gemini-2.0-flash-001",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        request = ConfidenceRequest(
            run_memory=[],
            step_goal_description="analyzing the code",
            next_step_goal_description="writing tests",
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(request)

        assert output is not None
        assert output.question is None
        assert output.output_confidence_score == 0.95
        assert output.proceed_confidence_score == 0.90

    def test_confidence_none_response(self, component):
        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=None,
            metadata=LlmResponseMetadata(
                model="gemini-2.0-flash-001",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        request = ConfidenceRequest(
            run_memory=[],
            step_goal_description="analyzing the code",
            next_step_goal_description="writing tests",
        )

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(request)

        assert output is not None
        assert output.output_confidence_score == 0.5
        assert output.proceed_confidence_score == 0.5
        assert output.question is None
