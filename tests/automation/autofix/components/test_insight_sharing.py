from unittest.mock import patch

import pytest
from johen import generate

from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmGenerateTextResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Message,
    Usage,
)
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.component import InsightSharingComponent
from seer.automation.autofix.components.insight_sharing.models import (
    InsightContextOutput,
    InsightSharingOutput,
    InsightSharingRequest,
)
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixContinuation
from seer.automation.state import TestMemoryState


class TestInsightSharingComponent:
    @pytest.fixture
    def component(self):
        state = TestMemoryState(next(generate(AutofixContinuation)))
        return InsightSharingComponent(
            AutofixContext(state=state, event_manager=AutofixEventManager(state=state))
        )

    @pytest.fixture
    def mock_llm_client(self):
        with patch(
            "seer.automation.autofix.components.insight_sharing.component.LlmClient"
        ) as mock:
            yield mock

    def test_invoke_with_insight(self, component, mock_llm_client):
        request = InsightSharingRequest(
            task_description="Test task",
            latest_thought="Latest thought",
            past_insights=["Past insight 1", "Past insight 2"],
            memory=[Message(role="user", content="Test memory")],
            generated_at_memory_index=0,
        )

        mock_generate_text_response = LlmGenerateTextResponse(
            message=Message(role="assistant", content="New insight"),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        mock_generate_structured_response = LlmGenerateStructuredResponse(
            parsed=InsightContextOutput(
                explanation="Test explanation",
                error_message_context=["Test error context"],
                codebase_context=[],
                stacktrace_context=[],
                event_log_context=[],
            ),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        mock_llm_client.generate_text.return_value = mock_generate_text_response
        mock_llm_client.generate_structured.return_value = mock_generate_structured_response

        result = component.invoke(request, mock_llm_client)
        assert isinstance(result, InsightSharingOutput)
        assert result.insight == "New insight"
        assert result.justification == "Test explanation"
        assert result.error_message_context == ["Test error context"]
        assert result.codebase_context == []
        assert result.stacktrace_context == []
        assert result.breadcrumb_context == []

    def test_invoke_with_no_insight(self, component, mock_llm_client):
        request = InsightSharingRequest(
            task_description="Test task",
            latest_thought="Latest thought",
            past_insights=["Past insight 1", "Past insight 2"],
            memory=[Message(role="user", content="Test memory")],
            generated_at_memory_index=0,
        )

        mock_llm_client.generate_text.return_value = LlmGenerateTextResponse(
            message=Message(role="assistant", content="<NO_INSIGHT/>"),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = component.invoke(request, mock_llm_client)

        assert result is None

    def test_invoke_with_error(self, component, mock_llm_client):
        request = InsightSharingRequest(
            task_description="Test task",
            latest_thought="Latest thought",
            past_insights=["Past insight 1"],
            memory=[Message(role="user", content="Test memory")],
            generated_at_memory_index=0,
        )

        mock_completion_1 = LlmGenerateTextResponse(
            message=Message(role="assistant", content="New insight"),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        mock_completion_2 = LlmGenerateStructuredResponse(
            parsed=InsightContextOutput(
                explanation="Test explanation",
                error_message_context=["Test error context"],
                codebase_context=[],
                stacktrace_context=[],
                event_log_context=[],
            ),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        mock_llm_client.generate_text.return_value = mock_completion_1
        mock_llm_client.generate_structured.return_value = mock_completion_2

        # make sure no error is raised
        component.invoke(request, mock_llm_client)

    def test_exception_is_caught(self, component, mock_llm_client):
        request = InsightSharingRequest(
            task_description="Test task",
            latest_thought="Latest thought",
            past_insights=["Past insight 1"],
            memory=[Message(role="user", content="Test memory")],
            generated_at_memory_index=0,
        )

        mock_llm_client.generate_text.side_effect = Exception("Test exception")

        # make sure no error is raised
        result = component.invoke(request, mock_llm_client)
        assert result is None
