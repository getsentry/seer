from unittest.mock import MagicMock, patch

import pytest

from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.component import InsightSharingComponent
from seer.automation.autofix.components.insight_sharing.models import (
    InsightSharingOutput,
    InsightSharingRequest,
)


class TestInsightSharingComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=AutofixContext)
        mock_context.state = MagicMock()
        mock_context.skip_loading_codebase = True
        return InsightSharingComponent(mock_context)

    @pytest.fixture
    def mock_gpt_client(self):
        with patch(
            "seer.automation.autofix.components.insight_sharing.component.GptClient"
        ) as mock:
            yield mock

    def test_invoke_with_insight(self, component, mock_gpt_client):
        request = InsightSharingRequest(
            task_description="Test task",
            latest_thought="Latest thought",
            past_insights=["Past insight 1", "Past insight 2"],
            memory=[Message(role="user", content="Test memory")],
        )

        mock_completion_1 = MagicMock()
        mock_completion_1.choices[0].message.content = "New insight"
        mock_completion_1.usage = MagicMock(completion_tokens=10, prompt_tokens=20, total_tokens=30)

        mock_completion_2 = MagicMock()
        mock_completion_2.choices[0].message.parsed = MagicMock(
            explanation="Test explanation",
            error_message_context=["Test error context"],
            codebase_context=[],
            stacktrace_context=[],
            event_log_context=[],
        )
        mock_completion_2.choices[0].message.refusal = None

        mock_gpt_client.openai_client.chat.completions.create.return_value = mock_completion_1
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = mock_completion_2

        result = component.invoke(request, mock_gpt_client)
        assert isinstance(result, InsightSharingOutput)
        assert result.insight == "New insight"
        assert result.justification == "Test explanation"
        assert result.error_message_context == ["Test error context"]
        assert result.codebase_context == []
        assert result.stacktrace_context == []
        assert result.breadcrumb_context == []

    def test_invoke_with_no_insight(self, component, mock_gpt_client):
        request = InsightSharingRequest(
            task_description="Test task",
            latest_thought="Latest thought",
            past_insights=["Past insight 1", "Past insight 2"],
            memory=[Message(role="user", content="Test memory")],
        )

        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "<NO_INSIGHT/>"

        mock_gpt_client.openai_client.chat.completions.create.return_value = mock_completion

        result = component.invoke(request, mock_gpt_client)

        assert result is None

    def test_invoke_with_error(self, component, mock_gpt_client):
        request = InsightSharingRequest(
            task_description="Test task",
            latest_thought="Latest thought",
            past_insights=["Past insight 1"],
            memory=[Message(role="user", content="Test memory")],
        )

        mock_completion_1 = MagicMock()
        mock_completion_1.choices[0].message.content = "New insight"
        mock_completion_1.usage = MagicMock(completion_tokens=10, prompt_tokens=20, total_tokens=30)

        mock_completion_2 = MagicMock()
        mock_completion_2.choices[0].message.refusal = "Test refusal"

        mock_gpt_client.openai_client.chat.completions.create.return_value = mock_completion_1
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = mock_completion_2

        # make sure no error is raised
        component.invoke(request, mock_gpt_client)
