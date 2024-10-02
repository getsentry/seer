from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ParsedChatCompletion, ParsedChatCompletionMessage, ParsedChoice

from seer.automation.agent.client import GptClient
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
    def mock_gpt_agent(self):
        with patch("seer.automation.autofix.components.root_cause.component.GptAgent") as mock:
            yield mock

    def test_root_cause_simple_response_parsing(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "Anything really",
            "Reproduction steps",
        ]

        mock_gpt_client = MagicMock()
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = (
            ParsedChatCompletion(
                id="1",
                choices=[
                    ParsedChoice(
                        index=0,
                        message=ParsedChatCompletionMessage(
                            role="assistant",
                            content=None,
                            function_call=None,
                            tool_calls=None,
                            parsed=MultipleRootCauseAnalysisOutputPrompt(
                                cause=RootCauseAnalysisItemPrompt(
                                    title="Missing Null Check",
                                    description="The root cause of the issue is ...",
                                    reproduction="Steps to reproduce",
                                    relevant_code=None,
                                )
                            ),
                            refusal=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                created=1234567890,
                model="gpt-4o-2024-08-06",
                object="chat.completion",
                system_fingerprint="test",
                usage=None,
            )
        )

        module = Module()
        module.constant(GptClient, mock_gpt_client)
        with module:
            output = component.invoke(MagicMock())

            assert output is not None
            assert len(output.causes) == 1
            assert output.causes[0].title == "Missing Null Check"
            assert output.causes[0].description == "The root cause of the issue is ..."
            assert output.causes[0].reproduction == "Steps to reproduce"
            assert output.causes[0].code_context is None

    def test_no_root_causes_response(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.return_value = "<NO_ROOT_CAUSES>"

        output = component.invoke(MagicMock())

        assert output is None
        # Ensure that the second run (reproduction) and the formatter are not called when <NO_ROOT_CAUSES> is returned
        assert mock_gpt_agent.return_value.run.call_count == 1
