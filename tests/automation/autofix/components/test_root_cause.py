from unittest.mock import MagicMock, patch

import openai
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
from seer.automation.autofix.components.is_root_cause_obvious import IsRootCauseObviousOutput
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPrompt,
    RootCauseAnalysisItemPrompt,
    RootCauseAnalysisRelevantContext,
    RootCauseAnalysisRequest,
    RootCauseRelevantCodeSnippet,
    RootCauseRelevantContext,
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

    @pytest.fixture
    def mock_is_obvious_component(self):
        with patch(
            "seer.automation.autofix.components.root_cause.component.IsRootCauseObviousComponent"
        ) as mock:
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
                    title="Missing Null Check",
                    description="The root cause of the issue is ...",
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
        assert output.causes[0].code_context is None

    def test_no_root_causes_response(self, component, mock_agent):
        mock_agent.return_value.run.return_value = "<NO_ROOT_CAUSES> this is too hard, I give up"

        output = component.invoke(MagicMock())

        assert output.causes == []
        assert output.termination_reason == "this is too hard, I give up"
        # Ensure that the formatter is not called when <NO_ROOT_CAUSES> is returned
        assert mock_agent.return_value.run.call_count == 1

        mock_agent.return_value.run.return_value = "<NO_ROOT_CAUSES> this is too hard, I give up </NO_ROOT_CAUSES> you do it yourself buddy"

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

    def test_root_cause_with_obvious_root_cause(
        self, component, mock_agent, mock_is_obvious_component
    ):
        mock_is_obvious = MagicMock()
        mock_is_obvious.invoke.return_value = IsRootCauseObviousOutput(is_root_cause_clear=True)
        mock_is_obvious_component.return_value = mock_is_obvious

        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    title="Test Root Cause",
                    description="Description",
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
            component.invoke(RootCauseAnalysisRequest(event_details=MagicMock(spec=EventDetails)))

        # Verify agent was created without tools
        mock_agent.assert_called_once()
        tools_arg = mock_agent.call_args[1]["tools"]
        assert tools_arg is None

    def test_root_cause_with_non_obvious_root_cause(
        self, component, mock_agent, mock_is_obvious_component
    ):
        mock_is_obvious = MagicMock()
        mock_is_obvious.invoke.return_value = IsRootCauseObviousOutput(is_root_cause_clear=False)
        mock_is_obvious_component.return_value = mock_is_obvious

        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    title="Test Root Cause",
                    description="Description",
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
            component.invoke(RootCauseAnalysisRequest(event_details=MagicMock(spec=EventDetails)))

        # Verify agent was created with tools
        mock_agent.assert_called_once()
        tools_arg = mock_agent.call_args[1]["tools"]
        assert tools_arg is not None

    def test_root_cause_with_initial_memory_skips_is_obvious(
        self, component, mock_agent, mock_is_obvious_component
    ):
        mock_is_obvious = MagicMock()
        mock_is_obvious_component.return_value = mock_is_obvious

        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    title="Test Root Cause",
                    description="Description",
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
            component.invoke(
                RootCauseAnalysisRequest(
                    event_details=MagicMock(spec=EventDetails),
                    initial_memory=[Message(role="user", content="Hello")],
                )
            )

        mock_is_obvious.invoke.assert_not_called()

        # Verify agent was created with tools (default behavior when skipping is_obvious)
        mock_agent.assert_called_once()
        tools_arg = mock_agent.call_args[1]["tools"]
        assert tools_arg is not None

    def test_root_cause_with_line_numbers(self, component, mock_agent):
        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    title="Test Root Cause",
                    description="Description",
                    relevant_code=RootCauseAnalysisRelevantContext(
                        snippets=[
                            RootCauseRelevantContext(
                                id=0,
                                title="Test Context",
                                description="Test function",
                                snippet=RootCauseRelevantCodeSnippet(
                                    file_path="test.py",
                                    snippet="def test_function():\n    return True",
                                    repo_name=None,
                                ),
                            )
                        ]
                    ),
                )
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        # Mock file contents
        file_contents = (
            "# Some comment\n" "def test_function():\n" "    return True\n" "# More code\n"
        )
        component.context.get_file_contents = MagicMock(return_value=file_contents)

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(MagicMock())

        assert output.causes[0].code_context[0].snippet.start_line == 1
        assert output.causes[0].code_context[0].snippet.end_line == 3

    def test_root_cause_line_numbers_file_not_found(self, component, mock_agent):
        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    title="Test Root Cause",
                    description="Description",
                    relevant_code=RootCauseAnalysisRelevantContext(
                        snippets=[
                            RootCauseRelevantContext(
                                id=0,
                                title="Test Context",
                                description="Test function",
                                snippet=RootCauseRelevantCodeSnippet(
                                    file_path="nonexistent.py",
                                    snippet="def test_function():\n    return True",
                                    repo_name=None,
                                ),
                            )
                        ]
                    ),
                )
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        # Mock file contents to raise an exception
        component.context.get_file_contents = MagicMock(side_effect=FileNotFoundError)

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(MagicMock())

        # Verify that the output is still generated but without line numbers
        assert output.causes[0].code_context[0].snippet.start_line is None
        assert output.causes[0].code_context[0].snippet.end_line is None

    def test_root_cause_line_numbers_no_match(self, component, mock_agent):
        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
            "Formatter",
        ]

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=MultipleRootCauseAnalysisOutputPrompt(
                cause=RootCauseAnalysisItemPrompt(
                    title="Test Root Cause",
                    description="Description",
                    relevant_code=RootCauseAnalysisRelevantContext(
                        snippets=[
                            RootCauseRelevantContext(
                                id=0,
                                title="Test Context",
                                description="Test function",
                                snippet=RootCauseRelevantCodeSnippet(
                                    file_path="test.py",
                                    snippet="def test_function():\n    return True",
                                    repo_name=None,
                                ),
                            )
                        ]
                    ),
                )
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        # Mock file contents with different code
        file_contents = "def different_function():\n    return False"
        component.context.get_file_contents = MagicMock(return_value=file_contents)

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(MagicMock())

        # Verify that the output is still generated but without line numbers
        assert output.causes[0].code_context[0].snippet.start_line is None
        assert output.causes[0].code_context[0].snippet.end_line is None

    def test_root_cause_length_limit_fallback(self, component, mock_agent):
        mock_agent.return_value.run.side_effect = [
            "Some root cause analysis",
        ]

        mock_llm_client = MagicMock()
        # First call raises LengthFinishReasonError
        mock_llm_client.generate_structured.side_effect = [
            openai.LengthFinishReasonError(completion=MagicMock()),
            # Second call (with gpt-4o-mini) succeeds
            LlmGenerateStructuredResponse(
                parsed=MultipleRootCauseAnalysisOutputPrompt(
                    cause=RootCauseAnalysisItemPrompt(
                        title="Test Root Cause",
                        description="Description",
                        relevant_code=None,
                    )
                ),
                metadata=LlmResponseMetadata(
                    model="gpt-4o-mini",
                    provider_name=LlmProviderType.OPENAI,
                    usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                ),
            ),
        ]

        module = Module()
        module.constant(LlmClient, mock_llm_client)

        with module:
            output = component.invoke(MagicMock())

        # Verify the output is correct
        assert output.causes[0].title == "Test Root Cause"
        assert output.causes[0].description == "Description"
        assert output.causes[0].code_context is None

        # Verify that generate_structured was called twice
        assert mock_llm_client.generate_structured.call_count == 2

        # Verify first call used gpt-4o-2024-08-06
        first_call = mock_llm_client.generate_structured.call_args_list[0]
        assert first_call[1]["model"].model_name == "gpt-4o-2024-08-06"

        # Verify second call used gpt-4o-mini
        second_call = mock_llm_client.generate_structured.call_args_list[1]
        assert second_call[1]["model"].model_name == "gpt-4o-mini"
