from unittest.mock import MagicMock, patch
from xml.etree.ElementTree import ParseError

import pytest

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseAnalysisOutput,
    RootCauseRelevantCodeSnippet,
    RootCauseRelevantContext,
)


class TestRootCauseComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=AutofixContext)
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
            "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description></potential_cause></potential_root_causes>",
        ]

        output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 1
        assert output.causes[0].title == "Missing Null Check"
        assert output.causes[0].description == "The root cause of the issue is ..."
        assert output.causes[0].likelihood == 0.9
        assert output.causes[0].actionability == 1.0
        assert output.causes[0].code_context is None

    def test_root_cause_code_context_response_parsing(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "Anything really",
            "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description><code_context><code_snippet><title>Add Null Check</title><description>This fix involves adding ...</description><code file_path='some/app/path.py' repo_name='owner/repo'>def foo():</code></code_snippet></code_context></potential_cause></potential_root_causes>",
        ]

        output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 1
        assert output.causes[0].title == "Missing Null Check"
        assert output.causes[0].description == "The root cause of the issue is ..."
        assert output.causes[0].likelihood == 0.9
        assert output.causes[0].actionability == 1.0
        assert output.causes[0].code_context is not None
        assert len(output.causes[0].code_context) == 1
        assert output.causes[0].code_context[0].title == "Add Null Check"
        assert output.causes[0].code_context[0].description == "This fix involves adding ..."
        assert output.causes[0].code_context[0].snippet.file_path == "some/app/path.py"
        assert output.causes[0].code_context[0].snippet.repo_name == "owner/repo"
        assert output.causes[0].code_context[0].snippet.snippet == "def foo():"

    def test_root_cause_multiple_causes_response_parsing(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "Anything really",
            "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description></potential_cause><potential_cause likelihood='0.5' actionability='0.5'><title>Incorrect API Usage</title><description>Another potential cause is ...</description></potential_cause></potential_root_causes>",
        ]
        output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 2
        assert output.causes[0].title == "Missing Null Check"
        assert output.causes[1].title == "Incorrect API Usage"

    def test_root_cause_empty_response_parsing(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "Anything really",
            "<potential_root_causes></potential_root_causes>",
        ]

        output = component.invoke(MagicMock())

        assert output is None

    def test_root_cause_invalid_xml_response(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "Anything really",
            "<potential_root_causes><invalid_xml></potential_root_causes>",
        ]

        with pytest.raises(ParseError):
            component.invoke(MagicMock())

    def test_root_cause_missing_required_fields(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "Anything really",
            "<potential_root_causes><potential_cause likelihood='0.9'><title>Missing Null Check</title></potential_cause></potential_root_causes>",
        ]

        with pytest.raises(ValueError):
            component.invoke(MagicMock())

    def test_root_cause_invalid_likelihood_actionability(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "Anything really",
            "<potential_root_causes><potential_cause likelihood='2.0' actionability='-0.5'><title>Invalid Values</title><description>Test</description></potential_cause></potential_root_causes>",
        ]

        with pytest.raises(ValueError):
            component.invoke(MagicMock())

    def test_root_cause_no_formatter_response(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.side_effect = [
            "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Test</title><description>Test</description></potential_cause></potential_root_causes>",
            None,
        ]

        output = component.invoke(MagicMock())

        assert output is None

    def test_root_cause_analysis_output_model(self):
        output = RootCauseAnalysisOutput(
            causes=[
                RootCauseAnalysisItem(
                    id=0,
                    title="Test Cause",
                    description="Test Description",
                    likelihood=0.8,
                    actionability=0.9,
                    code_context=[
                        RootCauseRelevantContext(
                            id=0,
                            title="Test Fix",
                            description="Test Fix Description",
                            snippet=RootCauseRelevantCodeSnippet(
                                file_path="test.py",
                                snippet="def test():\n    pass",
                                repo_name="owner/repo",
                            ),
                        )
                    ],
                )
            ]
        )

        assert len(output.causes) == 1
        assert output.causes[0].id == 0
        assert output.causes[0].title == "Test Cause"
        assert output.causes[0].description == "Test Description"
        assert output.causes[0].likelihood == 0.8
        assert output.causes[0].actionability == 0.9
        assert len(output.causes[0].code_context or []) == 1
        if output.causes[0].code_context:
            assert output.causes[0].code_context[0].id == 0
            assert output.causes[0].code_context[0].title == "Test Fix"
            assert output.causes[0].code_context[0].description == "Test Fix Description"
            if output.causes[0].code_context[0].snippet:
                assert output.causes[0].code_context[0].snippet.file_path == "test.py"
                assert output.causes[0].code_context[0].snippet.snippet == "def test():\n    pass"
                assert output.causes[0].code_context[0].snippet.repo_name == "owner/repo"

    def test_no_root_causes_response(self, component, mock_gpt_agent):
        mock_gpt_agent.return_value.run.return_value = "<NO_ROOT_CAUSES>"

        output = component.invoke(MagicMock())

        assert output is None
        # Ensure that the second run (formatter) is not called when <NO_ROOT_CAUSES> is returned
        assert mock_gpt_agent.return_value.run.call_count == 1
