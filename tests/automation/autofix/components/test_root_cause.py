from unittest.mock import MagicMock, patch
from xml.etree.ElementTree import ParseError

import pytest

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseAnalysisOutput,
    RootCauseSuggestedFix,
    RootCauseSuggestedFixSnippet,
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

    @pytest.fixture
    def mock_gpt_client(self):
        with patch("seer.automation.autofix.components.root_cause.component.GptClient") as mock:
            yield mock

    def test_root_cause_simple_response_parsing(self, component, mock_gpt_agent, mock_gpt_client):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description></potential_cause></potential_root_causes>"
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(
                content="<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description></potential_cause></potential_root_causes>"
            ),
            None,
        )

        output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 1
        assert output.causes[0].title == "Missing Null Check"
        assert output.causes[0].description == "The root cause of the issue is ..."
        assert output.causes[0].likelihood == 0.9
        assert output.causes[0].actionability == 1.0
        assert output.causes[0].suggested_fixes is None

    def test_root_cause_suggested_fix_response_parsing(
        self, component, mock_gpt_agent, mock_gpt_client
    ):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description><suggested_fixes><suggested_fix elegance='0.9'><title>Add Null Check</title><description>This fix involves adding ...</description><snippet file_path='some/app/path.py'>def foo():</snippet></suggested_fix></suggested_fixes></potential_cause></potential_root_causes>"
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(
                content="<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description><suggested_fixes><suggested_fix elegance='0.9'><title>Add Null Check</title><description>This fix involves adding ...</description><snippet file_path='some/app/path.py'>def foo():</snippet></suggested_fix></suggested_fixes></potential_cause></potential_root_causes>"
            ),
            None,
        )

        output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 1
        assert output.causes[0].title == "Missing Null Check"
        assert output.causes[0].description == "The root cause of the issue is ..."
        assert output.causes[0].likelihood == 0.9
        assert output.causes[0].actionability == 1.0
        assert output.causes[0].suggested_fixes is not None
        assert len(output.causes[0].suggested_fixes) == 1
        assert output.causes[0].suggested_fixes[0].title == "Add Null Check"
        assert output.causes[0].suggested_fixes[0].description == "This fix involves adding ..."
        assert output.causes[0].suggested_fixes[0].elegance == 0.9
        assert output.causes[0].suggested_fixes[0].snippet.file_path == "some/app/path.py"
        assert output.causes[0].suggested_fixes[0].snippet.snippet == "def foo():"

    def test_root_cause_multiple_causes_response_parsing(
        self, component, mock_gpt_agent, mock_gpt_client
    ):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description></potential_cause><potential_cause likelihood='0.5' actionability='0.5'><title>Incorrect API Usage</title><description>Another potential cause is ...</description></potential_cause></potential_root_causes>"
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(
                content="<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Missing Null Check</title><description>The root cause of the issue is ...</description></potential_cause><potential_cause likelihood='0.5' actionability='0.5'><title>Incorrect API Usage</title><description>Another potential cause is ...</description></potential_cause></potential_root_causes>"
            ),
            None,
        )

        output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 2
        assert output.causes[0].title == "Missing Null Check"
        assert output.causes[1].title == "Incorrect API Usage"

    def test_root_cause_empty_response_parsing(self, component, mock_gpt_agent, mock_gpt_client):
        mock_gpt_agent.return_value.run.return_value = (
            "<potential_root_causes></potential_root_causes>"
        )
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(content="<potential_root_causes></potential_root_causes>"),
            None,
        )

        output = component.invoke(MagicMock())

        assert output is not None
        assert len(output.causes) == 0

    def test_root_cause_invalid_xml_response(self, component, mock_gpt_agent, mock_gpt_client):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><invalid_xml>"
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(content="<potential_root_causes><invalid_xml>"),
            None,
        )

        with pytest.raises(ParseError):
            component.invoke(MagicMock())

    def test_root_cause_missing_required_fields(self, component, mock_gpt_agent, mock_gpt_client):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><potential_cause likelihood='0.9'><title>Missing Null Check</title></potential_cause></potential_root_causes>"
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(
                content="<potential_root_causes><potential_cause likelihood='0.9'><title>Missing Null Check</title></potential_cause></potential_root_causes>"
            ),
            None,
        )

        with pytest.raises(ValueError):
            component.invoke(MagicMock())

    def test_root_cause_invalid_likelihood_actionability(
        self, component, mock_gpt_agent, mock_gpt_client
    ):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><potential_cause likelihood='2.0' actionability='-0.5'><title>Invalid Values</title><description>Test</description></potential_cause></potential_root_causes>"
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(
                content="<potential_root_causes><potential_cause likelihood='2.0' actionability='-0.5'><title>Invalid Values</title><description>Test</description></potential_cause></potential_root_causes>"
            ),
            None,
        )

        with pytest.raises(ValueError):
            component.invoke(MagicMock())

    def test_root_cause_formatter_usage(self, component, mock_gpt_agent, mock_gpt_client):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Test</title><description>Test</description></potential_cause></potential_root_causes>"
        mock_gpt_client.return_value.completion.return_value = (
            MagicMock(
                content="<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Test</title><description>Test</description></potential_cause></potential_root_causes>"
            ),
            {"total_tokens": 100},
        )

        component.invoke(MagicMock())

        component.context.state.update.assert_called()
        component.context.state.update().__enter__().usage += {"total_tokens": 100}

    def test_root_cause_no_formatter_response(self, component, mock_gpt_agent, mock_gpt_client):
        mock_gpt_agent.return_value.run.return_value = "<potential_root_causes><potential_cause likelihood='0.9' actionability='1.0'><title>Test</title><description>Test</description></potential_cause></potential_root_causes>"
        mock_gpt_client.return_value.completion.return_value = (MagicMock(content=None), None)

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
                    suggested_fixes=[
                        RootCauseSuggestedFix(
                            id=0,
                            title="Test Fix",
                            description="Test Fix Description",
                            snippet=RootCauseSuggestedFixSnippet(
                                file_path="test.py", snippet="def test():\n    pass"
                            ),
                            elegance=0.7,
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
        assert len(output.causes[0].suggested_fixes or []) == 1
        if output.causes[0].suggested_fixes:
            assert output.causes[0].suggested_fixes[0].id == 0
            assert output.causes[0].suggested_fixes[0].title == "Test Fix"
            assert output.causes[0].suggested_fixes[0].description == "Test Fix Description"
            if output.causes[0].suggested_fixes[0].snippet:
                assert output.causes[0].suggested_fixes[0].snippet.file_path == "test.py"
                assert (
                    output.causes[0].suggested_fixes[0].snippet.snippet == "def test():\n    pass"
                )
            assert output.causes[0].suggested_fixes[0].elegance == 0.7
