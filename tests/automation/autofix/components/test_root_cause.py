import unittest
from unittest.mock import MagicMock, patch

from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseSuggestedFix,
    RootCauseSuggestedFixSnippet,
)


class TestRootCauseComponent(unittest.TestCase):
    @patch("seer.automation.autofix.components.root_cause.component.GptAgent")
    def test_root_cause_simple_response_parsing(self, mock_GptAgent):
        mock_GptAgent.return_value.run.return_value = """<thoughts>
            a thought
            </thoughts>

            <potential_root_causes>
            <potential_cause likelihood="0.9" actionability="1.0">
            <title>
            Missing Null Check
            </title>
            <description>
            The root cause of the issue is ...
            </description>
            </potential_cause>
            </potential_root_causes>
            """

        component = RootCauseAnalysisComponent(MagicMock())
        output = component.invoke(MagicMock())

        assert output is not None
        if output:
            assert output.causes == [
                RootCauseAnalysisItem(
                    id=0,
                    title="Missing Null Check",
                    description="The root cause of the issue is ...",
                    likelihood=0.9,
                    actionability=1.0,
                    suggested_fixes=None,
                )
            ]

    @patch("seer.automation.autofix.components.root_cause.component.GptAgent")
    def test_root_cause_suggested_fix_response_parsing(self, mock_GptAgent):
        mock_GptAgent.return_value.run.return_value = """<thoughts>
            a thought
            </thoughts>

            <potential_root_causes>
            <potential_cause likelihood="0.9" actionability="1.0">
            <title>
            Missing Null Check
            </title>
            <description>
            The root cause of the issue is ...
            </description>
            <suggested_fixes>
            <suggested_fix elegance="0.9">
            <title>
            Add Null Check for ...
            </title>
            <description>
            This fix involves adding ...
            </description>
            <snippet file_path="some/app/path.py">
            def foo():
            </snippet>
            </suggested_fix>
            </suggested_fixes>
            </potential_cause>
            </potential_root_causes>
            """

        component = RootCauseAnalysisComponent(MagicMock())
        output = component.invoke(MagicMock())

        assert output is not None
        if output:
            assert output.causes == [
                RootCauseAnalysisItem(
                    id=0,
                    title="Missing Null Check",
                    description="The root cause of the issue is ...",
                    likelihood=0.9,
                    actionability=1.0,
                    suggested_fixes=[
                        RootCauseSuggestedFix(
                            id=0,
                            title="Add Null Check for ...",
                            description="This fix involves adding ...",
                            snippet=RootCauseSuggestedFixSnippet(
                                file_path="some/app/path.py", snippet="def foo():"
                            ),
                            elegance=0.9,
                        )
                    ],
                )
            ]

    @patch("seer.automation.autofix.components.root_cause.component.GptAgent")
    def test_root_cause_multiple_simple_response_parsing(self, mock_GptAgent):
        mock_GptAgent.return_value.run.return_value = """<thoughts>
            a thought
            </thoughts>

            <potential_root_causes>
            <potential_cause likelihood="0.9" actionability="1.0">
            <title>
            Missing Null Check
            </title>
            <description>
            The root cause of the issue is ...
            </description>
            </potential_cause>
            <potential_cause likelihood="0.5" actionability="0.5">
            <title>
            Missing Null Check again
            </title>
            <description>
            Blah
            </description>
            </potential_cause>
            </potential_root_causes>
            """

        component = RootCauseAnalysisComponent(MagicMock())
        output = component.invoke(MagicMock())

        assert output is not None
        if output:
            assert output.causes == [
                RootCauseAnalysisItem(
                    id=0,
                    title="Missing Null Check",
                    description="The root cause of the issue is ...",
                    likelihood=0.9,
                    actionability=1.0,
                    suggested_fixes=None,
                ),
                RootCauseAnalysisItem(
                    id=1,
                    title="Missing Null Check again",
                    description="Blah",
                    likelihood=0.5,
                    actionability=0.5,
                    suggested_fixes=None,
                ),
            ]

    @patch("seer.automation.autofix.components.root_cause.component.GptAgent")
    def test_root_cause_empty_response_parsing(self, mock_GptAgent):
        mock_GptAgent.return_value.run.return_value = (
            "<potential_root_causes></potential_root_causes>"
        )

        component = RootCauseAnalysisComponent(MagicMock())
        output = component.invoke(MagicMock())

        assert output is not None
        if output:
            assert output.causes == []
