from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseRelevantCodeSnippet,
    RootCauseRelevantContext,
)


class TestPlannerModels:
    def test_root_cause_conversion(self):
        prompt_xml = RootCausePlanTaskPromptXml.from_root_cause(
            RootCauseAnalysisItem(
                id=1,
                title="title",
                description="description",
                likelihood=0.5,
                actionability=0.75,
                code_context=[
                    RootCauseRelevantContext(
                        id=2,
                        title="fix_title",
                        description="fix_description",
                        snippet=RootCauseRelevantCodeSnippet(
                            file_path="file_path", snippet="snippet", repo_name="owner/repo"
                        ),
                    )
                ],
            )
        )

        assert prompt_xml.title == "title"
        assert prompt_xml.description == "description"
        assert prompt_xml.contexts[0].title == "fix_title"
        assert prompt_xml.contexts[0].description == "fix_description"
        assert prompt_xml.contexts[0].snippet is not None
        if prompt_xml.contexts[0].snippet is not None:
            assert prompt_xml.contexts[0].snippet.file_path == "file_path"
            assert prompt_xml.contexts[0].snippet.snippet == "snippet"
