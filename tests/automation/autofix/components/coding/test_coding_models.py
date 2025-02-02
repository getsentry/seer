from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.root_cause.models import (
    RelevantCodeFile,
    RootCauseAnalysisItem,
    TimelineEvent,
)


class TestPlannerModels:
    def test_root_cause_conversion(self):
        prompt_xml = RootCausePlanTaskPromptXml.from_root_cause(
            RootCauseAnalysisItem(
                id=1,
                root_cause_reproduction=[
                    TimelineEvent(
                        title="fix_title",
                        code_snippet_and_analysis="fix_description",
                        timeline_item_type="code",
                        relevant_code_file=RelevantCodeFile(
                            file_path="file_path",
                            repo_name="owner/repo",
                        ),
                        is_most_important_event=True,
                    )
                ],
            )
        )

        assert prompt_xml.contexts[0].title == "fix_title"
        assert prompt_xml.contexts[0].description == "fix_description"
        assert prompt_xml.contexts[0].relevant_code_file_path == "file_path"
        assert prompt_xml.contexts[0].relevant_code_repo_name == "owner/repo"
