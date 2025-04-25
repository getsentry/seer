from pydantic import BaseModel

from seer.automation.codebase.models import PrFile
from seer.automation.codegen.models import CodegenRelevantWarningsRequest, StaticAnalysisSuggestion
from seer.automation.models import IssueDetails


class EvalItem(BaseModel):
    """
    An item in the evaluation dataset.
    """

    request: CodegenRelevantWarningsRequest
    """
    The request for the evaluation, as it would be sent to Seer by Overwatch.
    """
    suggestions: list[StaticAnalysisSuggestion]
    """
    The expected suggestions for the given PR.
    """
    pr_files: list[PrFile]
    """
    The PR files for the given request.
    This is saved so we don't have to actually reach out to GitHub to get the PR files while running the evaluation.
    """
    issues: list[IssueDetails]
    """
    The issues for the given request.
    This is saved so we don't have to actually reach out to Sentry to get the issues while running the evaluation.
    """


class CodegenRelevantWarningsEvaluationRequest(BaseModel):
    dataset_name: str
    run_name: str
    run_description: str | None = None
    test: bool = False
    run_on_item_id: str | None = None
    random_for_test: bool = False
    n_runs_per_item: int = 1
