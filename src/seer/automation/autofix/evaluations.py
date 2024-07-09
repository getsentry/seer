import textwrap
from typing import TypedDict, cast
from xml.etree import ElementTree as ET

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe
from pydantic_xml import attr, element

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseAnalysisItemPromptXml,
    RootCauseSuggestedFix,
    RootCauseSuggestedFixSnippet,
)
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixRequest,
    AutofixRequestOptions,
    AutofixRootCauseUpdatePayload,
    AutofixUpdateType,
    ChangesStep,
)
from seer.automation.autofix.models import RootCauseStep as RootCauseStepModel
from seer.automation.autofix.runs import create_initial_autofix_run
from seer.automation.autofix.steps.planning_chain import (
    AutofixPlanningStep,
    AutofixPlanningStepRequest,
)
from seer.automation.autofix.steps.root_cause_step import RootCauseStep, RootCauseStepRequest
from seer.automation.models import EventDetails, PromptXmlModel
from seer.automation.pipeline import PIPELINE_SYNC_SIGNAL
from seer.automation.utils import (
    escape_multi_xml,
    extract_text_inside_tags,
    extract_xml_element_text,
)


class CodeDiff(PromptXmlModel):
    file_path: str = attr()
    code_diff: str


class RootCauseExpectedOutput(PromptXmlModel, tag="expected_solution"):
    root_cause: str = element()
    solution_summary: str = element()
    diff: CodeDiff


@observe(name="Sync run root cause evaluation on item")
def sync_run_root_cause(item: DatasetItemClient):
    run_id = None

    request = AutofixRequest.model_validate(item.input.get("request"))

    request.options = AutofixRequestOptions(disable_codebase_indexing=True)

    state = create_initial_autofix_run(request)
    with state.update() as cur:
        cur.signals.append(PIPELINE_SYNC_SIGNAL)

    run_id = state.get().run_id

    RootCauseStep.get_signature(
        RootCauseStepRequest(
            run_id=run_id,
        )
    ).apply()

    state_after_root_cause = state.get()
    root_cause_step = state_after_root_cause.steps[-1]

    if not isinstance(root_cause_step, RootCauseStepModel) or not root_cause_step.causes:
        raise ValueError("Expected root cause step")

    return root_cause_step.causes


@observe(name="Sync run execution evaluation on item")
def sync_run_execution(item: DatasetItemClient):
    request = AutofixRequest.model_validate(item.input.get("request"))
    request.options = AutofixRequestOptions(disable_codebase_indexing=True)

    expected_output = RootCauseExpectedOutput.model_validate(
        {
            "diff": item.input.get("snippet_diff"),
            "root_cause": item.input.get("root_cause"),
            "solution_summary": item.input.get("solution_summary"),
        }
    )

    state = create_initial_autofix_run(request)

    event_manager = AutofixEventManager(state)
    with state.update() as cur:
        cur.signals.append(PIPELINE_SYNC_SIGNAL)

        root_cause_step = cast(
            RootCauseStepModel, cur.find_or_add(event_manager.root_cause_analysis_step)
        )
        root_cause_step.causes = [
            RootCauseAnalysisItem(
                title=expected_output.root_cause,
                description="",
                likelihood=1.0,
                actionability=1.0,
                suggested_fixes=[
                    RootCauseSuggestedFix(
                        title=expected_output.solution_summary,
                        description="",
                        snippet=RootCauseSuggestedFixSnippet(
                            file_path=expected_output.diff.file_path,
                            snippet=expected_output.diff.code_diff,
                        ),
                        elegance=1.0,
                    )
                ],
            )
        ]

        run_id = cur.run_id

    event_manager.set_selected_root_cause(
        AutofixRootCauseUpdatePayload(
            type=AutofixUpdateType.SELECT_ROOT_CAUSE,
            cause_id=-1,
            fix_id=-1,
        )
    )

    AutofixPlanningStep.get_signature(AutofixPlanningStepRequest(run_id=run_id)).apply()

    state_after_execution = state.get()
    changes_step = state_after_execution.steps[-1]
    if not isinstance(changes_step, ChangesStep):
        raise ValueError("Expected changes step")

    changes = changes_step.changes

    if not changes:
        return None

    diffs: list[str] = []
    for change in changes:
        if change.diff_str:
            diffs.append(change.diff_str)

    return "\n".join(diffs)


@observe(name="Sync run evaluation on item")
def sync_run_evaluation_on_item(item: DatasetItemClient):
    run_id = None

    request = AutofixRequest.model_validate(item.input.get("request"))

    request.options = AutofixRequestOptions(disable_codebase_indexing=True)

    state = create_initial_autofix_run(request)
    with state.update() as cur:
        cur.signals.append(PIPELINE_SYNC_SIGNAL)

    run_id = state.get().run_id

    RootCauseStep.get_signature(
        RootCauseStepRequest(
            run_id=run_id,
        )
    ).apply()

    state_after_root_cause = state.get()
    root_cause_step = state_after_root_cause.steps[-1]

    if not isinstance(root_cause_step, RootCauseStepModel):
        raise ValueError("Expected root cause step")

    if not root_cause_step.causes:
        return None

    cause = root_cause_step.causes[0]
    cause_id = cause.id

    if not cause.suggested_fixes:
        return None

    fix_id = cause.suggested_fixes[0].id

    event_manager = AutofixEventManager(state)
    event_manager.set_selected_root_cause(
        AutofixRootCauseUpdatePayload(
            type=AutofixUpdateType.SELECT_ROOT_CAUSE,
            cause_id=cause_id,
            fix_id=fix_id,
        )
    )

    AutofixPlanningStep.get_signature(AutofixPlanningStepRequest(run_id=run_id)).apply()

    state_after_execution = state.get()
    changes_step = state_after_execution.steps[-1]
    if not isinstance(changes_step, ChangesStep):
        raise ValueError("Expected changes step")

    changes = changes_step.changes

    if not changes:
        return None

    diffs: list[str] = []
    for change in changes:
        if change.diff_str:
            diffs.append(change.diff_str)

    return "\n".join(diffs)


@observe(name="Score fix")
def score_fix_single_it(dataset_item: DatasetItemClient, predicted_diff: str, model: str) -> float:
    if not dataset_item.expected_output:
        raise ValueError("Expected output is missing from dataset item")

    request = AutofixRequest.model_validate(dataset_item.input.get("request"))

    event_details = EventDetails.from_event(request.issue.events[0])

    prompt = textwrap.dedent(
        """\
            {event_details}

            Given the above issue, we know the correct fix is:

            <expected_solution>
            <diff>
            {expected_diff}
            </diff>
            </expected_solution>

            The model outputted the following solution:

            <predicted_solution>
            {predicted_diff}
            </predicted_solution>

            Score how well the predicted solution matches the expected solution with a float score from 0 to 1, where 1 means the solution fully fixes the issue and 0 means the solution does not fix the issue at all.
            - Consider the context of the issue and the diff
            - Consider that there are multiple ways to fix an issue

            Think step-by-step inside a <thoughts> tag before giving a score.
            Return the score inside a <score> tag."""
    ).format(
        event_details=event_details.format_event(),
        expected_diff=dataset_item.expected_output.get("diff"),
        predicted_diff=predicted_diff,
    )
    response, usage = GptClient().completion(
        messages=[Message(role="user", content=prompt)], model=model
    )
    if not response.content:
        return 0

    tree = ET.fromstring(f"<root>{escape_multi_xml(response.content, ['score'])}</root>")
    score_str = extract_xml_element_text(tree, "score")
    score = float(score_str) if score_str else 0

    return score


RootCauseScoreResult = TypedDict(
    "RootCauseScoreResult",
    {
        "highest_score": float,
        "position_score": float,
        "mean_score": float,
    },
)


@observe(name="Score root cause iteration")
def score_root_cause_single_it(
    dataset_item: DatasetItemClient, causes: list[RootCauseAnalysisItem], model: str
) -> list[float] | None:
    if not dataset_item.expected_output:
        raise ValueError("Expected output is missing from dataset item")

    expected_output = RootCauseExpectedOutput.model_validate(dataset_item.expected_output)
    causes_xml = [RootCauseAnalysisItemPromptXml.from_model(cause) for cause in causes]

    solution_strs: list[str] = []
    for i, cause in enumerate(causes_xml):
        num = i + 1
        solution_strs.append(f"<solution_{num}>{cause.to_prompt_str()}</solution_{num}>")
    solutions_str = "\n".join(solution_strs)

    request = AutofixRequest.model_validate(dataset_item.input.get("request"))

    event_details = EventDetails.from_event(request.issue.events[0])

    prompt = textwrap.dedent(
        """\
            {event_details}

            Given the above issue, we know the correct analysis of the issue is:

            {expected_output}

            The model outputted the following possible root causes and solutions:

            <predicted_solutions>
            {predicted_solutions}
            </predicted_solutions>

            Score how well the predicted root cause and solution matches the expected root cause and solution with a float score from 0 to 1, where 1 means the solution fully fixes the issue and 0 means the solution does not fix the issue at all.
            - The model will return multiple predicted root causes and solutions, ordered from most likely to least likely.

            Think step-by-step inside a <thoughts> tag before giving scores.
            Score each solution inside a <score_{{n}}> tag, such as <score_1>0.5</score_1>, where n is the number of the solution."""
    ).format(
        event_details=event_details.format_event(),
        expected_output=expected_output.to_prompt_str(),
        predicted_solutions=solutions_str,
    )
    response, usage = GptClient().completion(
        model=model, messages=[Message(role="user", content=prompt)]
    )
    if not response.content:
        raise ValueError("No response content")

    scores: list[float] = []
    for i in range(len(causes_xml)):
        score_str = extract_text_inside_tags(response.content, f"score_{i + 1}")
        score = float(score_str) if score_str else 0
        scores.append(score)

    return scores


@observe(name="Score one")
def score_one(
    dataset_item: DatasetItemClient, predicted_diff_str: str, n_panel: int, model: str
) -> float:
    return round(
        sum([score_fix_single_it(dataset_item, predicted_diff_str, model) for _ in range(n_panel)])
        / n_panel,
        2,
    )


@observe(name="Score root cause")
def score_root_causes(
    dataset_item: DatasetItemClient, causes: list[RootCauseAnalysisItem], n_panel: int, model: str
) -> RootCauseScoreResult:
    all_results: list[list[float]] = []
    i = 0
    while i < n_panel:
        result = score_root_cause_single_it(dataset_item, causes, model)
        if result is None or len(result) != len(causes):
            continue
        else:
            all_results.append(result)
            i += 1

    mean_scores = [round(sum(scores) / len(scores), 2) for scores in zip(*all_results)]

    highest_score = max(mean_scores)
    mean_score = sum(mean_scores) / len(mean_scores)

    # Position score: 1.0 if the highest score is first, reduce the points the lower the score is, if it is at the last position, give 0.0
    position_score = 1.0 - (mean_scores.index(highest_score) / len(mean_scores))

    return {
        "highest_score": highest_score,
        "position_score": position_score,
        "mean_score": mean_score,
    }


def make_score_name(model: str, n_panel: int, name: str) -> str:
    return f"{name}_{model}_{n_panel}x_panel"
