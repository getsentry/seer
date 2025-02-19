import logging
import textwrap
from typing import Literal, TypedDict, cast

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe
from pydantic_xml import attr, element

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.root_cause.models import (
    RelevantCodeFile,
    RootCauseAnalysisItem,
    TimelineEvent,
)
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixRequest,
    AutofixRequestOptions,
    AutofixRootCauseUpdatePayload,
    AutofixSolutionUpdatePayload,
    AutofixUpdateType,
    ChangesStep,
)
from seer.automation.autofix.models import RootCauseStep as RootCauseStepModel
from seer.automation.autofix.runs import create_initial_autofix_run
from seer.automation.autofix.steps.coding_step import AutofixCodingStep, AutofixCodingStepRequest
from seer.automation.autofix.steps.root_cause_step import RootCauseStep, RootCauseStepRequest
from seer.automation.models import EventDetails, PromptXmlModel
from seer.automation.pipeline import PIPELINE_SYNC_SIGNAL
from seer.automation.utils import extract_text_inside_tags

logger = logging.getLogger(__name__)


class CodeDiff(PromptXmlModel):
    file_path: str = attr()
    code_diff: str


class RootCauseExpectedOutput(PromptXmlModel, tag="expected_solution"):
    root_cause: str = element()
    solution_summary: str = element()
    diff: CodeDiff


class AutofixRequestDict(TypedDict):
    request: dict  # Contains AutofixRequest data
    commit_sha: str


class SolutionDiffDict(TypedDict):
    description: str
    unified_diff: str


class ExpectedOutputDict(TypedDict):
    original_diff: str
    solution_diff: SolutionDiffDict
    root_cause: str


class DatasetItemMetadataDict(TypedDict):
    difficulty_level: Literal["easy", "medium", "hard"]
    solvable: bool


class DatasetItemDict(TypedDict):
    input: AutofixRequestDict
    metadata: DatasetItemMetadataDict
    expected_output: ExpectedOutputDict


@observe(name="Sync run root cause evaluation on item")
def sync_run_root_cause(item: DatasetItemClient):
    run_id = None

    input_data: AutofixRequestDict = item.input
    request = AutofixRequest.model_validate(input_data["request"])

    request.options = AutofixRequestOptions(
        disable_codebase_indexing=True, disable_interactivity=True
    )

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
    request.options = AutofixRequestOptions(
        disable_codebase_indexing=True, disable_interactivity=True
    )

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
                root_cause_reproduction=[
                    TimelineEvent(
                        title=expected_output.solution_summary,
                        code_snippet_and_analysis="",
                        timeline_item_type="code",
                        relevant_code_file=RelevantCodeFile(
                            file_path=expected_output.diff.file_path,
                            repo_name="",
                        ),
                        is_most_important_event=False,
                    )
                ],
            )
        ]

        run_id = cur.run_id

    event_manager.set_selected_root_cause(
        AutofixRootCauseUpdatePayload(
            type=AutofixUpdateType.SELECT_ROOT_CAUSE,
            cause_id=-1,
        )
    )
    event_manager.set_selected_solution(
        AutofixSolutionUpdatePayload(
            type=AutofixUpdateType.SELECT_SOLUTION,
            custom_solution=None,
            solution_selected=True,
        )
    )

    AutofixCodingStep.get_signature(AutofixCodingStepRequest(run_id=run_id)).apply()

    state_after_execution = state.get()
    changes_step = state_after_execution.steps[-1]
    if not isinstance(changes_step, ChangesStep):
        raise ValueError("Expected changes step")

    changes = changes_step.changes

    if not changes:
        raise ValueError("No changes found, expected changes")

    diffs: list[str] = []
    for change in changes:
        if change.diff_str:
            diffs.append(change.diff_str)

    return "\n".join(diffs)


@observe(name="Sync run evaluation on item")
def sync_run_evaluation_on_item(item: DatasetItemClient):
    run_id = None

    input_data: AutofixRequestDict = item.input
    request = AutofixRequest.model_validate(input_data["request"])

    request.options = AutofixRequestOptions(
        disable_codebase_indexing=True, disable_interactivity=True
    )

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
    root_cause_step = state_after_root_cause.find_step(key="root_cause_analysis")

    if not isinstance(root_cause_step, RootCauseStepModel) or not root_cause_step.causes:
        return None, None

    event_manager = AutofixEventManager(state)
    event_manager.set_selected_solution(
        AutofixSolutionUpdatePayload(
            type=AutofixUpdateType.SELECT_SOLUTION,
            custom_solution=None,
            solution_selected=True,
        )
    )

    try:
        AutofixCodingStep.get_signature(AutofixCodingStepRequest(run_id=run_id)).apply()

        state_after_execution = state.get()
        changes_step = state_after_execution.steps[-1]
        if not isinstance(changes_step, ChangesStep):
            return None, root_cause_step.causes

        changes = changes_step.changes

        if not changes:
            return None, root_cause_step.causes

        diffs: list[str] = []
        for change in changes:
            if change.diff_str:
                diffs.append(change.diff_str)

        return "\n".join(diffs), root_cause_step.causes
    except Exception as e:
        logger.exception(f"Error running evaluation: {e}")
        # Return the root cause step causes anyway to score.
        return None, root_cause_step.causes


@observe(name="Score fix")
def score_fix_single_it(
    dataset_item: DatasetItemClient, predicted_diff: str, model: str
) -> tuple[float, bool]:
    if not dataset_item.expected_output:
        raise ValueError("Expected output is missing from dataset item")

    input_data: AutofixRequestDict = dataset_item.input
    expected_output: ExpectedOutputDict = dataset_item.expected_output
    request = AutofixRequest.model_validate(input_data["request"])

    event_details = EventDetails.from_event(request.issue.events[0])

    prompt = textwrap.dedent(
        """\
            <goal>
            Score how well the predicted solution fixes the issue, given the known solution, with a float score from 0 to 1, where 1 means the solution fully fixes the issue and 0 means the solution does not fix the issue at all.
            </goal>

            <reasoning_rules>
            When thinking/reasoning about the score, consider the following:
            1. Use the known solution to understand what the issue is.
            2. Consider that there are multiple ways to fix an issue.
            3. Judge whether the predicted solution actually fixes the issue.
            </reasoning_rules>

            <output_format>
            1. Provide your reasoning of your judgement of the predicted solution inside a <reasoning> tag.
            2. Provide the score inside a <score> tag.
            3. Given all that you know, now provide your verdict of whether the predicted solution actually fixes the issue with a boolean value inside a <verdict> tag, such as <verdict>True</verdict> or <verdict>False</verdict>.
            </output_format>

            Given the below issue:

            <issue>
            {event_details}
            </issue>

            We have a known correct solution:

            <known_solution>
            <description>
            {expected_description}
            </description>
            <diff>
            {expected_diff}
            </diff>
            </known_solution>

            The model outputted the following solution:

            <predicted_solution>
            {predicted_diff}
            </predicted_solution>"""
    ).format(
        event_details=event_details.format_event(),
        expected_description=expected_output["solution_diff"]["description"],
        expected_diff=expected_output["solution_diff"]["unified_diff"],
        predicted_diff=predicted_diff,
    )
    response = LlmClient().generate_text(
        model=OpenAiProvider.model(model),
        prompt=prompt,
    )

    if not response.message.content:
        raise ValueError("No response content")

    score_str = extract_text_inside_tags(response.message.content, "score")
    score = float(score_str) if score_str else 0

    verdict = extract_text_inside_tags(response.message.content, "verdict")
    verdict_bool = (verdict or "False").lower() == "true"

    return score, verdict_bool


@observe(name="Score root cause iteration")
def score_root_cause_single_it(
    dataset_item: DatasetItemClient, causes: list[RootCauseAnalysisItem], model: str
) -> tuple[float, bool, bool]:
    if not dataset_item.expected_output:
        raise ValueError("Expected output is missing from dataset item")

    input_data: AutofixRequestDict = dataset_item.input
    expected_output: ExpectedOutputDict = dataset_item.expected_output
    root_cause_expected_str = expected_output.get("root_cause")
    cause_xml = RootCausePlanTaskPromptXml.from_root_cause(causes[0])

    request = AutofixRequest.model_validate(input_data["request"])

    event_details = EventDetails.from_event(request.issue.events[0])

    prompt = textwrap.dedent(
        """\
            {event_details}

            Given the above issue, we know the correct root cause of the issue is:

            <true_root_cause>
            {expected_output}

            The solution to this root cause is:
            {expected_solution_description}
            {expected_solution_diff}
            </true_root_cause>

            We have an AI model say that the root cause of the issue is:

            <predicted_root_cause>
            {predicted_solution}
            </predicted_root_cause>

            Score how well the AI model's predicted root cause aligns with the true known root cause with a float score from 0 to 1, where 1 means the predicted root cause is the correct root cause and 0 means the predicted root cause is completely incorrect.

            Provide your reasoning of why you gave this score inside a <reasoning> tag.

            Place the score inside a <score> tag, such as <score>0.5</score>.
            Also, return your verdict of whether the predicted root cause accurately represents the true root cause of the issue with a boolean value inside a <verdict> tag, such as <verdict>True</verdict> or <verdict>False</verdict>.
            You should also grade whether the model's predicted root cause would be helpful to the developer in fixing the issue with a boolean value inside a <helpful> tag, such as <helpful>True</helpful> or <helpful>False</helpful>."""
    ).format(
        event_details=event_details.format_event(),
        expected_output=root_cause_expected_str,
        expected_solution_description=expected_output["solution_diff"]["description"],
        expected_solution_diff=expected_output["solution_diff"]["unified_diff"],
        predicted_solution=cause_xml.to_prompt_str(),
    )
    response = LlmClient().generate_text(
        model=OpenAiProvider.model(model),
        prompt=prompt,
    )
    if not response.message.content:
        raise ValueError("No response content")

    score_str = extract_text_inside_tags(response.message.content, "score")
    score = float(score_str) if score_str else 0

    verdict_str = extract_text_inside_tags(response.message.content, "verdict")
    verdict_bool = (verdict_str or "False").lower() == "true"

    helpful_str = extract_text_inside_tags(response.message.content, "helpful")
    helpful_bool = (helpful_str or "False").lower() == "true"

    return score, verdict_bool, helpful_bool


@observe(name="Score one")
def score_one(
    dataset_item: DatasetItemClient, predicted_diff_str: str, n_panel: int, model: str
) -> tuple[float, bool]:
    results = [score_fix_single_it(dataset_item, predicted_diff_str, model) for _ in range(n_panel)]

    mean_score = round(sum([result[0] for result in results]) / n_panel, 2)

    # If at least half of the panel says the fix is correct, then the fix is correct.
    verdict = sum(1 for result in results if result[1]) >= len(results) / 2

    return mean_score, verdict


@observe(name="Score root cause")
def score_root_causes(
    dataset_item: DatasetItemClient, causes: list[RootCauseAnalysisItem], n_panel: int, model: str
) -> tuple[float, bool, bool]:
    results = [score_root_cause_single_it(dataset_item, causes, model) for _ in range(n_panel)]

    mean_score = round(sum([result[0] for result in results]) / len(results), 2)

    # If at least half of the panel says the fix is correct, then the fix is correct.
    verdict = sum(1 for result in results if result[1]) >= len(results) / 2

    helpful = sum(1 for result in results if result[2]) >= len(results) / 2

    return mean_score, verdict, helpful


def make_score_name(model: str, n_panel: int, name: str) -> str:
    return f"{name}_{model}_{n_panel}x_panel"
