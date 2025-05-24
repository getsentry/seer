import logging
import textwrap
from typing import Literal, TypedDict

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe
from pydantic import BaseModel
from pydantic_xml import attr, element

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.autofix.components.coding.models import RootCausePlanTaskPromptXml
from seer.automation.autofix.components.solution.models import SolutionTimelineEvent
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixRequestOptions,
    AutofixSolutionUpdatePayload,
    AutofixUpdateType,
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


@observe(name="Sync run evaluation on item")
def sync_run_evaluation_on_item(item: DatasetItemClient) -> AutofixContinuation:
    run_id = None

    input_data: AutofixRequestDict = item.input
    request = AutofixRequest.model_validate(input_data["request"])

    request.options = AutofixRequestOptions(
        disable_codebase_indexing=True, disable_interactivity=True, force_use_repos=True
    )

    state = create_initial_autofix_run(request)
    with state.update() as cur:
        cur.signals.append(PIPELINE_SYNC_SIGNAL)

    run_id = state.get().run_id

    try:
        RootCauseStep.get_signature(
            RootCauseStepRequest(
                run_id=run_id,
            )
        ).apply()

        state_after_root_cause = state.get()
        root_cause_step = state_after_root_cause.find_step(key="root_cause_analysis")

        if not isinstance(root_cause_step, RootCauseStepModel) or not root_cause_step.causes:
            return state.get()

        event_manager = AutofixEventManager(state)
        event_manager.set_selected_solution(
            AutofixSolutionUpdatePayload(
                type=AutofixUpdateType.SELECT_SOLUTION,
                custom_solution=None,
                solution_selected=True,
            )
        )

        AutofixCodingStep.get_signature(AutofixCodingStepRequest(run_id=run_id)).apply()

        return state.get()
    except Exception as e:
        logger.exception(f"Error running evaluation: {e}")
        # Return the state anyway to score.
        return state.get()


@observe(name="Score solution iteration")
def score_solution_single_it(
    dataset_item: DatasetItemClient, final_state: AutofixContinuation, model: str
) -> tuple[float, bool] | None:
    if not dataset_item.expected_output:
        raise ValueError("Expected output is missing from dataset item")

    input_data: AutofixRequestDict = dataset_item.input
    expected_output: ExpectedOutputDict = dataset_item.expected_output
    request = AutofixRequest.model_validate(input_data["request"])

    event_details = EventDetails.from_event(request.issue.events[0])

    predicted_solution = final_state.solution_step

    if not predicted_solution:
        return None

    class SolutionPrinting(BaseModel):
        solution: list[SolutionTimelineEvent]

    predicted_solution_str = SolutionPrinting(solution=predicted_solution.solution).model_dump_json(
        indent=2
    )

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
            {predicted_solution_str}
            </predicted_solution>"""
    ).format(
        event_details=event_details.format_event(),
        expected_description=expected_output["solution_diff"]["description"],
        expected_diff=expected_output["solution_diff"]["unified_diff"],
        predicted_solution_str=predicted_solution_str,
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


@observe(name="Score coding")
def score_coding_single_it(
    dataset_item: DatasetItemClient, final_state: AutofixContinuation, model: str
) -> tuple[float, float] | None:

    predicted_solution = final_state.solution_step

    if not predicted_solution:
        return None

    class SolutionPrinting(BaseModel):
        solution: list[SolutionTimelineEvent]

    predicted_solution_str = SolutionPrinting(solution=predicted_solution.solution).model_dump_json(
        indent=2
    )

    changes_step = final_state.changes_step
    if not changes_step or not changes_step.changes:
        return None

    final_diff_str = changes_step.changes[0].diff_str

    prompt = textwrap.dedent(
        """\
            You are an expert at judging code changes and code quality.

            <goal>
            Score the code from the outputted diff given the instructions.
            </goal>

            <output_format>
            1. Provide your reasoning of your judgement of the predicted solution inside a <reasoning> tag.
            2. Provide a float score from 0 to 1 of whether the diff is complete and correct, whether it includes the instructions, inside a <correctness_score> tag. For example, if the diff is missing a step, then the score should be low. If the diff doesn't import a necessary module, then the score should be low. A high score means the diff contains all the necessary code changes and is correct (imports required modules, does not remove any necessary code, etc.).
            3. Provide a float score from 0 to 1 of how targeted and concise the diff is, inside a <conciseness_score> tag. For example, if the diff includes extra changes that were not asked for in the instructions, then the conciseness score should be low.
            </output_format>

            Given the below instructions:

            <instructions>
            {predicted_solution_str}
            </instructions>

            The AI model outputted the following diff:

            <predicted_diff>
            {final_diff_str}
            </predicted_diff>"""
    ).format(predicted_solution_str=predicted_solution_str, final_diff_str=final_diff_str)
    response = LlmClient().generate_text(
        model=OpenAiProvider.model(model),
        prompt=prompt,
    )

    if not response.message.content:
        raise ValueError("No response content")

    correctness_score_str = extract_text_inside_tags(response.message.content, "correctness_score")
    correctness_score = float(correctness_score_str) if correctness_score_str else 0

    conciseness_score_str = extract_text_inside_tags(response.message.content, "conciseness_score")
    conciseness_score = float(conciseness_score_str) if conciseness_score_str else 0

    return correctness_score, conciseness_score


@observe(name="Score root cause iteration")
def score_root_cause_single_it(
    dataset_item: DatasetItemClient, final_state: AutofixContinuation, model: str
) -> tuple[float, bool, bool] | None:
    if not dataset_item.expected_output:
        raise ValueError("Expected output is missing from dataset item")

    input_data: AutofixRequestDict = dataset_item.input
    expected_output: ExpectedOutputDict = dataset_item.expected_output
    root_cause_expected_str = expected_output.get("root_cause")

    if not final_state.root_cause_step or not final_state.root_cause_step.causes:
        return None

    cause_xml = RootCausePlanTaskPromptXml.from_root_cause(final_state.root_cause_step.causes[0])

    request = AutofixRequest.model_validate(input_data["request"])

    event_details = EventDetails.from_event(
        event=request.issue.events[0], issue_title=request.issue.title
    )

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


@observe(name="Score solution")
def score_solution(
    dataset_item: DatasetItemClient, final_state: AutofixContinuation, n_panel: int, model: str
) -> tuple[float, bool] | None:
    results = [score_solution_single_it(dataset_item, final_state, model) for _ in range(n_panel)]

    if any(result is None for result in results):
        return None

    results = [result for result in results if result is not None]

    mean_score = round(sum([result[0] for result in results]) / n_panel, 2)

    # If at least half of the panel says the fix is correct, then the fix is correct.
    verdict = sum(1 for result in results if result[1]) >= len(results) / 2

    return mean_score, verdict


@observe(name="Score coding")
def score_coding(
    dataset_item: DatasetItemClient, final_state: AutofixContinuation, n_panel: int, model: str
) -> tuple[float, float] | None:
    results = [score_coding_single_it(dataset_item, final_state, model) for _ in range(n_panel)]

    if any(result is None for result in results):
        return None

    results = [result for result in results if result is not None]

    mean_correctness_score = round(sum([result[0] for result in results]) / n_panel, 2)
    mean_conciseness_score = round(sum([result[1] for result in results]) / n_panel, 2)

    return mean_correctness_score, mean_conciseness_score


@observe(name="Score root cause")
def score_root_causes(
    dataset_item: DatasetItemClient, final_state: AutofixContinuation, n_panel: int, model: str
) -> tuple[float, bool, bool] | None:
    results = [score_root_cause_single_it(dataset_item, final_state, model) for _ in range(n_panel)]

    if any(result is None for result in results):
        return None

    results = [result for result in results if result is not None]

    mean_score = round(sum([result[0] for result in results]) / len(results), 2)

    # If at least half of the panel says the fix is correct, then the fix is correct.
    verdict = sum(1 for result in results if result[1]) >= len(results) / 2

    helpful = sum(1 for result in results if result[2]) >= len(results) / 2

    return mean_score, verdict, helpful


def make_score_name(model: str, n_panel: int, name: str) -> str:
    return f"{name}_{model}_{n_panel}x_panel"
