import textwrap
from xml.etree import ElementTree as ET

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
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
from seer.automation.autofix.utils import escape_multi_xml, extract_xml_element_text
from seer.automation.models import EventDetails
from seer.automation.pipeline import PIPELINE_SYNC_SIGNAL


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
def score_fix_single_it(dataset_item: DatasetItemClient, predicted_diff: str) -> float:
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
    response, usage = GptClient(model="gpt-4-0125-preview").completion(
        messages=[Message(role="user", content=prompt)]
    )
    if not response.content:
        return 0

    tree = ET.fromstring(f"<root>{escape_multi_xml(response.content, ['score'])}</root>")
    score_str = extract_xml_element_text(tree, "score")
    score = float(score_str) if score_str else 0

    return score


@observe(name="Score one")
def score_one(dataset_item: DatasetItemClient, predicted_diff_str: str, n_panel=3) -> float:
    return round(
        sum([score_fix_single_it(dataset_item, predicted_diff_str) for _ in range(n_panel)])
        / n_panel,
        2,
    )
