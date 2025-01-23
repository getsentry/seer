import datetime
import logging
import random
from typing import Literal, Type, cast

import sentry_sdk
from langfuse import Langfuse

from celery_app.app import celery_app
from seer.automation.agent.models import Message
from seer.automation.agent.utils import parse_json_with_keys
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.comment_thread import (
    CommentThreadComponent,
    CommentThreadRequest,
)
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.evaluations import (
    make_score_name,
    score_one,
    score_root_causes,
    sync_run_evaluation_on_item,
    sync_run_execution,
    sync_run_root_cause,
)
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixCommentThreadPayload,
    AutofixCreateBranchUpdatePayload,
    AutofixCreatePrUpdatePayload,
    AutofixEvaluationRequest,
    AutofixRequest,
    AutofixRestartFromPointPayload,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    AutofixUpdateCodeChangePayload,
    AutofixUpdateRequest,
    AutofixUpdateType,
    AutofixUserMessagePayload,
    ChangesStep,
    CommentThread,
    DefaultStep,
    Step,
)
from seer.automation.autofix.runs import create_initial_autofix_run
from seer.automation.autofix.state import ContinuationState
from seer.automation.autofix.steps.coding_step import AutofixCodingStep, AutofixCodingStepRequest
from seer.automation.autofix.steps.root_cause_step import RootCauseStep, RootCauseStepRequest
from seer.automation.models import InitializationError
from seer.automation.utils import process_repo_provider, raise_if_no_genai_consent
from seer.configuration import AppConfig
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session
from seer.dependency_injection import inject, injected
from seer.rpc import get_sentry_client

logger = logging.getLogger(__name__)


def get_autofix_state_from_pr_id(provider: str, pr_id: int) -> ContinuationState | None:
    with Session() as session:
        run_state = (
            session.query(DbRunState)
            .join(DbPrIdToAutofixRunIdMapping, DbPrIdToAutofixRunIdMapping.run_id == DbRunState.id)
            .filter(
                DbPrIdToAutofixRunIdMapping.provider == process_repo_provider(provider),
                DbPrIdToAutofixRunIdMapping.pr_id == pr_id,
            )
            .order_by(DbRunState.id.desc())
            .first()
        )
        if run_state is None:
            return None

        continuation = ContinuationState(run_state.id)

        return continuation


def get_autofix_state(
    *, group_id: int | None = None, run_id: int | None = None
) -> ContinuationState | None:
    with Session() as session:
        run_state: DbRunState | None = None
        if group_id is not None:
            if run_id is not None:
                raise ValueError("Either group_id or run_id must be provided, not both")
            run_state = (
                session.query(DbRunState)
                .filter(DbRunState.group_id == group_id)
                .order_by(DbRunState.id.desc())
                .first()
            )
        elif run_id is not None:
            run_state = session.query(DbRunState).filter(DbRunState.id == run_id).first()
        else:
            raise ValueError("Either group_id or run_id must be provided")

        if run_state is None:
            return None

        continuation = ContinuationState(run_state.id)

        return continuation


def get_all_autofix_runs_after(after: datetime.datetime):
    with Session() as session:
        runs = (
            session.query(DbRunState)
            .filter(DbRunState.last_triggered_at > after, DbRunState.type == "autofix")
            .all()
        )
        return [ContinuationState(run.id) for run in runs]


@celery_app.task(time_limit=15)
def check_and_mark_recent_autofix_runs():
    logger.info("Checking and marking recent autofix runs")
    after = datetime.datetime.now() - datetime.timedelta(hours=1, minutes=15)
    logger.info(f"Getting all autofix runs after {after}")
    runs = get_all_autofix_runs_after(after)
    logger.info(f"Got {len(runs)} runs")
    for run in runs:
        check_and_mark_if_timed_out(run)


def check_and_mark_if_timed_out(state: ContinuationState):
    with state.update() as cur:
        if cur.is_running and cur.has_timed_out:
            cur.mark_running_steps_errored()
            cur.status = AutofixStatus.ERROR

            # Will log to Sentry. We will have an alert set up to notify us when this happens.
            logger.error(f"Autofix run {cur.run_id} has timed out")


@inject
def run_autofix_root_cause(
    request: AutofixRequest,
    app_config: AppConfig = injected,
):
    state = create_initial_autofix_run(request)

    cur_state = state.get()

    # Process has no further work.
    if cur_state.status in AutofixStatus.terminal():
        logger.warning(f"Ignoring job, state {cur_state.status}")
        return

    RootCauseStep.get_signature(
        RootCauseStepRequest(
            run_id=cur_state.run_id,
        ),
        queue=app_config.CELERY_WORKER_QUEUE,
    ).apply_async()

    return cur_state.run_id


@inject
def run_autofix_execution(
    request: AutofixUpdateRequest,
    app_config: AppConfig = injected,
):
    state = ContinuationState(request.run_id)

    raise_if_no_genai_consent(state.get().request.organization_id)

    with state.update() as cur:
        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    event_manager.send_coding_start()

    payload = cast(AutofixRootCauseUpdatePayload, request.payload)

    try:
        event_manager.set_selected_root_cause(payload)
        cur = state.get()

        # Process has no further work.
        if cur.status in AutofixStatus.terminal():
            logger.warning(f"Ignoring job, state {cur.status}")
            return

        AutofixCodingStep.get_signature(
            AutofixCodingStepRequest(
                run_id=cur.run_id,
            ),
            queue=app_config.CELERY_WORKER_QUEUE,
        ).apply_async()
    except InitializationError as e:
        sentry_sdk.capture_exception(e)
        raise e


@inject
def run_autofix_push_changes(
    request: AutofixUpdateRequest,
    app_config: AppConfig = injected,
):
    if not isinstance(request.payload, AutofixCreatePrUpdatePayload) and not isinstance(
        request.payload, AutofixCreateBranchUpdatePayload
    ):
        raise ValueError("Invalid payload type for create_pr or create_branch")

    state = ContinuationState(request.run_id)

    raise_if_no_genai_consent(state.get().request.organization_id)

    with state.update() as cur:
        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    context = AutofixContext(state=state, event_manager=event_manager)

    context.commit_changes(
        repo_external_id=request.payload.repo_external_id,
        repo_id=request.payload.repo_id,
        make_pr=request.payload.make_pr,
    )


@inject
def restart_step_with_user_response(
    state: ContinuationState,
    memory: list[Message],
    text: str,
    event_manager: AutofixEventManager,
    step_to_restart: Step,
    step_class: Type[AutofixCodingStep | RootCauseStep],
    step_request_class: Type[AutofixCodingStepRequest | RootCauseStepRequest],
    app_config: AppConfig = injected,
):
    cur_state = state.get()
    if memory:
        tool_call_id = memory[-1].tool_call_id
        if tool_call_id:
            user_response = Message(role="tool", content=text, tool_call_id=tool_call_id)
            if memory[-1].role == "tool":
                memory[-1] = user_response
            else:
                memory.append(user_response)
            event_manager.restart_step(step_to_restart)
            step_class.get_signature(
                step_request_class(
                    run_id=cur_state.run_id,
                    initial_memory=memory,
                ),
                queue=app_config.CELERY_WORKER_QUEUE,
            ).apply_async()


def receive_user_message(request: AutofixUpdateRequest):
    if not isinstance(request.payload, AutofixUserMessagePayload):
        raise ValueError("Invalid payload type for user_message")

    state = ContinuationState(request.run_id)
    cur_state = state.get()
    step_to_restart = cur_state.find_last_step_waiting_for_response()

    # check the state to see if we're responding to a question or interjecting
    if step_to_restart:
        # user response to question
        with state.update() as cur:
            cur.mark_triggered()
        event_manager = AutofixEventManager(state)
        context = AutofixContext(state=state, event_manager=event_manager)

        is_coding_step = step_to_restart.key == "plan"
        memory = (
            context.get_memory("plan_and_code")
            if is_coding_step
            else context.get_memory("root_cause_analysis")
        )

        question = ""
        if memory and len(memory) >= 2 and memory[-2].tool_calls:
            question = parse_json_with_keys(memory[-2].tool_calls[0].args, ["question"])["question"]

        with state.update() as cur:
            if isinstance(cur.steps[-1], DefaultStep):
                cur.steps[-1].insights.append(
                    InsightSharingOutput(
                        insight=f"_{question}_  {request.payload.text}",
                        justification="USER",
                        generated_at_memory_index=len(memory) - 1 if memory else -1,
                    )
                )

        if is_coding_step:
            restart_step_with_user_response(
                state,
                memory,
                request.payload.text,
                event_manager,
                step_to_restart,
                AutofixCodingStep,
                AutofixCodingStepRequest,
            )
        else:
            restart_step_with_user_response(
                state,
                memory,
                request.payload.text,
                event_manager,
                step_to_restart,
                RootCauseStep,
                RootCauseStepRequest,
            )

    else:
        # user interjection
        with state.update() as cur:
            if cur.steps:
                cur.steps[-1].receive_user_message(request.payload.text)
                if isinstance(cur.steps[-1], DefaultStep):
                    cur.steps[-1].insights.append(
                        InsightSharingOutput(
                            insight=request.payload.text,
                            justification="USER",
                            generated_at_memory_index=-1,
                        )
                    )


def truncate_memory_to_match_insights(memory: list[Message], step: DefaultStep):
    truncated_memory = []
    for insight in step.insights[::-1]:
        if insight.generated_at_memory_index >= 0:
            new_memory = memory[: insight.generated_at_memory_index + 1]
            # include extra memory items to satisfy tool calls, or cut out the tool calls if no responses available
            if new_memory and new_memory[-1].tool_calls:
                num_tool_calls = len(new_memory[-1].tool_calls)
                if insight.generated_at_memory_index + num_tool_calls < len(memory):
                    new_memory.extend(
                        memory[
                            insight.generated_at_memory_index
                            + 1 : insight.generated_at_memory_index
                            + num_tool_calls
                            + 1
                        ]
                    )
                else:
                    new_memory = new_memory[:-1]
            truncated_memory = new_memory
            break
    if not step.insights:
        truncated_memory = memory[: step.initial_memory_length]
    return truncated_memory if truncated_memory else memory


@inject
def restart_from_point_with_feedback(
    request: AutofixUpdateRequest,
    app_config: AppConfig = injected,
):
    if not isinstance(request.payload, AutofixRestartFromPointPayload):
        raise ValueError("Invalid payload type for restart_from_point_with_feedback")

    state = ContinuationState(request.run_id)
    event_manager = AutofixEventManager(state)

    step_index = request.payload.step_index
    insight_card_index = request.payload.retain_insight_card_index

    event_manager.reset_steps_to_point(step_index, insight_card_index)

    context = AutofixContext(
        state=state, sentry_client=get_sentry_client(), event_manager=event_manager
    )
    step_to_restart = next(
        (step for step in reversed(state.get().steps) if isinstance(step, DefaultStep)), None
    )
    if not step_to_restart:
        raise ValueError("No DefaultStep found in steps")

    is_coding_step = step_to_restart.key == "plan"
    memory = (
        context.get_memory("plan_and_code")
        if is_coding_step
        else context.get_memory("root_cause_analysis")
    )
    memory = truncate_memory_to_match_insights(memory, step_to_restart)

    # add feedback to memory and to insights
    if request.payload.message:
        # enforce alternating user/assistant messages
        for item in reversed(memory):
            if item.role == "user":
                memory.append(Message(content=".", role="assistant"))
                break
            elif item.role == "assistant":
                break
        memory.append(Message(content=request.payload.message, role="user"))

        if request.payload.add_to_insights:
            with state.update() as cur:
                if isinstance(cur.steps[-1], DefaultStep):
                    cur.steps[-1].insights.append(
                        InsightSharingOutput(
                            insight=request.payload.message,
                            justification="USER",
                            generated_at_memory_index=len(memory) - 1,
                        )
                    )
    elif memory and memory[-1].role == "assistant":
        memory.append(Message(content=".", role="user"))

    # restart the step
    event_manager.restart_step(step_to_restart)
    if is_coding_step:
        AutofixCodingStep.get_signature(
            AutofixCodingStepRequest(
                run_id=state.get().run_id,
                initial_memory=memory,
            ),
            queue=app_config.CELERY_WORKER_QUEUE,
        ).apply_async()
    else:
        RootCauseStep.get_signature(
            RootCauseStepRequest(
                run_id=state.get().run_id,
                initial_memory=memory,
            ),
            queue=app_config.CELERY_WORKER_QUEUE,
        ).apply_async()


def update_code_change(request: AutofixUpdateRequest):
    if not isinstance(request.payload, AutofixUpdateCodeChangePayload):
        raise ValueError("Invalid payload type for update_code_change")

    state = ContinuationState(request.run_id)
    cur_state = state.get()

    repo_id = request.payload.repo_id
    hunk_index = request.payload.hunk_index
    lines = request.payload.lines
    file_path = request.payload.file_path

    # get the last step and make sure it's a changes step
    last_step = cur_state.steps[-1]
    if not isinstance(last_step, ChangesStep):
        return
    changes = last_step.changes

    # find the change with the matching repo_external_id
    matching_change = None
    change_index = 0
    for i, change in enumerate(changes):
        if change.repo_external_id == repo_id:
            matching_change = change
            change_index = i
            break
    if not matching_change:
        raise ValueError("No matching change found")

    # check that we have a matching file patch in the codebase state using the file path
    matching_file_patch = None
    file_patch_index = 0
    for i, file_patch in enumerate(matching_change.diff):
        if file_patch.path == file_path:
            file_patch_index = i
            matching_file_patch = file_patch
            break
    if not matching_file_patch:
        raise ValueError("No matching file patch found")

    # check that we have a matching hunk in the file patch using the hunk index
    if hunk_index < 0 or hunk_index >= len(matching_file_patch.hunks):
        raise ValueError("Hunk index is out of range")
    matching_hunk = matching_file_patch.hunks[hunk_index]

    # calculate the change in hunk length
    old_hunk_length = matching_hunk.target_length
    new_hunk_length = len([line for line in lines if line.line_type in [" ", "+"]])
    length_diff = new_hunk_length - old_hunk_length

    # replace the hunk lines and update its length
    with state.update() as cur:
        # Update line numbers within the current hunk
        current_target_line_no = matching_hunk.target_start
        for line in lines:
            if line.line_type in [" ", "+"]:
                line.target_line_no = current_target_line_no
                current_target_line_no += 1
            elif line.line_type == "-":
                line.target_line_no = None

        matching_hunk.lines = lines
        matching_hunk.target_length = new_hunk_length
        matching_file_patch.hunks[hunk_index] = matching_hunk

        # update line numbers in subsequent hunks and their lines
        for subsequent_hunk in matching_file_patch.hunks[hunk_index + 1 :]:
            subsequent_hunk.target_start += length_diff
            for line in subsequent_hunk.lines:
                if line.target_line_no is not None:
                    line.target_line_no += length_diff

        matching_change.diff[file_patch_index] = matching_file_patch
        last_step.changes[change_index] = matching_change
        cur.steps[-1] = last_step


def comment_on_thread(request: AutofixUpdateRequest):
    if not isinstance(request.payload, AutofixCommentThreadPayload):
        raise ValueError("Invalid payload type for comment_on_thread")

    state = ContinuationState(request.run_id)

    event_manager = AutofixEventManager(state)
    context = AutofixContext(
        state=state, event_manager=event_manager, sentry_client=get_sentry_client()
    )

    step_index = request.payload.step_index

    # create a new thread if needed
    step = state.get().steps[step_index]
    if not step.active_comment_thread or step.active_comment_thread.id != request.payload.thread_id:
        with state.update() as cur:
            cur.steps[step_index].active_comment_thread = CommentThread(
                id=request.payload.thread_id,
                selected_text=request.payload.selected_text,
            )
    else:
        with state.update() as cur:
            # update the selected text in case it changed
            cur.steps[step_index].active_comment_thread.selected_text = (
                request.payload.selected_text
            )

    # add comment to the thread
    message = Message(
        role="user",
        content=request.payload.message,
    )
    with state.update() as cur:
        cur.steps[step_index].active_comment_thread.messages.append(message)

    # fetch memory from the step
    is_coding_step = step.key == "plan"
    memory = (
        context.get_memory("plan_and_code")
        if is_coding_step
        else context.get_memory("root_cause_analysis")
    )

    # ask LLM for response
    step = state.get().steps[step_index]
    comment_thread_component = CommentThreadComponent(context=context)
    response = comment_thread_component.invoke(
        CommentThreadRequest(
            run_memory=memory,
            thread_memory=step.active_comment_thread.messages,
            selected_text=step.active_comment_thread.selected_text,
        )
    )
    with state.update() as cur:
        cur.steps[step_index].active_comment_thread.messages.append(
            Message(content=response.comment_in_response, role="assistant")
        )
        if response.action_requested:
            cur.steps[step_index].active_comment_thread.is_completed = True

    if response.action_requested:
        formatted_thread_memory = (
            "Based on the following conversation, rethink your analysis:\n"
            + "\n".join(
                [
                    f"{message.role}: {message.content}"
                    for message in step.active_comment_thread.messages
                ]
            )
        )
        # rethink from correct point
        restart_from_point_with_feedback(
            AutofixUpdateRequest(
                run_id=request.run_id,
                payload=AutofixRestartFromPointPayload(
                    type=AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK,
                    step_index=step_index,
                    retain_insight_card_index=request.payload.retain_insight_card_index,
                    message=formatted_thread_memory,
                    add_to_insights=False,
                ),
            )
        )


@inject
def run_autofix_evaluation(
    request: AutofixEvaluationRequest,
    app_config: AppConfig = injected,
):
    langfuse = Langfuse()

    dataset = langfuse.get_dataset(request.dataset_name)
    items = dataset.items

    if request.run_on_item_id:
        items = [item for item in items if item.id == request.run_on_item_id]

    if request.test and not request.run_on_item_id:
        if request.random_for_test:
            items = random.sample(items, 1)
        else:
            items = items[:1]

    logger.info(
        f"Starting autofix evaluation for dataset {request.dataset_name} with run name '{request.run_name}'."
    )
    logger.info(f"Number of items: {len(items)}")
    logger.info(f"Total number of runs: {len(items) * request.n_runs_per_item}")

    for i, item in enumerate(items):
        # Note: This will add ALL the dataset item runs into the CPU queue.
        # As we are not going to be running this in prod yet, it's fine to leave as is.
        # If we do decide to run in prod, should find a way to not overwhelm the CPU queue.
        for _ in range(request.n_runs_per_item):
            run_autofix_evaluation_on_item.apply_async(
                (),
                dict(
                    item_id=item.id,
                    run_name=request.run_name,
                    run_description=request.run_description,
                    run_type=request.run_type,
                    item_index=i,
                    item_count=len(items),
                ),
                queue=app_config.CELERY_WORKER_QUEUE,
            )


@celery_app.task()
def run_autofix_evaluation_on_item(
    *,
    item_id: str,
    run_name: str,
    run_description: str,
    run_type: Literal["execution", "full", "root_cause"],
    item_index: int,
    item_count: int,
):
    langfuse = Langfuse()

    dataset_item = langfuse.get_dataset_item(item_id)

    logger.info(
        f"Starting autofix evaluation for item {item_id}, number {item_index}/{item_count}, with run name '{run_name}'."
    )

    scoring_n_panel = 5
    scoring_model = "o1-mini-2024-09-12"

    diff: str | None = None
    causes: list[RootCauseAnalysisItem] | None = None

    with dataset_item.observe(run_name=run_name, run_description=run_description) as trace_id:
        if run_type == "root_cause":
            try:
                causes = sync_run_root_cause(dataset_item, langfuse_session_id=trace_id)
            except Exception as e:
                logger.error(f"Error running root cause analysis: {e}")

            if causes:
                root_cause_score = score_root_causes(
                    dataset_item,
                    causes,
                    n_panel=scoring_n_panel,
                    model=scoring_model,
                    langfuse_session_id=trace_id,
                )

                # Will output 4 scores for a run item:
                # - `"highest_score"`: The score for the highest scored cause out of all the returned root causes.
                # - `"positioning_score"`: Positioning of the highest scored cause, if the highest scored cause is first, this score is `1.0`. If it is last, it will be `0.0`. The score is calculated proportional to the number of causes provided.
                # - `"mean_score"`: The mean score of all the root causes.
                # - `"error_weighted_score"` This score is the same as the highest score but scored 0 if there is an error or no root cause returned. This is used to weight the score in the aggregated run result.

                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=root_cause_score.get("highest_score"),
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="highest_score"
                    ),
                    value=root_cause_score.get("highest_score"),
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="positioning_score"
                    ),
                    value=root_cause_score.get("position_score"),
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="mean_score"
                    ),
                    value=root_cause_score.get("mean_score"),
                )
            else:
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=0,
                )
        elif run_type == "execution":
            try:
                diff = sync_run_execution(dataset_item, langfuse_session_id=trace_id)
            except Exception as e:
                logger.error(f"Error running evaluation: {e}")

            if diff:
                score = score_one(
                    dataset_item,
                    diff,
                    n_panel=scoring_n_panel,
                    model=scoring_model,
                    langfuse_session_id=trace_id,
                )

                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=score,
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="score"
                    ),
                    value=score,
                )
            else:
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=0,
                )
        else:
            try:
                diff, causes = sync_run_evaluation_on_item(
                    dataset_item, langfuse_session_id=trace_id
                )
            except Exception as e:
                logger.exception(f"Error running evaluation: {e}")

            if diff:
                score, verdict = score_one(
                    dataset_item,
                    diff,
                    n_panel=scoring_n_panel,
                    model=scoring_model,
                    langfuse_session_id=trace_id,
                )

                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="is_fixed"
                    ),
                    value=1 if verdict else 0,
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=score,
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="score"
                    ),
                    value=score,
                )
            else:
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=0,
                )

            if causes:
                root_cause_score, root_cause_verdict = score_root_causes(
                    dataset_item,
                    causes,
                    n_panel=scoring_n_panel,
                    model=scoring_model,
                    langfuse_session_id=trace_id,
                )

                # Will output 4 scores for a run item:
                # - `"highest_score"`: The score for the highest scored cause out of all the returned root causes.
                # - `"error_weighted_score"` This score is the same as the highest score but scored 0 if there is an error or no root cause returned. This is used to weight the score in the aggregated run result.

                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="rc_is_correct"
                    ),
                    value=1 if root_cause_verdict else 0,
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="rc_error_weighted_score"
                    ),
                    value=root_cause_score,
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="rc_score"
                    ),
                    value=root_cause_score,
                )
            else:
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="rc_error_weighted_score"
                    ),
                    value=0,
                )
