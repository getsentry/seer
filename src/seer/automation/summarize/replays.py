import json
import textwrap

from langfuse.decorators import observe
from pydantic import BaseModel, model_validator

from seer.automation.agent.client import GptClient
from seer.automation.utils import extract_parsed_model
from seer.dependency_injection import inject, injected


class ReplayEvent(BaseModel):
    message: str
    category: str
    data: dict | None = None
    type: str

    @model_validator(mode="after")
    def validate_data(self):
        if self.type == "error":
            assert self.data is not None, "data is required for error event"

        return self


class Replay(BaseModel):
    events: list[ReplayEvent]


class Step(BaseModel):
    description: str
    referenced_ids: list[int]
    error_group_ids: list[int]


class ReplaySummary(BaseModel):
    user_steps_taken: list[Step]
    pages_visited: list[str]
    user_journey_summary: str


class SummarizeReplaysRequest(BaseModel):
    group_id: int
    replays: list[Replay]


class CommonStep(BaseModel):
    reasoning: str
    description: str


class CommonReplaySummary(BaseModel):
    common_user_steps_taken: list[CommonStep]
    reproduction_steps: str
    issue_impact_summary: str


class SummarizeReplaysResponse(BaseModel):
    common_steps: list[CommonStep]
    reproduction: str
    impact_summary: str

    @classmethod
    def from_parsed_model(cls, parsed_model: CommonReplaySummary):
        return cls(
            common_steps=parsed_model.common_user_steps_taken,
            reproduction=parsed_model.reproduction_steps,
            impact_summary=parsed_model.issue_impact_summary,
        )


def find_steps_around_group_id(
    replay_summary: ReplaySummary,
    target_group_id: int,
    max_steps_before: int = 16,
    max_steps_after: int = 4,
) -> list[Step]:
    result = []
    found_index = -1

    # Find the step containing the target group ID
    for index, step in enumerate(replay_summary.user_steps_taken):
        if target_group_id in step.error_group_ids:
            found_index = index
            break

    if found_index == -1:
        return []  # Target group ID not found

    # Collect steps before
    start_index = max(0, found_index - max_steps_before)
    result.extend(replay_summary.user_steps_taken[start_index:found_index])

    # Add the step with the target group ID
    result.append(replay_summary.user_steps_taken[found_index])

    # Collect steps after
    end_index = min(len(replay_summary.user_steps_taken), found_index + max_steps_after + 1)
    result.extend(replay_summary.user_steps_taken[found_index + 1 : end_index])

    return result


@observe(name="Single replay summary")
@inject
def run_single_replay_summary(replay: Replay, gpt_client: GptClient = injected) -> ReplaySummary:
    replay_prompt = textwrap.dedent(
        """\
        You are an exceptional developer that analyzes a replay of a user's interaction with an application and can summarize it in 1-2 sentences.
        {replay_data}

        1. Analyze the replay, and summarize the steps the user took in an ordered list. Describe them as you would a story.
        - It's important you mention all issues and errors that ocurred in detail in a step.
        - Remove any unnecessary words and keep it concise, for example "Clicked on the red button".

        2. Provide a list of the pages that the user visited, just name the pages don't provide the entire URL.

        3. Provide a summary of what the user did in a couple of sentences. If the user encountered any errors, describe them and what the user was doing when they encountered them.
        - Include all errors that ocurred, they're important.
        - Don't try to assume or describe anything about the error, just say there was an error.
        - When mentioning an error, don't assume anything about what caused the error, just describe what the user was doing when they encountered the error.
        - Did the user continue after the error? Did they see an error message? Did they quit the application? Describe what they did in that case.
        - Be very specific about what the user did around an error, did a button trigger it? If yes then which, did it just happen when they navigated to a page? Describe what the user was doing around the error.
        - Don't try to analyze the user's behavior, just describe what they did."""
    ).format(replay_data=json.dumps(replay.model_dump(mode="json")))

    replay_message_dicts = [
        {
            "content": replay_prompt,
            "role": "user",
        },
    ]

    completion = gpt_client.openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=replay_message_dicts,  # type: ignore
        response_format=ReplaySummary,
        temperature=0.0,
    )

    return extract_parsed_model(completion)


def step_to_string(step: Step, target_group_id: int):
    content = step.description
    if target_group_id in step.error_group_ids:
        content += " <-- The issue occurred here"

    return content


def steps_to_string(steps: list[Step], target_group_id: int):
    return "\n".join([step_to_string(step, target_group_id) for step in steps])


def targeted_steps_to_string(replay_summary: ReplaySummary, target_group_id: int) -> str:
    steps_around_error = find_steps_around_group_id(replay_summary, target_group_id)

    return steps_to_string(steps_around_error, target_group_id)


@observe(name="Cross session replay summary")
@inject
def run_cross_session_completion(all_steps: list[str], gpt_client: GptClient = injected):
    replay_prompt = textwrap.dedent(
        """\
        You are an exceptional developer that analyzes the replay of multiple users' interactions with an application and can understand the impact and common issues that occur.
        {all_steps}

        1. Analyze the replays, and summarize the common steps that users took in an ordered list. Describe them as you would a story.
        - Include only the common steps taken between all the users to give a good understanding of what caused the issue.
        - It's important you mention all issues and errors that ocurred in detail in a step.
        - Remove any unnecessary words and keep it concise, for example "Clicked on the red button".
        - Don't try to assume or describe anything about the error or the application, just say there was an error.
        - What was the common result after the issue occurred? Did the app crash? Did the users rage click? Or did they just continue?

        2. Provide a summary of the reproduction steps to trigger the issue.
        - Only include relevant steps to trigger the issue. Other irrelevant obvious steps don't need to be there.

        3. Provide a 1-2 sentence summary of the impact the issue had on users.
        - Focus on the impact the marked issue in each step had on user
        - Did this cause the app to crash? Did an error message show? Did it show a blank page? What was the result of this error.
        - Start with the actions to produce the issue, then explain the impact"""
    ).format(all_steps=json.dumps(all_steps))

    replay_message_dicts = [
        {
            "content": replay_prompt,
            "role": "user",
        },
    ]

    completion = gpt_client.openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=replay_message_dicts,  # type: ignore
        response_format=CommonReplaySummary,
        temperature=0.0,
    )

    return extract_parsed_model(completion)


@observe(name="Summarize Replay for Issue")
def summarize_replays(request: SummarizeReplaysRequest) -> SummarizeReplaysResponse:
    replay_summaries = []
    for replay in request.replays:
        replay_summaries.append(run_single_replay_summary(replay))

    all_targeted_steps = [
        targeted_steps_to_string(summary, request.group_id) for summary in replay_summaries
    ]

    common_summary = run_cross_session_completion(all_targeted_steps)

    return SummarizeReplaysResponse.from_parsed_model(common_summary)
