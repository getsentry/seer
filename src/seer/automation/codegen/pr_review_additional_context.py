import logging
from typing import Dict, List, Optional, Set, cast

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.codegen.models import (
    CodeFetchIssuesOutput,
    CodeFetchIssuesRequest,
    CodegenPRReviewAdditionalContextOutput,
    CodegenPRReviewAdditionalContextRequest,
)
from seer.automation.codegen.relevant_warnings_component import FetchIssuesComponent
from seer.automation.models import EventDetails, IssueDetails, Profile, SentryEventData, TraceEvent
from seer.automation.pipeline import BaseComponent
from seer.dependency_injection import inject, injected
from seer.rpc.client import RpcClient


class PrReviewAdditionalContext(
    BaseComponent[CodegenPRReviewAdditionalContextRequest, CodegenPRReviewAdditionalContextOutput]
):
    """
    Provides additional context for PR reviews such as Sentry profile data and trace information.
    """

    def __init__(self, pr_files: List[Dict[str, any]]):
        self.pr_files = pr_files
        self.profile = None
        self.trace = None

    @observe(name="Generate Additional Context for PR Review")
    @ai_track(description="Generate Additional Context for PR Review")
    @inject
    def invoke(
        self, request: CodegenPRReviewAdditionalContextRequest, rpc_client: RpcClient = injected
    ) -> CodegenPRReviewAdditionalContextOutput:
        """
        Main method to generate additional context for PR reviews.
        """

        # 1. Fetch issues related to the PR.
        fetch_issues_component = FetchIssuesComponent(self.context)
        fetch_issues_request = CodeFetchIssuesRequest(
            organization_id=self.request.organization_id, pr_files=request.pr_files
        )
        fetch_issues_output: CodeFetchIssuesOutput = fetch_issues_component.invoke(
            fetch_issues_request
        )
        # Clamp issue to max_num_issues_analyzed
        all_selected_issues = list(
            itertools.chain.from_iterable(fetch_issues_output.filename_to_issues.values())
        )
        all_selected_issues = all_selected_issues[: self.request.max_num_issues_analyzed]

        # 2. Fetch profile data from issues.
        self.profile = self.fetch_profile_from_issues(all_selected_issues)

        # 3. Fetch trace data from issues.
        self.trace = self.fetch_trace_from_issues(all_selected_issues)

        # 3. Return the constructed LLM prompt.
        return CodegenPRReviewAdditionalContextOutput(
            additional_context=self._construct_llm_prompt()
        )

    def _construct_llm_prompt(self) -> str | None:
        context_parts = []

        if self.profile:
            profile_context = (
                "## Performance Profile Data\n"
                "The following performance profile was detected for code related to this PR:\n"
                f"- Profile ID: {self.profile.id}\n"
            )

            if hasattr(self.profile, "transaction") and self.profile.transaction:
                profile_context += f"- Transaction: {self.profile.transaction}\n"

            if hasattr(self.profile, "frames") and self.profile.frames:
                profile_context += "- Key frames detected in the profile that may be relevant\n"

            context_parts.append(profile_context)

        if self.trace:
            trace_context = (
                "## Trace Information\n"
                "The following trace data was detected for code related to this PR:\n"
                f"- Trace ID: {self.trace.id}\n"
            )

            if hasattr(self.trace, "spans") and self.trace.spans:
                trace_context += f"- Contains {len(self.trace.spans)} spans\n"

            context_parts.append(trace_context)

        if not context_parts:
            return None

        return "\n".join(context_parts)

    @observe(name="Get Profiles for PR Review")
    @ai_track(description="Get Profiles for PR Review")
    @inject
    def fetch_profile_from_issues(
        self, issues: List[IssueDetails], rpc_client: RpcClient = injected
    ) -> Dict[str, Profile]:
        profiles = {}
        try:
            # Get all issues with events that have profiles
            for issue in issues:
                if not issue.events:
                    continue

                event = issue.events[0]
                if not event.profile_id:
                    continue

                profile_data = rpc_client.call(
                    "get_profile_details",
                    organization_id=event.organization_id,
                    project_id=event.project_id,
                    profile_id=event.profile_id,
                )  # expecting data compatible with Profile model
                if not profile_data:
                    return "Could not fetch profile."
                try:
                    profile = Profile.model_validate(profile_data)
                    return profile.format_profile(context_before=100, context_after=100)
                except Exception as e:
                    self.logger.exception(f"Could not parse profile from tool call: {e}")
                    return "Could not fetch profile."

            return profiles
        except Exception as e:
            self.logger.exception(f"Error fetching profiles from issues: {e}")
            return {}

    @observe(name="Get Traces for PR Review")
    @ai_track(description="Get Traces for PR Review")
    @inject
    def fetch_trace_from_issues(
        self, issues: List[IssueDetails], rpc_client: RpcClient = injected
    ) -> Dict[str, SentryEventData]:
        traces = {}
        try:
            # Get all issues with events that have traces
            for issue in issues:
                if not issue.events:
                    continue

                event = issue.events[0]
                if not event.trace_id:
                    continue

                # Skip if we already have this trace
                if event.trace_id in traces:
                    continue

                error_data = rpc_client.call(
                    "get_error_event_details",
                    project_id=event.project_id,
                    event_id=event.event_id,
                )  # expecting data compatible with SentryEventData model
                if not error_data:
                    self.logger.warning(f"Could not fetch trace for event {event.event_id}")
                    continue

                data = cast(SentryEventData, error_data)
                try:
                    traces[event.trace_id] = data
                except Exception as e:
                    self.logger.exception(f"Could not process trace data: {e}")
                    continue

            return traces
        except Exception as e:
            self.logger.exception(f"Error fetching traces from issues: {e}")
            return {}
