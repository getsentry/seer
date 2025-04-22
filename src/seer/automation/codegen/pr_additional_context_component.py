from typing import Dict, List, Optional, Set, cast

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.codegen.models import (
    CodeFetchIssuesOutput,
    CodeFetchIssuesRequest,
    PrAdditionalContext,
    PrAdditionalContextOutput,
    PrAdditionalContextRequest,
)
from seer.automation.codegen.relevant_warnings_component import FetchIssuesComponent
from seer.automation.models import EventDetails, IssueDetails, Profile, SentryEventData, TraceEvent
from seer.automation.pipeline import BaseComponent
from seer.dependency_injection import inject, injected
from seer.rpc.client import RpcClient

MAX_NUM_ISSUES_ANALYZED = 500


class PrAdditionalContext(BaseComponent[PrAdditionalContextRequest, PrAdditionalContextOutput]):
    """
    Provides additional context for PRs such as Sentry profile data and trace information.
    """

    @observe(name="Generate Additional Context for the PR")
    @ai_track(description="Generate Additional Context for the PR")
    @inject
    def invoke(
        self, request: PrAdditionalContextRequest, rpc_client: RpcClient = injected
    ) -> PrAdditionalContextOutput:
        self.rpc_client = rpc_client

        filename_to_additional_context: Dict[str, PrAdditionalContext] = {}

        # 1. Fetch issues related to the PR's files and functions
        fetch_issues_component = FetchIssuesComponent(self.context)
        fetch_issues_request = CodeFetchIssuesRequest(
            organization_id=request.organization_id, pr_files=request.pr_files
        )
        fetch_issues_output: CodeFetchIssuesOutput = fetch_issues_component.invoke(
            fetch_issues_request
        )

        # 2. Fetch profile and trace data from issues (up to the limit)
        total_issues_processed = 0
        for filename, issues in fetch_issues_output.filename_to_issues.items():
            if total_issues_processed + len(issues) >= MAX_NUM_ISSUES_ANALYZED:
                break
            total_issues_processed += len(issues)

            profiles = self._fetch_profile_from_issues(issues)
            traces = self._fetch_trace_from_issues(issues)

            filename_to_additional_context[filename] = PrAdditionalContext(
                profiles=profiles,
                traces=traces,
            )

        # 3. Return the additional contexts
        return PrAdditionalContextOutput(
            filename_to_additional_context=filename_to_additional_context
        )

    @observe(name="Get Profiles for PR Review")
    @ai_track(description="Get Profiles for PR Review")
    def _fetch_profile_from_issues(self, issues: List[IssueDetails]) -> List[Profile]:
        profiles: List[Profile] = []

        try:
            event_id_to_event: Dict[str, EventDetails] = {}

            # Take the first event from each issue as representative
            for issue in issues:
                if not issue.events:
                    continue
                event = issue.events[0]
                if not event.profile_id:
                    continue
                event_id_to_event[event.event_id] = event

            # Fetch profiles for the collected events
            for event in event_id_to_event.values():
                profile_data = self.rpc_client.call(
                    "get_profile_details",
                    organization_id=event.organization_id,
                    project_id=event.project_id,
                    profile_id=event.profile_id,
                )
                if not profile_data:
                    self.logger.warning(f"Could not fetch profile for event {event_id}")
                    continue
                try:
                    profile = Profile.model_validate(profile_data)
                    profiles.append(profile)
                except Exception as e:
                    self.logger.exception(f"Could not parse profile from tool call: {e}")
                    continue

            return profiles
        except Exception as e:
            self.logger.exception(f"Error fetching profiles from issues: {e}")
            return []

    @observe(name="Get Traces for PR Review")
    @ai_track(description="Get Traces for PR Review")
    def _fetch_trace_from_issues(self, issues: List[IssueDetails]) -> List[SentryEventData]:
        traces = []
        try:
            # Get all issues with events that have traces
            for issue in issues:
                if not issue.events:
                    continue

                event = issue.events[0]
                if not event.trace_id:
                    continue

                # Skip if we already have this trace ID
                if any(trace.id == event.trace_id for trace in traces):
                    continue

                error_data = self.rpc_client.call(
                    "get_error_event_details",
                    project_id=event.project_id,
                    event_id=event.event_id,
                )  # expecting data compatible with SentryEventData model
                if not error_data:
                    self.logger.warning(f"Could not fetch trace for event {event.event_id}")
                    continue

                try:
                    data = SentryEventData.model_validate(error_data)
                    traces.append(data)
                except Exception as e:
                    self.logger.exception(f"Could not process trace data: {e}")
                    continue

            return traces
        except Exception as e:
            self.logger.exception(f"Error fetching traces from issues: {e}")
            return []
