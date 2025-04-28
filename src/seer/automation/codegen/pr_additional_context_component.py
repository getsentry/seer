from typing import Dict, List, Optional, Set, cast

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.codegen.models import (
    PrAdditionalContextComponent,
    PrAdditionalContextOutput,
    PrAdditionalContextRequest,
)
from seer.automation.component import BaseComponent
from seer.automation.models import EventDetails, IssueDetails, Profile, SentryEventData, TraceEvent
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient

MAX_NUM_ISSUES_ANALYZED = 500


class PrAdditionalContextComponent(
    BaseComponent[PrAdditionalContextRequest, PrAdditionalContextOutput]
):
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

        filename_to_additional_context: Dict[str, PrAdditionalContextComponent] = {}

        # Fetch profile and trace data from issues (up to the issues limit)
        total_issues_processed = 0
        for filename, issues in request.filename_to_issues.items():
            if total_issues_processed + len(issues) >= MAX_NUM_ISSUES_ANALYZED:
                break
            total_issues_processed += len(issues)

            profiles = self._fetch_profiles_from_issues(issues)
            traces = self._fetch_traces_from_issues(issues)

            filename_to_additional_context[filename] = PrAdditionalContextComponent(
                profiles=profiles,
                traces=traces,
            )

        return PrAdditionalContextOutput(
            filename_to_additional_context=filename_to_additional_context
        )

    @observe(name="Get Profiles for PR Review")
    @ai_track(description="Get Profiles for PR Review")
    def _fetch_profiles_from_issues(self, issues: List[IssueDetails]) -> List[Profile]:
        profiles: List[Profile] = [] # TODO - maybe needs to be a map

        try:
            # Take the first event from each issue as representative
            for issue in issues:
                if not issue.events:
                    continue
                event = issue.events[0]

                profile_id = 'TODO'
                # TODO: Looks like I'll need to add an rpc method in sentry to fetch profile id from event :(
                # In Sentry frontend it uses this api https://us.sentry.io/api/0/organizations/sentry/profiling/flamegraph/?project=6178942&event.type%3Atransaction%20transaction%3A${encodeURIComponent(transactionName)}&statsPeriod=7d

                # Fetch profiles for the selected event
                profile_data = self.rpc_client.call(
                    "get_profile_details",
                    organization_id=self.context.repo.organization_id,
                    project_id=self.context.repo.project_id,
                    profile_id=profile_id,
                )
                if not profile_data:
                    self.logger.warning(f"Could not fetch profile for event {event.event_id}")
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
    def _fetch_traces_from_issues(self, issues: List[IssueDetails]) -> List[SentryEventData]:
        # TODO - :( need access to the data returned from here - https://us.sentry.io/api/0/organizations/${organizationSlug}/events/${projectName}:${eventId}/?referrer=trace-details-summary
