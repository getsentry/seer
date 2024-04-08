import textwrap
from typing import Any

import sentry_sdk
from langsmith import traceable

from celery_app.models import UpdateCodebaseTaskRequest
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.assessment.component import ProblemDiscoveryComponent
from seer.automation.autofix.components.assessment.models import ProblemDiscoveryRequest
from seer.automation.autofix.components.executor.component import ExecutorComponent
from seer.automation.autofix.components.executor.models import ExecutorRequest
from seer.automation.autofix.components.planner.component import PlanningComponent
from seer.automation.autofix.components.planner.models import PlanningRequest, PlanStep
from seer.automation.autofix.components.retriever import RetrieverComponent, RetrieverRequest
from seer.automation.autofix.models import (
    AutofixOutput,
    AutofixRequest,
    AutofixStatus,
    EventDetails,
    ProblemDiscoveryResult,
    PullRequestResult,
    RepoDefinition,
)
from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.tasks import update_codebase_index
from seer.automation.pipeline import Pipeline


class Autofix(Pipeline):
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @traceable(name="Autofix Run", tags=["autofix:v1.1"])
    def invoke(self, request: AutofixRequest):
        try:
            autofix_logger.info(f"Beginning autofix for issue {request.issue.id}")

            self.context.event_manager.send_initial_steps()

            event_details = EventDetails.from_event(request.issue.events[0])
            problem_discovery_output = ProblemDiscoveryComponent(self.context).invoke(
                ProblemDiscoveryRequest(
                    event_details=event_details,
                    instruction=request.instruction,
                )
            )

            if not problem_discovery_output:
                autofix_logger.warning(f"Problem discovery agent did not return a valid response")
                return

            problem_discovery_payload = ProblemDiscoveryResult(
                status=(
                    "CONTINUE"
                    if problem_discovery_output.actionability_score >= 0.6
                    else "CANCELLED"
                ),
                description=problem_discovery_output.description,
                reasoning=problem_discovery_output.reasoning,
            )

            self.context.event_manager.send_problem_discovery_result(problem_discovery_payload)

            if problem_discovery_payload.status == "CANCELLED":
                autofix_logger.info(f"Problem is not actionable")
                return

            for repo in request.repos:
                self.context.event_manager.send_codebase_indexing_repo_check_message(repo.full_name)
                if self.context.has_codebase_index(repo):
                    self.context.event_manager.send_codebase_indexing_repo_exists_message(
                        repo.full_name
                    )
                else:
                    autofix_logger.info(f"Creating codebase index for repo {repo.name}")
                    self.context.event_manager.send_codebase_index_creation_message(repo.full_name)
                    with sentry_sdk.start_span(
                        op="seer.automation.autofix.codebase_index.create",
                        description="Create codebase index",
                    ) as span:
                        span.set_tag("repo", repo.name)
                        self.context.create_codebase_index(repo)
                    autofix_logger.info(f"Codebase index created for repo {repo.name}")
                    self.context.event_manager.send_codebase_index_creation_complete_message(
                        repo.full_name
                    )

            for repo_id, codebase in self.context.codebases.items():
                repo_full_name = codebase.repo_client.repo_full_name
                if request.base_commit_sha:
                    autofix_logger.info(
                        f"Updating codebase index for repo {repo_full_name} to {request.base_commit_sha}"
                    )
                    self.context.event_manager.send_codebase_index_update_wait_message(
                        repo_full_name
                    )
                    with sentry_sdk.start_span(
                        op="seer.automation.autofix.codebase_index.update",
                        description="Update codebase index",
                    ) as span:
                        span.set_tag("repo", codebase.repo_info.external_slug)
                        codebase.update(request.base_commit_sha, is_temporary=True)
                    self.context.event_manager.send_codebase_index_up_to_date_message(
                        repo_full_name
                    )

                elif codebase.is_behind():
                    if self.context.diff_contains_stacktrace_files(repo_id, event_details):
                        autofix_logger.debug(
                            f"Waiting for codebase index update for repo {codebase.repo_info.external_slug}"
                        )
                        self.context.event_manager.send_codebase_index_update_wait_message(
                            repo_full_name
                        )
                        with sentry_sdk.start_span(
                            op="seer.automation.autofix.codebase_index.update",
                            description="Update codebase index",
                        ) as span:
                            span.set_tag("repo", codebase.repo_info.external_slug)
                            codebase.update()
                        autofix_logger.debug(f"Codebase index updated")
                        self.context.event_manager.send_codebase_index_up_to_date_message(
                            repo_full_name
                        )
                    else:
                        update_codebase_index.apply_async(
                            (UpdateCodebaseTaskRequest(repo_id=repo_id).model_dump(),),
                            countdown=10 * 60,
                        )  # 10 minutes
                        autofix_logger.info(f"Codebase indexing scheduled for later")
                        self.context.event_manager.send_codebase_index_update_scheduled_message(
                            repo_full_name
                        )
                else:
                    autofix_logger.debug(f"Codebase is up to date")
                    self.context.event_manager.send_codebase_index_up_to_date_message(
                        repo_full_name
                    )

            if not self.context.codebases:
                autofix_logger.warning(f"No codebase indexes")
                sentry_sdk.capture_message(
                    f"No codebases found for organization {request.organization_id} and project {request.project_id}'s repos: {', '.join([repo.name for repo in request.repos])}"
                )
                self.context.event_manager.send_autofix_complete(None)
                return

            self.context.event_manager.send_codebase_indexing_result(AutofixStatus.COMPLETED)

            self.context.process_event_paths(event_details)

            try:
                planning_output = PlanningComponent(self.context).invoke(
                    PlanningRequest(
                        event_details=event_details,
                        problem=problem_discovery_output,
                        instruction=request.instruction,
                    )
                )

                if not planning_output:
                    autofix_logger.warning(f"Planning agent did not return a valid response")
                    self.context.event_manager.send_planning_result(None)
                    return

                self.context.event_manager.send_planning_result(planning_output)

                autofix_logger.info(
                    f"Planning complete; there are {len(planning_output.steps)} steps in the plan to execute."
                )
            except Exception as e:
                autofix_logger.error(f"Failed to plan: {e}")
                sentry_sdk.capture_exception(e)
                self.context.event_manager.send_planning_result(None)
                return

            retriever = RetrieverComponent(self.context)
            executor = ExecutorComponent(self.context)
            for i, step in enumerate(planning_output.steps):
                self.context.event_manager.send_execution_step_start(step.id)

                autofix_logger.info(f"Executing step: {i}/{len(planning_output.steps)}")

                self.run_executor_with_retriever(retriever, executor, step, event_details)

                self.context.event_manager.send_execution_step_result(
                    step.id, AutofixStatus.COMPLETED
                )

            slug_id: str | None = self._get_slug_id(request.organization_id)

            prs = self._create_prs(
                planning_output.title,
                planning_output.description,
                planning_output.steps,
                slug_id,
                request,
            )

            outputs = []
            if prs:
                # TODO: Support more than 1 PR...
                pr = prs[0]
                output = AutofixOutput(
                    title=planning_output.title,
                    description=planning_output.description,
                    pr_url=pr.pr_url,
                    repo_name=f"{pr.repo.owner}/{pr.repo.name}",
                    pr_number=pr.pr_number,
                    diff=pr.diff,
                    diff_str=pr.diff_str,
                    usage=self.context.state.get().usage,
                )
                self.context.event_manager.send_autofix_complete(output)
                outputs.append(output)
            else:
                self.context.event_manager.send_autofix_complete(None)

            file_changes = {}
            for repo_id, codebase in self.context.codebases.items():
                file_changes[repo_id] = codebase.file_changes

            return {"outputs": outputs, "prs": prs, "file_changes": file_changes}
        except Exception as e:
            autofix_logger.error(f"Failed to complete autofix")
            autofix_logger.exception(e)
            sentry_sdk.capture_exception(e)

            self.context.event_manager.send_autofix_complete(None)
        finally:
            self.context.cleanup()
            autofix_logger.info(f"Autofix complete for issue {request.issue.id}")

    def _create_prs(
        self,
        title: str,
        description: str,
        steps: list[Any],
        slug_id: str | None,
        request: AutofixRequest,
    ):
        state = self.context.state.get()
        prs: list[PullRequestResult] = []
        for codebase in self.context.codebases.values():
            if codebase.file_changes:
                if self.context.commit_changes:
                    branch_ref = codebase.repo_client.create_branch_from_changes(
                        pr_title=title, file_changes=codebase.file_changes
                    )

                    if branch_ref is None:
                        autofix_logger.warning(f"Failed to create branch from changes")
                        return None

                    pr_title = f"""🤖 {title}"""

                    ref_note = ""
                    if slug_id:
                        issue_url = (
                            f"https://sentry.io/organizations/{slug_id}/issues/{request.issue.id}/"
                        )
                        issue_link = (
                            f"[{request.issue.short_id}]({issue_url})"
                            if request.issue.short_id
                            else issue_url
                        )
                        ref_note = f"Fixes {issue_link}\n"

                    pr_description = textwrap.dedent(
                        """\
                        👋 Hi there! This PR was automatically generated 🤖
                        {user_line}

                        {ref_note}
                        {description}

                        #### The steps that were performed:
                        {steps}

                        ### 📣 Instructions for the reviewer which is you, yes **you**:
                        - **If these changes were incorrect, please close this PR and comment explaining why.**
                        - **If these changes were incomplete, please continue working on this PR then merge it.**
                        - **If you are feeling confident in my changes, please merge this PR.**

                        This will greatly help us improve the autofix system. Thank you! 🙏

                        If there are any questions, please reach out to the [AI/ML Team](https://github.com/orgs/getsentry/teams/machine-learning-ai) on [#proj-autofix](https://sentry.slack.com/archives/C06904P7Z6E)

                        ### 🤓 Stats for the nerds:
                        Prompt tokens: **{prompt_tokens}**
                        Completion tokens: **{completion_tokens}**
                        Total tokens: **{total_tokens}**"""
                    ).format(
                        user_line=(
                            f"\nTriggered by {request.invoking_user.display_name}"
                            if request.invoking_user
                            else ""
                        ),
                        ref_note=ref_note,
                        description=description,
                        steps="\n".join([f"{i + 1}. {step.title}" for i, step in enumerate(steps)]),
                        prompt_tokens=state.usage.prompt_tokens,
                        completion_tokens=state.usage.completion_tokens,
                        total_tokens=state.usage.total_tokens,
                    )

                    pr = codebase.repo_client.create_pr_from_branch(
                        branch_ref, pr_title, pr_description
                    )

                    diff, diff_str = codebase.get_file_patches()
                    prs.append(
                        PullRequestResult(
                            pr_number=pr.number,
                            pr_url=pr.html_url,
                            repo=RepoDefinition(
                                provider=codebase.repo_client.provider,
                                owner=codebase.repo_client.repo_owner,
                                name=codebase.repo_client.repo_name,
                            ),
                            diff=diff,
                            diff_str=diff_str,
                        )
                    )
                else:
                    # Dry no-commit run
                    diff, diff_str = codebase.get_file_patches()
                    prs.append(
                        PullRequestResult(
                            pr_number=0,
                            pr_url="https://dry-run",
                            repo=RepoDefinition(
                                provider=codebase.repo_client.provider,
                                owner=codebase.repo_client.repo_owner,
                                name=codebase.repo_client.repo_name,
                            ),
                            diff=diff,
                            diff_str=diff_str,
                        )
                    )

                # TODO: Support more than 1 PR
                return prs

        return prs

    def _get_slug_id(self, organization_id: int) -> str | None:
        slug: str | None = None
        try:
            response = self.context.sentry_client.call(
                "get_organization_slug", org_id=organization_id
            )
            slug = None if response is None else response.get("slug", None)
            if slug == None:
                autofix_logger.warn(
                    f"Slug lookup call for organization {organization_id} succeeded but returned value None."
                )
        except Exception as e:
            autofix_logger.warn(f"Failed to get slug for organization {organization_id}")
            autofix_logger.exception(e)
            sentry_sdk.capture_exception(e)
            slug = None
        return slug

    @traceable(name="Executor with Retriever", run_type="llm")
    def run_executor_with_retriever(
        self,
        retriever: RetrieverComponent,
        executor: ExecutorComponent,
        step: PlanStep,
        event_details: EventDetails,
    ):
        retriever_output = retriever.invoke(
            RetrieverRequest(
                text=step.text,
            )
        )

        executor.invoke(
            ExecutorRequest(
                event_details=event_details,
                retriever_dump=(
                    retriever_output.to_xml().to_prompt_str() if retriever_output else None
                ),
                task=step.text,
            )
        )
