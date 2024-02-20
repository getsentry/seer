import json
import logging
import textwrap
import xml.etree.ElementTree as ET

import sentry_sdk
from langsmith import RunTree, traceable

from celery_app.models import UpdateCodebaseTaskRequest
from seer.automation.agent.agent import GptAgent
from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.event_manager import AutofixEventManager, AutofixStatus
from seer.automation.autofix.models import (
    AutofixOutput,
    AutofixRequest,
    PlanningInput,
    PlanningOutput,
    PlanStep,
    ProblemDiscoveryOutput,
    ProblemDiscoveryRequest,
    ProblemDiscoveryResult,
    PullRequestResult,
    RepoDefinition,
    Stacktrace,
)
from seer.automation.autofix.prompts import (
    ExecutionPrompts,
    PlanningPrompts,
    ProblemDiscoveryPrompt,
)
from seer.automation.autofix.tools import BaseTools, CodeActionTools
from seer.automation.autofix.utils import escape_multi_xml, extract_xml_element_text
from seer.automation.codebase.models import DocumentChunkWithEmbedding
from seer.automation.codebase.tasks import update_codebase_index

logger = logging.getLogger("autofix")


class Autofix:
    stacktrace: Stacktrace

    def __init__(
        self,
        request: AutofixRequest,
        event_manager: AutofixEventManager,
    ):
        self.request = request
        self.usage = Usage()
        self.event_manager = event_manager

        self.context = AutofixContext(
            request.organization_id,
            request.project_id,
            request.repos,
        )

    @traceable(name="Autofix Run")
    def run(self, run_tree: RunTree):
        metadata = run_tree.extra.get("metadata", {})
        metadata["request"] = self.request.model_dump()
        try:
            logger.info(f"Beginning autofix for issue {self.request.issue.id}")

            events = self.request.issue.events
            stacktrace = events[-1].get_stacktrace() if events else None
            if not stacktrace:
                logger.warning(f"No stacktrace found for issue {self.request.issue.id}")

                self.event_manager.send_no_stacktrace_error()
                return
            self.stacktrace = stacktrace

            self.event_manager.send_initial_steps()

            problem_discovery_output = self.run_problem_discovery_agent()

            if not problem_discovery_output:
                logger.warning(f"Problem discovery agent did not return a valid response")
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

            self.event_manager.send_problem_discovery_result(problem_discovery_payload)

            if problem_discovery_payload.status == "CANCELLED":
                logger.info(f"Problem is not actionable")
                return

            for repo in self.request.repos:
                if not self.context.has_codebase_index(repo):
                    logger.info(f"Creating codebase index for repo {repo.repo_name}")
                    self.event_manager.send_codebase_creation_message()
                    with sentry_sdk.start_span(
                        op="seer.automation.autofix.codebase_index.create",
                        description="Create codebase index",
                    ) as span:
                        span.set_tag("repo", repo.repo_name)
                        self.context.create_codebase_index(repo)
                    logger.info(f"Codebase index created for repo {repo.repo_name}")

            for repo_id, codebase in self.context.codebases.items():
                if codebase.is_behind():
                    if self.context.diff_contains_stacktrace_files(repo_id, self.stacktrace):
                        logger.debug(
                            f"Waiting for codebase index update for repo {codebase.repo_info.external_slug}"
                        )
                        self.event_manager.send_codebase_indexing_message()
                        with sentry_sdk.start_span(
                            op="seer.automation.autofix.codebase_index.update",
                            description="Update codebase index",
                        ) as span:
                            span.set_tag("repo", codebase.repo_info.external_slug)
                            codebase.update()
                        logger.debug(f"Codebase index updated")
                    else:
                        update_codebase_index.apply_async(
                            (UpdateCodebaseTaskRequest(repo_id=repo_id).model_dump(),),
                            countdown=3 * 60,
                        )  # 3 minutes
                        logger.info(f"Codebase indexing scheduled for later")
                else:
                    logger.debug(f"Codebase is up to date")
                    self.event_manager.send_codebase_indexing_result("COMPLETED")

            if not self.context.codebases:
                logger.warning(f"No codebase indexes")
                sentry_sdk.capture_message(
                    f"No codebases found for organization {self.request.organization_id} and project {self.request.project_id}'s repos: {', '.join([repo.repo_name for repo in self.request.repos])}"
                )
                self.event_manager.mark_running_steps_errored()
                self.event_manager.send_autofix_complete(None)
                return

            self.event_manager.send_codebase_indexing_result(AutofixStatus.COMPLETED)

            self.context.process_stacktrace(self.stacktrace)

            try:
                planning_output = self.run_planning_agent(
                    PlanningInput(problem=problem_discovery_output)
                )

                if not planning_output:
                    logger.warning(f"Planning agent did not return a valid response")
                    # self.event_manager.send_planning_result(None)
                    return

                # self.event_manager.send_planning_result(planning_output)

                logger.info(
                    f"Planning complete; there are {len(planning_output.steps)} steps in the plan to execute."
                )
            except Exception as e:
                logger.error(f"Failed to plan: {e}")
                sentry_sdk.capture_exception(e)
                # self.event_manager.send_planning_result(None)
                return

            for i, step in enumerate(planning_output.steps):
                # self.event_manager.send_execution_step_start(step.id)

                logger.info(f"Executing step: {i}/{len(planning_output.steps)}")
                self.run_execution_agent(step)

                # self.event_manager.send_execution_step_result(step.id, AutofixStatus.COMPLETED)

            logger.debug(
                "File changes:",
                [codebase.file_changes for codebase in self.context.codebases.values()],
            )

            prs = self._create_prs(
                planning_output.title, planning_output.description, planning_output.steps
            )

            outputs = []
            if prs:
                # TODO: Support more than 1 PR...
                pr = prs[0]
                output = AutofixOutput(
                    title=planning_output.title,
                    description=planning_output.description,
                    pr_url=pr.pr_url,
                    repo_name=f"{pr.repo.repo_owner}/{pr.repo.repo_name}",
                    pr_number=pr.pr_number,
                    usage=self.usage,
                )
                self.event_manager.send_autofix_complete(output)
                outputs.append(output)

                metadata.setdefault("prs", []).extend([pr.model_dump() for pr in prs])

            file_changes = {}
            for repo_id, codebase in self.context.codebases.items():
                file_changes[repo_id] = codebase.file_changes

            return {"outputs": outputs, "prs": prs, "file_changes": file_changes}
        except Exception as e:
            logger.error(f"Failed to complete autofix")
            logger.exception(e)
            sentry_sdk.capture_exception(e)

            self.event_manager.mark_running_steps_errored()
            self.event_manager.send_autofix_complete(None)
        finally:
            self.context.cleanup()
            logger.info(f"Autofix complete for issue {self.request.issue.id}")

    def _create_prs(self, title: str, description: str, steps: list[PlanStep]):
        prs: list[PullRequestResult] = []
        for codebase in self.context.codebases.values():
            if codebase.file_changes:
                branch_ref = codebase.repo_client.create_branch_from_changes(
                    pr_title=title, file_changes=codebase.file_changes
                )

                if branch_ref is None:
                    logger.warning(f"Failed to create branch from changes")
                    return None

                pr = codebase.repo_client.create_pr_from_branch(
                    branch_ref, title, description, steps, self.request.issue.id, self.usage
                )

                prs.append(
                    PullRequestResult(
                        pr_number=pr.number,
                        pr_url=pr.html_url,
                        repo=RepoDefinition(
                            repo_provider=codebase.repo_client.provider,
                            repo_owner=codebase.repo_client.repo_owner,
                            repo_name=codebase.repo_client.repo_name,
                        ),
                    )
                )

                # TODO: Support more than 1 PR
                return prs

        return prs

    def _parse_problem_discovery_response(self, response: str) -> ProblemDiscoveryOutput | None:
        try:
            problem_el = ET.fromstring(
                f"<response>{escape_multi_xml(response, ['description', 'reasoning', 'actionability_score'])}</response>"
            ).find("problem")

            if problem_el is None:
                logger.warning(f"Problem discovery response does not contain a problem: {response}")
                return None

            description = extract_xml_element_text(problem_el, "description")

            reasoning = extract_xml_element_text(problem_el, "reasoning")

            actionability_score = float(
                extract_xml_element_text(problem_el, "actionability_score") or 0.0
            )

            if description is None or reasoning is None:
                logger.warning(
                    f"Problem discovery response does not contain a description, reasoning, or actionability score: {response}"
                )
                return None

            output = ProblemDiscoveryOutput(
                description=description,
                reasoning=reasoning,
                actionability_score=actionability_score,
            )

            return output
        except ET.ParseError as e:
            logger.warning(f"Problem discovery response is not valid XML: {e}")

        return None

    @traceable(run_type="llm", name="Problem Discovery Agent")
    def run_problem_discovery_agent(
        self, request: ProblemDiscoveryRequest | None = None
    ) -> ProblemDiscoveryOutput | None:
        problem_discovery_agent = GptAgent(
            name="problem-discovery",
            memory=[
                Message(
                    role="system",
                    content=ProblemDiscoveryPrompt.format_system_msg(
                        err_msg=self.request.issue.title,
                        stack_str=self.stacktrace.to_str(),
                    ),
                )
            ],
        )

        if request:
            # message = ProblemDiscoveryPrompt.format_feedback_msg(
            #     request.message, request.previous_output
            # )
            raise NotImplementedError("Problem discovery feedback not implemented yet.")
        else:
            message = ProblemDiscoveryPrompt.format_default_msg(
                additional_context=self.request.additional_context
            )

        problem_discovery_response = problem_discovery_agent.run(message)

        self.usage += problem_discovery_agent.usage

        if problem_discovery_response is None:
            logger.warning(f"Problem discovery agent did not return a valid response")
            return None

        return self._parse_problem_discovery_response(problem_discovery_response)

    def _parse_planning_output(self, response: str) -> PlanningOutput | None:
        parsed_output = ET.fromstring(
            f'<response>{escape_multi_xml(response, ["title", "description", "step"])}</response>'
        )

        try:
            title_element = parsed_output.find(".//*title")
            description_element = parsed_output.find(".//*description")
            steps_element = parsed_output.find(".//*steps")

            if steps_element is None:
                logger.warning(f"Planning response does not contain steps element: {response}")
                return None

            step_elements = steps_element.findall("step")

            if len(step_elements) == 0:
                logger.warning(f"Planning response does not contain any steps: {response}")
                return None

            steps = [
                PlanStep(
                    id=i,
                    title=step_element.attrib["title"],
                    text=textwrap.dedent(step_element.text or ""),
                )
                for i, step_element in enumerate(step_elements)
            ]

            title = textwrap.dedent(title_element.text or "") if title_element is not None else None
            description = (
                textwrap.dedent(description_element.text or "")
                if description_element is not None
                else None
            )

            if title is None or description is None:
                logger.warning(
                    f"Planning response does not contain a title, description, or plan: {response}"
                )
                return None

            return PlanningOutput(title=title, description=description, steps=steps)
        except AttributeError as e:
            logger.warning(f"Planning response does not contain a title, description, or plan: {e}")
            return None

    @traceable(run_type="llm", name="Planning Agent")
    def run_planning_agent(self, input: PlanningInput) -> PlanningOutput | None:
        assert (
            input.message or input.previous_output or input.problem
        ), "PlanningInput requires at least one of the fields: 'message', 'previous_output', or 'problem'."

        planning_agent_tools = BaseTools(self.context)

        planning_agent = GptAgent(
            name="planner",
            tools=planning_agent_tools.get_tools(),
            memory=[
                Message(
                    role="system",
                    content=PlanningPrompts.format_system_msg(
                        err_msg=self.request.issue.title,
                        stack_str=self.stacktrace.to_str(),
                    ),
                )
            ],
        )

        if input.message:
            message = ""
        elif input.problem:
            # TODO: Remove this and also find how to address mismatches in the stack trace path and the actual filepaths
            message = PlanningPrompts.format_default_msg(
                problem=input.problem,
                additional_context=self.request.additional_context or "",
            )
        else:
            raise ValueError(
                "PlanningInput requires at least one of the fields: 'message' or 'problem'."
            )

        planning_response = planning_agent.run(message)

        self.usage += planning_agent.usage

        if planning_response is None:
            logger.warning(f"Planning agent did not return a valid response")
            return None

        return self._parse_planning_output(planning_response)

    @traceable(run_type="retriever", name="Execution Agent Plan Step Context Fetcher")
    def _get_plan_step_context(self, plan_item: PlanStep):
        logger.debug(f"Getting context for plan item: {plan_item}")

        def json_parser(x) -> list[str] | None:
            return json.loads(x) if x else None

        # Identify good search queries for the plan item
        queries, message, usage = GptClient().completion_with_parser(
            model="gpt-4-0125-preview",
            messages=[
                Message(role="system", content=PlanningPrompts.format_plan_item_query_system_msg()),
                Message(
                    role="user",
                    content=PlanningPrompts.format_plan_item_query_default_msg(plan_item=plan_item),
                ),
            ],
            parser=json_parser,
        )

        self.usage += usage

        logger.debug(f"Search queries: {queries}")

        if not queries:
            logger.warning(f"No search queries found for plan item: {plan_item}")
            return ""

        context_dump = ""
        unique_chunks: dict[str, DocumentChunkWithEmbedding] = {}
        for query in queries:
            retrived_chunks = self.context.query(query, top_k=2)
            for chunk in retrived_chunks:
                unique_chunks[chunk.hash] = chunk
        chunks = list(unique_chunks.values())

        logger.debug(f"Retrieved {len(chunks)} unique chunks.")

        for chunk in chunks:
            context_dump += f"\n\n{chunk.get_dump_for_llm(self.context.get_codebase(chunk.repo_id).repo_info.external_slug)}"

        return context_dump

    @traceable(run_type="llm", name="Execution Agent")
    def run_execution_agent(self, plan_item: PlanStep):
        # TODO: make this more robust
        try:
            context_dump = self._get_plan_step_context(plan_item)
        except Exception as e:
            logger.error(f"Failed to get context for plan item: {e}")
            sentry_sdk.capture_exception(e)
            context_dump = ""

        context_dump = ""

        code_action_tools = CodeActionTools(
            self.context,
        )

        execution_agent = GptAgent(
            name="executor",
            tools=code_action_tools.get_tools(),
            memory=[
                Message(
                    role="system",
                    content=ExecutionPrompts.format_system_msg(
                        context_dump,
                        error_message=self.request.issue.title,
                        stack_trace=self.stacktrace.to_str(),
                    ),
                ),
            ],
            stop_message="<DONE>",
        )

        execution_agent.run(ExecutionPrompts.format_default_msg(plan_item=plan_item))

        self.usage += execution_agent.usage
