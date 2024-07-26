from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from celery_app.config import CeleryQueues
from seer.automation.autofix.components.executor.component import ExecutorComponent
from seer.automation.autofix.components.executor.models import ExecutorRequest
from seer.automation.autofix.components.planner.models import (
    CreateFilePromptXml,
    PlanningOutput,
    ReplaceCodePromptXml,
)
from seer.automation.autofix.components.retriever import RetrieverComponent, RetrieverRequest
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import AutofixStatus
from seer.automation.autofix.steps.change_describer_step import (
    AutofixChangeDescriberRequest,
    AutofixChangeDescriberStep,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.codebase.models import BaseDocument
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineChain, PipelineStepTaskRequest


class AutofixExecutionStepRequest(PipelineStepTaskRequest):
    task_index: int
    planning_output: PlanningOutput


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def autofix_execution_task(*args, request: dict[str, Any]):
    AutofixExecutionStep(request).invoke()


class AutofixExecutionStep(AutofixPipelineStep, PipelineChain):
    """
    This class represents the execution pipeline in the autofix system. It is responsible for
    executing the fixes suggested by the planning component based on the root cause analysis.
    """

    name = "AutofixExecutionStep"
    request: AutofixExecutionStepRequest

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> AutofixExecutionStepRequest:
        return AutofixExecutionStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return autofix_execution_task

    @observe(name="Autofix - Execution Step")
    @ai_track(description="Autofix - Execution Step")
    def _invoke(self, **kwargs):
        retriever = RetrieverComponent(self.context)
        executor = ExecutorComponent(self.context)
        task = self.request.planning_output.tasks[self.request.task_index]

        self.context.event_manager.send_execution_step_start(self.request.task_index)

        document: BaseDocument | None = None
        retriever_dump: str | None = None

        if isinstance(task, ReplaceCodePromptXml):
            # For replace code tasks, we just need to retrieve the document.
            contents = self.context.get_file_contents(task.file_path, repo_name=task.repo_name)
            if contents:
                document = BaseDocument(
                    path=task.file_path,
                    text=contents,
                )
        elif isinstance(task, CreateFilePromptXml):
            # For create file tasks, we need to find relevant context for the new file.
            retriever_output = retriever.invoke(RetrieverRequest(text=task.to_prompt_str()))
            if retriever_output:
                retriever_dump = retriever_output.to_xml().to_prompt_str()

        event_details = EventDetails.from_event(self.context.state.get().request.issue.events[0])

        executor.invoke(
            ExecutorRequest(
                event_details=event_details,
                retriever_dump=retriever_dump,
                documents=[document] if document else [],
                task=task.to_prompt_str(),
                repo_name=task.repo_name,
            )
        )

        self.context.event_manager.send_execution_step_result(
            self.request.task_index, AutofixStatus.COMPLETED
        )

        if self.request.task_index == len(self.request.planning_output.tasks) - 1:
            self.next(
                AutofixChangeDescriberStep.get_signature(
                    AutofixChangeDescriberRequest(**self.step_request_fields)
                ),
                queue=CeleryQueues.DEFAULT,
            )
        else:
            self.next(
                AutofixExecutionStep.get_signature(
                    AutofixExecutionStepRequest(
                        **self.step_request_fields,
                        task_index=self.request.task_index + 1,
                        planning_output=self.request.planning_output,
                    )
                ),
                queue=CeleryQueues.CUDA,
            )
