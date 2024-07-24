from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, GptAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.executor.models import ExecutorOutput, ExecutorRequest
from seer.automation.autofix.components.executor.prompts import ExecutionPrompts
from seer.automation.autofix.tools import CodeActionTools
from seer.automation.component import BaseComponent


class ExecutorComponent(BaseComponent[ExecutorRequest, ExecutorOutput]):
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @observe(name="Executor")
    @ai_track(description="Executor")
    def invoke(self, request: ExecutorRequest) -> None:
        code_action_tools = CodeActionTools(self.context)

        execution_agent = GptAgent(
            config=AgentConfig(
                system_prompt=ExecutionPrompts.format_system_msg(),
                stop_message="<DONE>",
            ),
            name="Executor",
            tools=code_action_tools.get_tools(),
        )

        execution_agent.run(
            ExecutionPrompts.format_default_msg(
                retriever_dump=request.retriever_dump,
                documents=request.documents,
                repo_name=request.repo_name,
                event=request.event_details,
                task=request.task,
            ),
            context=self.context,
        )

        with self.context.state.update() as cur:
            cur.usage += execution_agent.usage
