from langsmith import traceable

from seer.automation.agent.agent import GptAgent
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.executor.models import ExecutorOutput, ExecutorRequest
from seer.automation.autofix.components.executor.prompts import ExecutionPrompts
from seer.automation.autofix.tools import CodeActionTools
from seer.automation.component import BaseComponent


class ExecutorComponent(BaseComponent[ExecutorRequest, ExecutorOutput]):
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @traceable(name="Executor", run_type="llm", tags=["executor:v1.1"])
    def invoke(self, request: ExecutorRequest) -> None:
        with self.context.state.update() as cur:
            code_action_tools = CodeActionTools(self.context)

            execution_agent = GptAgent(
                name="executor",
                tools=code_action_tools.get_tools(),
                memory=[
                    Message(
                        role="system",
                        content=ExecutionPrompts.format_system_msg(),
                    ),
                ],
                stop_message="<DONE>",
            )

            execution_agent.run(
                ExecutionPrompts.format_default_msg(
                    retriever_dump=request.retriever_dump,
                    error_message=request.event_details.title,
                    exceptions=request.event_details.exceptions,
                    task=request.task,
                )
            )

            cur.usage += execution_agent.usage
