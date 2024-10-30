import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ContextManager, Optional

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.models import Message, Usage
from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.component import create_insight_output
from seer.automation.autofix.components.insight_sharing.models import InsightSharingRequest
from seer.automation.autofix.models import AutofixContinuation, AutofixStatus, DefaultStep
from seer.automation.state import State
from seer.bootup import module
from seer.configuration import configuration_module

logger = logging.getLogger(__name__)


class AutofixAgent(LlmAgent):
    def __init__(
        self,
        config: AgentConfig,
        context: AutofixContext,
        tools: Optional[list[FunctionTool]] = None,
        memory: Optional[list[Message]] = None,
        name: str = "Agent",
    ):
        super().__init__(config=config, tools=tools, memory=memory, name=name)
        self.context = context
        self.executor = ThreadPoolExecutor(max_workers=1, initializer=self.load_thread_context)

    @property
    def queued_user_messages(self):
        state = self.context.state.get()
        if state.steps:
            return state.steps[-1].queued_user_messages
        return []

    @queued_user_messages.setter
    def queued_user_messages(self, value: list[str]):
        with self.context.state.update() as cur:
            if not cur.steps:
                raise ValueError("No step to set queued user messages for")
            cur.steps[-1].queued_user_messages = value

    def should_continue(self, run_config: RunConfig) -> bool:
        if self.context.state.get().steps[-1].status == AutofixStatus.WAITING_FOR_USER_RESPONSE:
            return False

        return super().should_continue(run_config)

    def run_iteration(self, run_config: RunConfig):
        logger.debug(f"----[{self.name}] Running Iteration {self.iterations}----")

        # Use any queued user messages
        if self.config.interactive:
            self.use_user_messages()

        completion = self.get_completion(run_config)

        # interrupt if user message is queued and awaiting handling
        if self.queued_user_messages:
            self.use_user_messages()
            return

        self.memory.append(completion.message)

        # log thoughts to the user
        cur = self.context.state.get()
        if (
            completion.message.content
            and self.config.interactive
            and not cur.request.options.disable_interactivity
        ):
            text_before_tag = completion.message.content.split("<")[0]
            text = text_before_tag
            if text:
                cur_step_idx = len(cur.steps) - 1
                if cur_step_idx >= 0 and isinstance(cur.steps[-1], DefaultStep):

                    def get_cur_step(state: AutofixContinuation):
                        return state.steps[cur_step_idx]

                    def get_usage(state: AutofixContinuation):
                        return state.usage

                    self.executor.submit(
                        self.share_insights,
                        text,
                        self.context.state.lens(get_cur_step),
                        self.context.state.lens(get_usage),
                        len(self.memory) - 1,
                    )

        # call any tools the model wants to use
        if completion.message.tool_calls:
            for tool_call in completion.message.tool_calls:
                tool_response = self.call_tool(tool_call)
                self.memory.append(tool_response)

        self.iterations += 1
        self.usage += completion.metadata.usage

        if run_config.memory_storage_key:
            self.context.store_memory(run_config.memory_storage_key, self.memory)

        return self.memory

    @contextlib.contextmanager
    def manage_run(self) -> ContextManager[Any]:
        with super().manage_run(), self.executor:
            yield

    def use_user_messages(self):
        # adds any queued user messages to the memory
        user_msgs = self.queued_user_messages
        if user_msgs:
            # enforce alternating user/assistant messages
            msg = "\n".join(user_msgs)
            for item in reversed(self.memory):
                if item.role == "user":
                    self.memory.append(Message(content=".", role="assistant"))
                    break
                elif item.role == "assistant":
                    break
            self.memory.append(Message(content=msg, role="user"))

            self.queued_user_messages = []
            self.context.event_manager.add_log(
                "Thanks for the input. I'm thinking through it now..."
            )

    def load_thread_context(self):
        module.enable()
        configuration_module.enable()

    def share_insights(
        self,
        text: str,
        step_state: State[DefaultStep],
        usage_state: State[Usage],
        generated_at_memory_index: int,
    ):
        step = step_state.get()
        past_insights = step.get_all_insights()
        insight_card = create_insight_output(
            usage_state,
            InsightSharingRequest(
                latest_thought=text,
                memory=self.memory,
                task_description=step.description,
                past_insights=past_insights,
                generated_at_memory_index=generated_at_memory_index,
            ),
        )

        if insight_card:
            with step_state.update() as step:
                step.insights.append(insight_card)
