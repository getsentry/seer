import logging
import threading
from typing import Optional

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.models import Message
from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.component import InsightSharingComponent
from seer.automation.autofix.components.insight_sharing.models import InsightSharingRequest
from seer.automation.autofix.models import AutofixStatus, DefaultStep
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
        if self.context.state.get().steps[-1].queued_user_messages:
            self.use_user_messages()
            return

        self.memory.append(completion.message)

        # log thoughts to the user
        if (
            completion.message.content
            and self.config.interactive
            and not self.context.state.get().request.options.disable_interactivity
        ):
            text_before_tag = completion.message.content.split("<")[0]
            text = text_before_tag
            if text:
                self.run_in_thread(
                    func=self.share_insights, args=(self.context, text, len(self.memory) - 1)
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

    def use_user_messages(self):
        # adds any queued user messages to the memory
        user_msgs = self.context.state.get().steps[-1].queued_user_messages
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

            with self.context.state.update() as cur:
                cur.steps[-1].queued_user_messages = []
            self.context.event_manager.add_log(
                "Thanks for the input. I'm thinking through it now..."
            )

    def run_in_thread(self, func, args):
        def wrapper():
            # ensures dependency injection works
            module.enable()
            configuration_module.enable()

            func(*args)

        threading.Thread(target=wrapper).start()

    def share_insights(self, context: AutofixContext, text: str, generated_at_memory_index: int):
        # generate insights
        insight_sharing = InsightSharingComponent(context)
        past_insights = context.state.get().get_all_insights()
        insight_card = insight_sharing.invoke(
            InsightSharingRequest(
                latest_thought=text,
                memory=self.memory,
                task_description=context.state.get().get_step_description(),
                past_insights=past_insights,
                generated_at_memory_index=generated_at_memory_index,
            )
        )
        # add the insight card to the current step
        if insight_card:
            if len(context.state.get().get_all_insights()) == len(
                past_insights
            ):  # in case something else was added in parallel, don't add this insight
                with context.state.update() as cur:
                    if cur.steps and isinstance(cur.steps[-1], DefaultStep):
                        step = cur.steps[-1]
                        step.insights.append(insight_card)
                        cur.steps[-1] = step
