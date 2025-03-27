import contextlib
import logging
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Callable, Optional

from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.models import (
    LlmGenerateTextResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Message,
    ToolCall,
    Usage,
)
from seer.automation.agent.tools import ClaudeTool, FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.component import create_insight_output
from seer.automation.autofix.models import AutofixContinuation, AutofixStatus, DefaultStep
from seer.automation.state import State
from seer.dependency_injection import copy_modules_initializer
from seer.utils import backoff_on_exception

logger = logging.getLogger(__name__)


class AutofixAgent(LlmAgent):
    futures: list[Future]
    executor: Executor
    accumulated_thinking_chunks: list[str]

    def __init__(
        self,
        config: AgentConfig,
        context: AutofixContext,
        tools: Optional[list[FunctionTool | ClaudeTool]] = None,
        memory: Optional[list[Message]] = None,
        name: str = "Agent",
    ):
        super().__init__(config=config, tools=tools, memory=memory, name=name)
        self.context = context
        self.accumulated_thinking_chunks = []

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

    def _submit_insight(self, text: str):
        """
        Helper method to submit an insight for processing in parallel.

        Args:
            text: The text to process for insights
        """
        # Only submit if we have meaningful content and interactivity is enabled
        if not text.strip():
            return

        cur = self.context.state.get()
        if not self.config.interactive or cur.request.options.disable_interactivity:
            return

        # Get trace and observation IDs for langfuse
        trace_id = langfuse_context.get_current_trace_id()
        observation_id = langfuse_context.get_current_observation_id()

        # Share insights in parallel
        cur_step_idx = len(cur.steps) - 1
        self.futures.append(
            self.executor.submit(
                self.share_insights,
                text,
                cur_step_idx,
                self.context.state,
                max(0, len(self.memory) - 1),
                langfuse_parent_trace_id=trace_id,  # type: ignore
                langfuse_parent_observation_id=observation_id,  # type: ignore
            )
        )

    def _get_completion(self, run_config: RunConfig):
        """
        Streams the preliminary output to the current step and only returns when output is complete
        """
        content_chunks = []
        thinking_content_chunks = []
        thinking_signature = None
        tool_calls = []
        usage = Usage()

        # Reset accumulated thinking chunks for this completion
        self.accumulated_thinking_chunks = []

        stream = self.client.generate_text_stream(
            messages=self.memory,
            model=run_config.model,
            system_prompt=run_config.system_prompt if run_config.system_prompt else None,
            tools=(self.tools if len(self.tools) > 0 else None),
            temperature=run_config.temperature or 0.0,
            reasoning_effort=run_config.reasoning_effort,
            first_token_timeout=(
                90.0
                if (run_config.model.provider_name == LlmProviderType.OPENAI)
                and run_config.model.model_name.startswith("o")
                else 40.0
            ),
            max_tokens=run_config.max_tokens,
        )

        cleared = False
        for chunk in stream:
            if self.queued_user_messages:  # user interruption
                return

            if isinstance(chunk, tuple):
                with self.context.state.update() as cur:
                    cur_step = cur.steps[-1]
                    if not cleared:
                        cur_step.clear_output_stream()
                        cleared = True
                    if chunk[0] == "thinking_content" or chunk[0] == "content":
                        cur_step.receive_output_stream(chunk[1])
                if chunk[0] == "thinking_content":
                    thinking_content_chunks.append(chunk[1])

                    # Accumulate thinking chunks for insight sharing
                    self.accumulated_thinking_chunks.append(chunk[1])

                    # Check if we have enough accumulated content to share insights
                    # Only share insights if we have at least 1500 characters and the accumulated text contains a newline
                    accumulated_text = "".join(self.accumulated_thinking_chunks)

                    if (
                        len(accumulated_text) >= 1500
                        and "\n"
                        in accumulated_text  # Only share when we have at least one complete line
                    ):
                        # Reset for next batch
                        text_to_process = accumulated_text
                        self.accumulated_thinking_chunks = []

                        # Submit for insight processing
                        self._submit_insight(text_to_process)

                elif chunk[0] == "thinking_signature":
                    thinking_signature = chunk[1]

                    # If we have accumulated thinking content and received the thinking signature,
                    # share insights regardless of content length
                    if self.accumulated_thinking_chunks:
                        text_to_process = "".join(self.accumulated_thinking_chunks)
                        self.accumulated_thinking_chunks = []

                        # Submit for insight processing
                        self._submit_insight(text_to_process)

                elif chunk[0] == "content":
                    content_chunks.append(chunk[1])
            elif isinstance(chunk, ToolCall):
                tool_calls.append(chunk)
            elif isinstance(chunk, Usage):
                usage += chunk

        message = self.client.construct_message_from_stream(
            content_chunks=content_chunks,
            thinking_content_chunks=thinking_content_chunks,
            thinking_signature=thinking_signature,
            tool_calls=tool_calls,
            model=run_config.model,
        )

        return LlmGenerateTextResponse(
            message=message,
            metadata=LlmResponseMetadata(
                model=run_config.model.model_name,
                provider_name=run_config.model.provider_name,
                usage=usage,
            ),
        )

    def get_completion(
        self,
        run_config: RunConfig,
        max_tries: int = 4,
        sleep_sec_scaler: Callable[[int], float] = lambda num_tries: 2**num_tries,
    ):
        """
        Streams the preliminary output to the current step and only returns when output is complete.

        The completion request is retried `max_tries - 1` times if a retryable exception was just
        raised, e.g, Anthropic's API is overloaded.
        """
        is_exception_retryable = getattr(
            run_config.model, "is_completion_exception_retryable", lambda _: False
        )
        retrier = backoff_on_exception(
            is_exception_retryable, max_tries=max_tries, sleep_sec_scaler=sleep_sec_scaler
        )
        get_completion_retryable = retrier(self._get_completion)
        return get_completion_retryable(run_config)

    def run_iteration(self, run_config: RunConfig):
        logger.debug(f"----[{self.name}] Running Iteration {self.iterations}----")

        if self.iterations == 0 and run_config.memory_storage_key:
            self.context.store_memory(run_config.memory_storage_key, self.memory)

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
        if (
            completion.message.content  # only if we have something to summarize
            and completion.message.tool_calls  # only if the run is in progress
        ):
            text_before_tag = completion.message.content.split("<")[0]
            if text_before_tag:
                # Submit for insight processing
                self._submit_insight(text_before_tag)

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
    def manage_run(self):
        self.futures = []
        self.executor = ThreadPoolExecutor(max_workers=2, initializer=copy_modules_initializer())
        with super().manage_run(), self.executor:
            yield
        for future in self.futures:
            exc = future.exception()
            if exc is not None:
                raise exc

    def use_user_messages(self):
        cur_step = self.context.state.get().steps[-1]
        cur_step.clear_output_stream()

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
            self.context.event_manager.add_log("Got it. Initiating deep reflection...")

    @observe(name="Share Insights in parallel")
    @ai_track(description="Share Insights in parallel")
    def share_insights(
        self,
        text: str,
        cur_step_idx: int,
        state: State[AutofixContinuation],
        generated_at_memory_index: int,
    ):
        steps = state.get().steps
        if cur_step_idx >= len(steps) or not steps:
            return

        step = steps[cur_step_idx]
        if not isinstance(step, DefaultStep):
            return

        insight_card, usage = create_insight_output(
            latest_thought=text,
            past_insights=step.get_all_insights(exclude_user_messages=True),
            step_type=step.key,
            memory=self.memory,
            generated_at_memory_index=generated_at_memory_index,
        )

        with state.update() as cur:
            if insight_card:
                cur_step = cur.steps[cur_step_idx]
                assert isinstance(cur_step, DefaultStep)
                cur_step.insights.append(insight_card)
            cur.usage += usage
