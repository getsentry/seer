import contextlib
import logging
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Callable, Optional

import sentry_sdk
from langfuse.decorators import langfuse_context, observe

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
from seer.utils import backoff_on_exception, retry_once_with_modified_input

logger = logging.getLogger(__name__)

MAX_PARALLEL_TOOL_CALLS = 3


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

        target_step_id = None
        for step in reversed(cur.steps):
            if isinstance(step, DefaultStep) and step.id:
                target_step_id = step.id
                break

        if target_step_id:
            self.futures.append(
                self.executor.submit(
                    self.share_insights,
                    text,
                    self.context.state,
                    max(0, len(self.memory) - 1),
                    target_step_id,
                    langfuse_parent_trace_id=trace_id,  # type: ignore
                    langfuse_parent_observation_id=observation_id,  # type: ignore
                )
            )

    def _omit_biggest_n_tool_messages(self, messages: list[Message], n: int) -> list[Message]:
        """Creates a new list of messages with content of the n largest tool results ommitted."""
        tool_messages_with_size = [
            (i, len(msg.content or ""))
            for i, msg in enumerate(messages)
            if msg.role == "tool" and msg.content
        ]
        tool_messages_with_size.sort(key=lambda x: x[1], reverse=True)
        indices_to_truncate = {idx for idx, size in tool_messages_with_size[:n]}

        if indices_to_truncate:
            new_messages = []
            for i, msg in enumerate(messages):
                if i in indices_to_truncate:
                    modified_msg = msg.model_copy(deep=True)
                    modified_msg.content = "[omitted for brevity]"
                    new_messages.append(modified_msg)
                else:
                    new_messages.append(msg)
            return new_messages

        return messages

    def _get_completion(self, run_config: RunConfig, messages: list[Message]):
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
            messages=messages,
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
                    if chunk[0] == "thinking_content" or chunk[0] == "content":
                        if not cleared and len(chunk[1]) > 0:
                            cur_step.clear_output_stream()
                            cleared = True
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

        Additionally, if the input is too long, the tool messages are truncated and the completion is retried once.
        """

        def _long_wait_logger():
            self.context.event_manager.add_log(
                "Our LLM provider is overloaded. Now retrying desperately..."
            )

        is_exception_retryable = getattr(
            run_config.model, "is_completion_exception_retryable", lambda _: False
        )
        retrier = backoff_on_exception(
            is_exception_retryable,
            max_tries=max_tries,
            sleep_sec_scaler=sleep_sec_scaler,
            long_wait_callback=_long_wait_logger,
            long_wait_threshold_sec=10,
        )

        is_input_too_long_func = getattr(run_config.model, "is_input_too_long", lambda _: False)
        retry_input_too_long_decorator = retry_once_with_modified_input(
            in_input_bad=is_input_too_long_func,
            modify_input_func=lambda msgs: self._omit_biggest_n_tool_messages(msgs, 3),
            target_kwarg_name="messages_to_send",
        )

        def attempt_completion(messages_to_send: list[Message]):
            return self._get_completion(run_config, messages=messages_to_send)

        final_attempt_func = retry_input_too_long_decorator(retrier(attempt_completion))

        return final_attempt_func(messages_to_send=self.memory)

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
            for i, tool_call in enumerate(completion.message.tool_calls):
                duplicate_found = False
                for msg in self.memory[:-1]:
                    if msg.tool_calls:
                        if any(
                            tool_call.function == existing_tool_call.function
                            and self.parse_tool_arguments(
                                self.get_tool_by_name(tool_call.function), tool_call.args
                            )
                            == self.parse_tool_arguments(
                                self.get_tool_by_name(existing_tool_call.function),
                                existing_tool_call.args,
                            )
                            and tool_call.id != existing_tool_call.id
                            for existing_tool_call in msg.tool_calls
                        ):
                            duplicate_found = True
                            break
                if duplicate_found:
                    self.memory.append(
                        Message(
                            role="tool",
                            content="You've already called this tool with the same arguments, so it was not called again. Please refer to your old results. If you are struggling to find what you need, try using a different tool.",
                            tool_call_id=tool_call.id,
                            tool_call_function=tool_call.function,
                        )
                    )
                elif i < MAX_PARALLEL_TOOL_CALLS:  # only allowing up to 3 simultaneous tool calls
                    tool_response = self.call_tool(tool_call)
                    self.memory.append(tool_response)
                else:
                    self.memory.append(
                        Message(
                            role="tool",
                            content=f"This tool was not called as you cannot call more than {MAX_PARALLEL_TOOL_CALLS} tools at a time.",
                            tool_call_id=tool_call.id,
                            tool_call_function=tool_call.function,
                        )
                    )

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
    @sentry_sdk.trace
    def share_insights(
        self,
        text: str,
        state: State[AutofixContinuation],
        generated_at_memory_index: int,
        step_id: str,
    ):
        step = state.get().find_step(id=step_id)

        if not step or not isinstance(step, DefaultStep):
            logger.exception(
                f"Cannot add insight to step: step not found or not a DefaultStep. Step key: {step.key if step else 'None'}"
            )
            return

        insight_card, usage = create_insight_output(
            latest_thought=text,
            past_insights=step.get_all_insights(exclude_user_messages=True),
            step_type=step.key,
            memory=self.memory,
            generated_at_memory_index=generated_at_memory_index,
            context=self.context,
        )

        if insight_card:
            self.context.event_manager.send_insight(insight_card, step_id)

        with state.update() as cur:
            cur.usage += usage
