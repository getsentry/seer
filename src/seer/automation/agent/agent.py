import contextlib
import json
import logging
from typing import Optional

from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel, Field

from seer.automation.agent.client import LlmClient, LlmProvider
from seer.automation.agent.models import Message, ToolCall, Usage
from seer.automation.agent.tools import ClaudeTool, FunctionTool
from seer.automation.agent.utils import parse_json_with_keys
from seer.automation.utils import AgentError
from seer.dependency_injection import inject, injected

logger = logging.getLogger("autofix")


class AgentConfig(BaseModel):
    interactive: bool = False  # enables interactive user-facing features

    class Config:
        validate_assignment = True


class RunConfig(BaseModel):
    system_prompt: str | None = None
    prompt: str | None = None
    stop_message: str | None = Field(
        default=None, description="Message that signals the agent to stop"
    )
    max_iterations: int = Field(
        default=16, description="Maximum number of iterations the agent can perform"
    )
    max_tokens: int | None = None
    model: LlmProvider
    memory_storage_key: str | None = None
    temperature: float | None = 0.0
    run_name: str | None = None
    reasoning_effort: str | None = None


class LlmAgent:
    @inject
    def __init__(
        self,
        config: AgentConfig,
        client: LlmClient = injected,
        tools: Optional[list[FunctionTool | ClaudeTool]] = None,
        memory: Optional[list[Message]] = None,
        name: str = "Agent",
    ):
        self.config = config
        self.client = client
        self.tools = tools or []
        self.memory = memory or []
        self.usage = Usage()
        self.name = name
        self.iterations = 0

    def get_completion(self, run_config: RunConfig):
        return self.client.generate_text(
            messages=self.memory,
            model=run_config.model,
            system_prompt=run_config.system_prompt if run_config.system_prompt else None,
            tools=(self.tools if len(self.tools) > 0 else None),
            temperature=run_config.temperature or 0.0,
            reasoning_effort=run_config.reasoning_effort,
        )

    def run_iteration(self, run_config: RunConfig):
        logger.debug(f"----[{self.name}] Running Iteration {self.iterations}----")

        completion = self.get_completion(run_config)

        self.memory.append(completion.message)

        # call any tools the model wants to use
        if completion.message.tool_calls:
            for tool_call in completion.message.tool_calls:
                tool_response = self.call_tool(tool_call)
                self.memory.append(tool_response)

        self.iterations += 1
        self.usage += completion.metadata.usage

        return self.memory

    def should_continue(self, run_config: RunConfig) -> bool:
        # If this is the first iteration or there are no messages, continue
        if self.iterations == 0 or not self.memory:
            return True

        # Stop if we've reached the maximum number of iterations
        if self.iterations >= run_config.max_iterations:
            return False

        last_message = self.memory[-1]
        if last_message and last_message.role in ["assistant", "model"]:
            if last_message.content:
                # Stop if the stop message is found in the content
                if run_config.stop_message and run_config.stop_message in last_message.content:
                    return False
                # Stop if there are no tool calls
                if not last_message.tool_calls:
                    return False

        # Continue in all other cases
        return True

    @observe(name="Agent Run")
    def run(self, run_config: RunConfig):
        if run_config.run_name:
            langfuse_context.update_current_observation(name=run_config.run_name + " - Agent Run")
        elif self.name:
            langfuse_context.update_current_observation(name=self.name + " - Agent Run")

        if run_config.prompt:
            self.add_user_message(run_config.prompt)

        logger.debug(f"----[{self.name}] Running Agent----")

        with self.manage_run():
            while self.should_continue(run_config):
                self.run_iteration(run_config=run_config)

                if self.iterations >= run_config.max_iterations:
                    raise MaxIterationsReachedException(
                        f"Agent {self.name} reached maximum iterations without finishing."
                    )

        return self.get_last_assistant_message_content()

    @contextlib.contextmanager
    def manage_run(self):
        self.iterations = 0
        yield

    def add_user_message(self, content: str):
        self.memory.append(Message(role="user", content=content))

    def get_last_assistant_message_content(self) -> str | None:
        return (
            self.memory[-1].content if self.memory and self.memory[-1].role == "assistant" else None
        )

    def call_tool(self, tool_call: ToolCall) -> Message:
        logger.debug(f"[{tool_call.id}] Calling tool {tool_call.function}")

        tool = self.get_tool_by_name(tool_call.function)
        kwargs = self.parse_tool_arguments(tool, tool_call.args)
        tool_result = tool.call(
            **kwargs, tool_call_id=tool_call.id, current_memory_index=max(0, len(self.memory) - 1)
        )

        return Message(
            role="tool",
            content=tool_result,
            tool_call_id=tool_call.id,
            tool_call_function=tool_call.function,
        )

    def get_tool_by_name(self, name: str) -> FunctionTool | ClaudeTool:
        try:
            return next(tool for tool in self.tools if tool.name == name)
        except StopIteration:
            raise AgentError() from ValueError(f"Invalid tool name: {name}")

    def parse_tool_arguments(self, tool: FunctionTool | ClaudeTool, args: str) -> dict:
        if isinstance(tool, FunctionTool):
            try:
                return parse_json_with_keys(
                    args,
                    [param["name"] for param in tool.parameters if isinstance(param["name"], str)],
                )
            except Exception as e:
                raise AgentError() from ValueError(
                    f"Invalid tool arguments: {args}\nException: {e}"
                )
        elif isinstance(tool, ClaudeTool):
            return json.loads(args)

    def process_tool_calls(self, tool_calls: list[ToolCall]):
        for tool_call in tool_calls:
            tool_response = self.call_tool(tool_call)
            self.memory.append(tool_response)

    def update_usage(self, usage: Usage):
        self.usage += usage

    def process_message(self, message: Message):
        self.memory.append(message)

        if message.tool_calls:
            self.process_tool_calls(message.tool_calls)

        self.iterations += 1


class MaxIterationsReachedException(Exception):
    pass
