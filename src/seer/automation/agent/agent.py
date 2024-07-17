import logging
from abc import ABC, abstractmethod
from typing import Optional

from openai._types import NotGiven
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, Field

from seer.automation.agent.client import GptClient, LlmClient
from seer.automation.agent.models import Message, ToolCall, Usage
from seer.automation.agent.tools import FunctionTool
from seer.automation.agent.utils import parse_json_with_keys

logger = logging.getLogger("autofix")


class AgentConfig(BaseModel):
    max_iterations: int = Field(
        default=16, description="Maximum number of iterations the agent can perform"
    )
    model: str = Field(default="gpt-4-0613", description="The model to be used by the agent")
    stop_message: Optional[str] = Field(
        default=None, description="Message that signals the agent to stop"
    )

    class Config:
        validate_assignment = True


class LlmAgent(ABC):
    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[list[FunctionTool]] = None,
        memory: Optional[list[Message]] = None,
        name: str = "Agent",
    ):
        self.config = config
        self.tools = tools or []
        self.memory = memory or []
        self.usage = Usage()
        self.name = name
        self.iterations = 0

    @abstractmethod
    def run_iteration(self):
        pass

    def should_continue(self) -> bool:
        # If this is the first iteration or there are no messages, continue
        if self.iterations == 0 or not self.memory:
            return True

        # Stop if we've reached the maximum number of iterations
        if self.iterations >= self.config.max_iterations:
            return False

        last_message = self.memory[-1]
        if last_message and last_message.role in ["assistant", "model"]:
            if last_message.content:
                # Stop if the stop message is found in the content
                if self.config.stop_message and self.config.stop_message in last_message.content:
                    return False
                # Stop if there are no tool calls
                if not last_message.tool_calls:
                    return False

        # Continue in all other cases
        return True

    def run(self, prompt: str):
        self.add_user_message(prompt)
        logger.debug(f"----[{self.name}] Running Agent----")

        while self.should_continue():
            self.run_iteration()

        if self.iterations == self.config.max_iterations:
            raise MaxIterationsReachedException(
                f"Agent {self.name} reached maximum iterations without finishing."
            )

        return self.get_last_message_content()

    def add_user_message(self, content: str):
        self.memory.append(Message(role="user", content=content))

    def get_last_message_content(self) -> str | None:
        return self.memory[-1].content if self.memory else None

    def call_tool(self, tool_call: ToolCall) -> Message:
        logger.debug(f"[{tool_call.id}] Calling tool {tool_call.function}")

        tool = self.get_tool_by_name(tool_call.function)
        kwargs = self.parse_tool_arguments(tool, tool_call.args)
        tool_result = tool.call(**kwargs)

        return Message(role="tool", content=tool_result, tool_call_id=tool_call.id)

    def get_tool_by_name(self, name: str) -> FunctionTool:
        return next(tool for tool in self.tools if tool.name == name)

    def parse_tool_arguments(self, tool: FunctionTool, args: str) -> dict:
        return parse_json_with_keys(
            args, [param["name"] for param in tool.parameters if isinstance(param["name"], str)]
        )


class GptAgent(LlmAgent):
    def __init__(
        self,
        config: AgentConfig,
        client: Optional[LlmClient] = None,
        tools: Optional[list[FunctionTool]] = None,
        memory: Optional[list[Message]] = None,
        name: str = "GptAgent",
        chat_completion_kwargs: Optional[dict] = None,
    ):
        super().__init__(config, tools, memory, name)
        self.client = client or GptClient(model=config.model)
        self.chat_completion_kwargs = chat_completion_kwargs or {}

    def run_iteration(self):
        logger.debug(f"----[{self.name}] Running Iteration {self.iterations}----")

        message, usage = self.get_completion()
        self.process_message(message)
        self.update_usage(usage)

        return self.memory

    def get_completion(self):
        return self.client.completion(
            messages=self.memory,
            tools=([tool.to_dict() for tool in self.tools] if self.tools else NotGiven()),
            **self.chat_completion_kwargs,
        )

    def process_message(self, message: Message):
        self.memory.append(message)

        if message.tool_calls:
            converted_tool_calls = self.convert_tool_calls(message.tool_calls)
            self.process_tool_calls(converted_tool_calls)

        self.iterations += 1

    def convert_tool_calls(self, tool_calls: list[ChatCompletionMessageToolCall]) -> list[ToolCall]:
        return [
            ToolCall(
                id=tool_call.id, function=tool_call.function.name, args=tool_call.function.arguments
            )
            for tool_call in tool_calls
        ]

    def process_tool_calls(self, tool_calls: list[ToolCall]):
        for tool_call in tool_calls:
            tool_response = self.call_tool(tool_call)
            self.memory.append(tool_response)

    def update_usage(self, usage: Usage):
        self.usage += usage


class MaxIterationsReachedException(Exception):
    pass
