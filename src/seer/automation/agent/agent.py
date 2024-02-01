import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import openai_multi_tool_use_parallel_patch  # import applies the patch
from openai import OpenAI
from openai._types import NotGiven

from seer.automation.agent.tools import FunctionTool
from seer.automation.agent.types import Message, ToolCall, Usage

logger = logging.getLogger("autofix")


class LlmAgent(ABC):
    name: str
    tools: list[FunctionTool]
    memory: list[Message]

    openai_client: OpenAI
    iterations: int = 0
    max_iterations: int = 48

    def __init__(
        self,
        tools: list[FunctionTool] | None = None,
        memory: list[Message] | None = None,
        name="Agent",
        stop_message: str | None = None,
    ):
        self.tools = tools or []
        self.memory = memory or []
        self.usage = Usage()
        self.name = name
        self.openai_client = OpenAI()
        self.stop_message = stop_message

    @abstractmethod
    def run_iteration(self):
        pass

    def run(self, prompt: str):
        self.memory.append(
            Message(
                role="user",
                content=prompt,
            )
        )

        logger.debug(f"----[{self.name}] Running Agent----")
        logger.debug(f"Previous messages: ")
        for message in self.memory:
            logger.debug(f"{message.role}: {message.content}")

        while (
            self.iterations == 0
            or (
                not (
                    self.memory[-1].role
                    in ["assistant", "model"]  # Will never end on a message not from the assistant
                    and (
                        self.memory[-1].content
                        and (
                            self.stop_message in self.memory[-1].content
                        )  # If stop message is defined; will end if the assistant response contains the stop message
                        if self.stop_message
                        else self.memory[-1].content
                        is not None  # If a stop message is not defined; will end on any non-empty assistant response (OpenAI tool call does not output a message!)
                    )
                )
            )
            and self.iterations < self.max_iterations  # Went above max iterations
        ):
            # runs until the assistant sends a message with no more tool calls.
            self.run_iteration()

        if self.iterations == self.max_iterations:
            raise Exception(f"Agent {self.name} reached maximum iterations without finishing.")

        return self.memory[-1].content

    def call_tool(self, tool_call: ToolCall):
        logger.debug(
            f"[{tool_call.id}] Calling tool {tool_call.function} with arguments {tool_call.args}"
        )

        tool = next(tool for tool in self.tools if tool.name == tool_call.function)

        tool_result = tool.call(**tool_call.args)

        logger.debug(f"Tool {tool_call.function} returned \n{tool_result}")

        return Message(
            role="tool",
            content=tool_result,
            tool_call_id=tool_call.id,
        )


class GptAgent(LlmAgent):
    model: str = "gpt-4-0125-preview"

    chat_completion_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        tools: list[FunctionTool] | None = None,
        memory: list[Message] | None = None,
        name="GptAgent",
        chat_completion_kwargs={},
        stop_message: str | None = None,
    ):
        super().__init__(tools, memory, name=name, stop_message=stop_message)

        self.chat_completion_kwargs = chat_completion_kwargs

    def run_iteration(self):
        logger.debug(f"----[{self.name}] Running Iteration {self.iterations}----")

        messages = [{k: v for k, v in msg.dict().items() if v is not None} for msg in self.memory]
        # logger.debug(f"Messages: {messages}")

        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            tools=([tool.to_dict() for tool in self.tools] if len(self.tools) > 0 else NotGiven()),
            **self.chat_completion_kwargs,
        )

        response_message = completion.choices[0].message
        self.memory.append(Message(**response_message.dict()))

        logger.debug(f"Message content:\n{response_message.content}")

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                tool_response = self.call_tool(
                    ToolCall(
                        id=tool_call.id,
                        function=tool_call.function.name,
                        args=json.loads(tool_call.function.arguments),
                    )
                )

                self.memory.append(tool_response)

        self.iterations += 1

        if completion.usage:
            self.usage.completion_tokens += completion.usage.completion_tokens
            self.usage.prompt_tokens += completion.usage.prompt_tokens
            self.usage.total_tokens += completion.usage.total_tokens

        return self.memory
