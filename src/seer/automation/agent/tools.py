import logging
from typing import Any, Callable, Dict, List, Literal, TypedDict

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def get_full_exception_string(exc):
    result = str(exc)
    if exc.__cause__:
        if result:
            result += f"\n\nThe above exception was the direct cause of the following exception:\n\n{str(exc.__cause__)}"
        else:
            result = str(exc.__cause__)
    return result


class GPTFunctionSchema(TypedDict):
    type: Literal["function"]
    function: Dict[str, Any]


class ClaudeFunctionSchema(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]


class FunctionTool(BaseModel):
    name: str
    description: str
    fn: Callable
    parameters: List[Dict[str, str | List[str] | Dict[str, str]]]
    required: List[str] = []

    def call(self, **kwargs):
        try:
            return self.fn(**kwargs)
        except Exception as e:
            logger.exception(e)
            return f"Error: {get_full_exception_string(e)}"

    def to_dict(
        self, model: Literal["claude", "gpt"] = "gpt"
    ) -> GPTFunctionSchema | ClaudeFunctionSchema:
        base_schema = {
            "type": "object",
            "properties": {
                param["name"]: {
                    key: value
                    for key, value in {
                        "type": param["type"],
                        "description": param.get("description", ""),
                        "items": param.get("items"),
                    }.items()
                    if value is not None
                }
                for param in self.parameters
            },
            "required": self.required,
        }

        if model == "gpt":
            return GPTFunctionSchema(
                type="function",
                function={
                    "name": self.name,
                    "description": self.description,
                    "parameters": base_schema,
                },
            )
        elif model == "claude":
            return ClaudeFunctionSchema(
                name=self.name,
                description=self.description,
                input_schema=base_schema,
            )
