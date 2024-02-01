import logging
from typing import Callable, Dict, List

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FunctionTool(BaseModel):
    name: str
    description: str
    fn: Callable
    parameters: List[Dict[str, str]]
    required: List[str] = []

    def call(self, **kwargs):
        try:
            return self.fn(**kwargs)
        except Exception as e:
            logger.exception(e)
            return f"Error: {e}"

    def to_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: {
                            "type": param["type"],
                            "description": param.get("description", ""),
                        }
                        for param in self.parameters
                    },
                    "required": self.required,
                },
            },
        }


class FunctionParameter(BaseModel):
    name: str
    type: str
    description: str
    enum: List[str]
