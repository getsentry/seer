import logging
from typing import Callable, Dict, List

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


class ClaudeTool(BaseModel):
    type: str
    name: str
    fn: Callable

    def call(self, **kwargs):
        try:
            return self.fn(**kwargs)
        except Exception as e:
            logger.exception(e)
            return f"Error: {get_full_exception_string(e)}"
