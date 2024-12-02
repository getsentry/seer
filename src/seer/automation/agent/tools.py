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

    def _get_parameter_default(self, param_name: str) -> any:
        for param in self.parameters:
            if param["name"] == param_name and "default" in param:
                return param["default"]
        return None

    def call(self, **kwargs):
        try:
            # Check for required parameters
            missing_params = [param for param in self.required if param not in kwargs]
            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

            # Add defaults for parameters if specified
            for param in self.parameters:
                if param["name"] not in kwargs and "default" in param:
                    kwargs[param["name"]] = param["default"]

            return self.fn(**kwargs)
        except Exception as e:
            logger.exception(e)
            return f"Error: {get_full_exception_string(e)}"