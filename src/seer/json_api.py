import functools
import inspect
from typing import Any, Callable, List, Tuple, Type, TypeVar, get_type_hints

import sentry_sdk
from flask import Flask, request
from pydantic import BaseModel, ValidationError
from werkzeug.exceptions import BadRequest

_F = TypeVar("_F", bound=Callable[..., Any])

view_functions: List[Tuple[str, Callable[[], Any], Type[BaseModel], Type[BaseModel]]] = []


def json_api(url_rule: str) -> Callable[[_F], _F]:
    def decorator(implementation: _F) -> _F:
        spec = inspect.getfullargspec(implementation)
        annotations = get_type_hints(implementation)
        try:
            assert len(spec.args) == 1

            request_annotation: Type[BaseModel] = annotations[spec.args[0]]
            response_annotation: Type[BaseModel] = annotations["return"]
            assert issubclass(request_annotation, BaseModel)
            assert issubclass(response_annotation, BaseModel)
        except (KeyError, IndexError, AssertionError):
            raise ValueError(
                f"json_api implementations must have one non keyword, argument, annotated with a BaseModel and a BaseModel return value"
            )

        def wrapper() -> Any:
            data = request.get_json()
            if not isinstance(data, dict):
                sentry_sdk.capture_message(f"Data is not an object: {type(data)}")
                raise BadRequest("Data is not an object")

            try:
                result: BaseModel = implementation(**data)
            except ValidationError as e:
                sentry_sdk.capture_exception(e)
                raise BadRequest(str(e))

            return result.model_dump()

        functools.update_wrapper(wrapper, implementation)
        view_functions.append((url_rule, wrapper, request_annotation, response_annotation))

        return implementation

    return decorator


def register_json_api_views(app: Flask) -> None:
    for url_rule, wrapper, _, _ in view_functions:

        app.add_url_rule(url_rule, view_func=wrapper, methods=["POST"])
