import functools
import hashlib
import hmac
import inspect
from typing import Any, Callable, Type, TypeVar, get_type_hints

import sentry_sdk
from flask import Blueprint, request
from pydantic import BaseModel, ValidationError
from werkzeug.exceptions import BadRequest, Unauthorized

from seer.env import Environment
from seer.injector import inject, injected

_F = TypeVar("_F", bound=Callable[..., Any])


def json_api(blueprint: Blueprint, url_rule: str) -> Callable[[_F], _F]:
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
                "json_api implementations must have one non keyword, argument, annotated with a BaseModel and a BaseModel return value"
            )

        def wrapper() -> Any:
            raw_data = request.get_data()
            auth_header = request.headers.get("Authorization", "")

            # Optional for now during rollout, make this required after rollout.
            if auth_header.startswith("Rpcsignature "):
                parts = auth_header.split()
                if len(parts) != 2 or not compare_signature(request.url, raw_data, parts[1]):
                    sentry_sdk.capture_message("Rpcsignature did not match", level="critical")
                    raise Unauthorized("Rpcsignature did not match for given url and data")

            # Cached from ^^, this won't result in double read.
            data = request.get_json()

            if not isinstance(data, dict):
                sentry_sdk.capture_message(f"Data is not an object: {type(data)}")
                raise BadRequest("Data is not an object")

            try:
                result: BaseModel = implementation(request_annotation.model_validate(data))
            except ValidationError as e:
                sentry_sdk.capture_exception(e)
                raise BadRequest(str(e))

            return result.model_dump()

        functools.update_wrapper(wrapper, implementation)
        blueprint.add_url_rule(url_rule, view_func=wrapper, methods=["POST"])

        return implementation

    return decorator


@inject
def compare_signature(
    url: str, body: bytes, signature: str, environment: Environment = injected
) -> bool:
    """
    Compare request data + signature signed by one of the shared secrets.
    Once a key has been able to validate the signature other keys will
    not be attempted. We should only have multiple keys during key rotations.
    """
    if not signature.startswith("rpc0:"):
        sentry_sdk.capture_message("Signature did not start with rpc0:")
        return False

    _, signature_data = signature.split(":", 1)
    signature_input = b"%s:%s" % (
        url.encode(),
        body,
    )

    for key in environment.JSON_API_SHARED_SECRETS:
        computed = hmac.new(key.encode(), signature_input, hashlib.sha256).hexdigest()
        is_valid = hmac.compare_digest(computed.encode(), signature_data.encode())
        if is_valid:
            return True
        else:
            sentry_sdk.capture_message("Signature did not match hmac, trying other secrets")

    return False
