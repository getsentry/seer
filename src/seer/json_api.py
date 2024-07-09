import functools
import hashlib
import hmac
import inspect
import os
from typing import Any, Callable, List, Tuple, Type, TypeVar, get_type_hints

import sentry_sdk
from flask import Flask, request
from pydantic import BaseModel, ValidationError
from werkzeug.exceptions import BadRequest, Unauthorized

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
                "json_api implementations must have one non keyword, argument, annotated with a BaseModel and a BaseModel return value"
            )

        def wrapper() -> Any:
            raw_data = request.get_data()
            auth_header = request.headers.get("Authorization", "")

            # Optional for now during rollout, make this required after rollout.
            if auth_header.startswith("Rpcsignature "):
                parts = auth_header.split()
                if len(parts) != 2 or not compare_signature(request.url, raw_data, parts[1]):
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
        view_functions.append((url_rule, wrapper, request_annotation, response_annotation))

        return implementation

    return decorator


def register_json_api_views(app: Flask) -> None:
    for url_rule, wrapper, _, _ in view_functions:
        app.add_url_rule(url_rule, view_func=wrapper, methods=["POST"])


def get_json_api_shared_secrets() -> list[str]:
    result = os.environ.get("JSON_API_SHARED_SECRETS", "").split()
    # TODO: Add this back in after we confirm with safer behavior.
    # if not result:
    #     raise ValueError(
    #         "JSON_API_SHARED_SECRETS environment variable required to support signature based auth."
    #     )
    return result


def compare_signature(url: str, body: bytes, signature: str) -> bool:
    """
    Compare request data + signature signed by one of the shared secrets.
    Once a key has been able to validate the signature other keys will
    not be attempted. We should only have multiple keys during key rotations.
    """
    # During the transition, support running seer without the shared secrets.
    if not signature:
        return True

    secrets = get_json_api_shared_secrets()

    if not signature.startswith("rpc0:"):
        sentry_sdk.capture_message("Signature did not start with rpc0:")
        return True
        # return False

    _, signature_data = signature.split(":", 2)
    signature_input = b"%s:%s" % (
        url.encode(),
        body,
    )

    for key in secrets:
        computed = hmac.new(key.encode(), signature_input, hashlib.sha256).hexdigest()
        is_valid = hmac.compare_digest(computed.encode(), signature_data.encode())
        if is_valid:
            return True
        else:
            sentry_sdk.capture_message("Signature did not match hmac")

    sentry_sdk.capture_message("No signature matches found")
    return True
    # return False
