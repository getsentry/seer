import functools
import hashlib
import hmac
import inspect
import logging
from typing import Any, Callable, Type, TypeVar, get_type_hints

import jwt
import sentry_sdk
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from flask import Blueprint, request
from google.cloud import secretmanager
from pydantic import BaseModel, ValidationError
from werkzeug.exceptions import BadRequest, InternalServerError, Unauthorized

from seer.bootup import module, stub_module
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


_F = TypeVar("_F", bound=Callable[..., Any])


def access_secret(project_id: str, secret_id: str, version_id: str = "latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def get_public_key_from_secret(project_id: str, secret_id: str, version_id: str = "latest"):
    pem_data = access_secret(project_id, secret_id, version_id)
    public_key = serialization.load_pem_public_key(pem_data.encode(), backend=default_backend())
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return public_key_bytes


class PublicKeyBytes(BaseModel):
    bytes: bytes | None


@module.provider
def provide_public_key(config: AppConfig = injected) -> PublicKeyBytes:
    return PublicKeyBytes(
        bytes=(
            get_public_key_from_secret(config.GOOGLE_CLOUD_PROJECT, config.API_PUBLIC_KEY_SECRET_ID)
            if config.GOOGLE_CLOUD_PROJECT and config.API_PUBLIC_KEY_SECRET_ID
            else None
        )
    )


@stub_module.provider
def provide_public_key_stub() -> PublicKeyBytes:
    return PublicKeyBytes(bytes=None)


def json_api(blueprint: Blueprint, url_rule: str) -> Callable[[_F], _F]:
    @inject
    def decorator(
        implementation: _F, config: AppConfig = injected, public_key: PublicKeyBytes = injected
    ) -> _F:
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

        @inject
        def wrapper(config: AppConfig = injected) -> Any:
            raw_data = request.get_data()
            auth_header = request.headers.get("Authorization", "")

            if auth_header.startswith("Rpcsignature "):
                parts = auth_header.split()
                if len(parts) != 2 or not compare_signature(
                    body=raw_data,
                    signature=parts[1],
                ):
                    raise Unauthorized(f"Rpcsignature did not match for call to {request.url}")
            elif auth_header.startswith("Bearer "):
                token = auth_header.split()[1]
                try:
                    if public_key.bytes is None:
                        raise Unauthorized("Public key is not available")
                    # Verify the JWT token using PyJWT
                    jwt.decode(token, public_key.bytes, algorithms=["RS256"])

                    # Optionally, you can add additional checks here
                    # For example, checking the 'exp' claim for token expiration
                    # or verifying specific claims in the token payload

                    # If the token is successfully decoded and verified,
                    # the function will continue execution
                except jwt.ExpiredSignatureError:
                    raise Unauthorized("Token has expired")
                except jwt.InvalidSignatureError:
                    raise Unauthorized("Invalid signature")
                except jwt.InvalidTokenError:
                    raise Unauthorized("Invalid token")
                except Exception as e:
                    sentry_sdk.capture_exception(e)
                    raise InternalServerError("Something went wrong with the Bearer token auth")
            else:
                sentry_sdk.capture_message(f"No auth header found for request to {request.url}")
                # TODO: Actually raise unauthorized when we are sure we can enforce auth.
                # raise Unauthorized("No auth header found")

            # Cached from ^^, this won't result in double read.
            data = request.get_json()

            if not isinstance(data, dict):
                sentry_sdk.capture_message(f"Data is not an object: {type(data)}")
                raise BadRequest("Data is not an object")

            try:
                result: BaseModel = implementation(request_annotation.model_validate(data))
            except ValidationError as e:
                capture_alert(data)
                sentry_sdk.capture_exception(e)
                raise BadRequest(str(e))

            return result.model_dump()

        functools.update_wrapper(wrapper, implementation)
        blueprint.add_url_rule(url_rule, view_func=wrapper, methods=["POST"])

        return implementation

    return decorator


def capture_alert(data: dict):
    # Setting tags for Validation Errors on alerts
    if "context" in data and "id" in data["context"]:
        sentry_sdk.set_tag("alert_id", data["context"]["id"])
    if "organization_id" in data:
        sentry_sdk.set_tag("organization_id", data["organization_id"])
    if "project_id" in data:
        sentry_sdk.set_tag("project_id", data["project_id"])


def is_valid(payload: bytes, key: str, signature_data: str):
    computed = hmac.new(key.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed.encode(), signature_data.encode())


@inject
def compare_signature(*, body: bytes, signature: str, config: AppConfig = injected) -> bool:
    """
    Compare request data + signature signed by one of the shared secrets.
    Once a key has been able to validate the signature other keys will
    not be attempted. We should only have multiple keys during key rotations.
    """

    secrets = config.JSON_API_SHARED_SECRETS

    if not signature.startswith("rpc0:"):
        sentry_sdk.capture_message("Signature did not start with rpc0:")
        return False

    _, signature_data = signature.split(":", 2)

    for key in secrets:
        if is_valid(body, key, signature_data):
            return True

    sentry_sdk.capture_message("No signature matches found.")
    return False
