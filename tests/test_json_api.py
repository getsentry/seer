from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Blueprint, Flask
from johen import change_watcher
from pydantic import BaseModel

from seer.configuration import AppConfig
from seer.dependency_injection import Module, resolve
from seer.json_api import PublicKeyBytes, json_api


class DummyRequest(BaseModel):
    thing: str
    b: int


class DummyResponse(BaseModel):
    blah: str


def test_json_api_decorator():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        assert request.thing == "thing"
        assert request.b == 12
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)

    app_config = resolve(AppConfig)
    app_config.IGNORE_API_AUTH = True

    response = test_client.post("/v0/some/url", json={"thing": "thing", "b": 12})
    assert response.status_code == 200
    assert response.get_json() == {"blah": "do it"}

    assert my_endpoint(DummyRequest(thing="thing", b=12)) == DummyResponse(blah="do it")


def test_json_api_bearer_token_auth():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)

    app_config = resolve(AppConfig)
    app_config.IGNORE_API_AUTH = False
    app_config.DEV = False

    pk = resolve(PublicKeyBytes)
    pk.bytes = b"mock_public_key"

    with patch("seer.json_api.jwt.decode") as mock_jwt_decode:
        # Test valid token
        headers = {"Authorization": "Bearer valid_token"}
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 200
        mock_jwt_decode.assert_called_once_with(
            "valid_token", b"mock_public_key", algorithms=["RS256"]
        )

        # Test invalid token
        mock_jwt_decode.side_effect = jwt.InvalidTokenError
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 401
        assert b"Invalid token" in response.data

        # Test missing Authorization header
        response = test_client.post("/v0/some/url", json={"thing": "thing", "b": 12})
        assert response.status_code == 401
        assert (
            b"Neither Rpcsignature nor a Bearer token was included in authorization header!"
            in response.data
        )

        # Test incorrect Authorization header format
        headers = {"Authorization": "InvalidFormat token"}
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 401
        assert (
            b"Neither Rpcsignature nor a Bearer token was included in authorization header!"
            in response.data
        )


def test_json_api_auth_not_enforced():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)

    app_config = resolve(AppConfig)
    app_config.IGNORE_API_AUTH = False

    # Test that request is allowed without any auth when ENFORCE_API_AUTH is False
    response = test_client.post("/v0/some/url", json={"thing": "thing", "b": 12})
    assert response.status_code == 200
    assert response.get_json() == {"blah": "do it"}


def test_json_api_auth_with_real_jwt():

    app_config = resolve(AppConfig)
    app_config.IGNORE_API_AUTH = False
    app_config.DEV = False

    # Generate a test RSA key pair
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Convert public key to PEM format
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Create a test JWT token
    payload = {"sub": "1234567890", "name": "Test User", "iat": 1516239022}
    token = jwt.encode(payload, private_key, algorithm="RS256")

    module = Module()
    module.constant(PublicKeyBytes, PublicKeyBytes(bytes=public_key_pem))
    with module:
        app = Flask(__name__)
        blueprint = Blueprint("blueprint", __name__)
        test_client = app.test_client()

        @json_api(blueprint, "/v0/some/url")
        def my_endpoint(request: DummyRequest) -> DummyResponse:
            return DummyResponse(blah="do it")

        app.register_blueprint(blueprint)

        # Test valid token
        headers = {"Authorization": f"Bearer {token}"}
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 200
        assert response.get_json() == {"blah": "do it"}

        # Test invalid token
        invalid_token = jwt.encode(payload, "wrong_key", algorithm="HS256")
        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 401
        assert b"Invalid token" in response.data

        # Test expired token
        import time

        expired_payload = {"exp": int(time.time()) - 300}  # Token expired 5 minutes ago
        expired_token = jwt.encode(expired_payload, private_key, algorithm="RS256")
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 401
        assert b"Token has expired" in response.data


def test_json_api_signature_strict_mode_ignores_rpcsignature():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)

    headers = {}
    payload = {"thing": "thing", "b": 12}
    path = "/v0/some/url"
    status_code_watcher = change_watcher(
        lambda: test_client.post(path, json=payload, headers=headers).status_code
    )

    with Module() as injector:
        injector.get(AppConfig).JSON_API_SHARED_SECRETS = ["secret-one", "secret-two"]

        with status_code_watcher as changed:
            headers["Authorization"] = "Rpcsignature rpc0:some-token"

        assert changed.result == 200


@pytest.mark.skip(reason="Disable auth")
def test_json_api_signature_strict_mode():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)

    headers = {}
    payload = {"thing": "thing", "b": 12}
    path = "/v0/some/url"
    status_code_watcher = change_watcher(
        lambda: test_client.post(path, json=payload, headers=headers).status_code
    )

    with Module() as injector:
        injector.get(AppConfig).JSON_API_SHARED_SECRETS = ["secret-one", "secret-two"]

        with status_code_watcher as changed:
            headers["Authorization"] = "Rpcsignature rpc0:some-token"

        assert changed.from_value(200)
        assert changed.to_value(401)

        with status_code_watcher as changed:
            headers["Authorization"] = (
                "Rpcsignature rpc0:96f23d5b3df807a9dc91f090078a46c00e17fe8b0bc7ef08c9391fa8b37a66b5"
            )
        assert changed.to_value(200)

        with status_code_watcher as changed:
            path += "?nonce=1234"
        assert changed.to_value(401)

        with status_code_watcher as changed:
            headers["Authorization"] = (
                "Rpcsignature rpc0:487fb810a4e87faf306dc9637cec9aaea2be37247410391b372178ffc15af6a8"
            )
        assert changed.to_value(200)
