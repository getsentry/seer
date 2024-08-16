import os
from unittest.mock import patch

import jwt
import pytest
from flask import Blueprint, Flask
from johen import change_watcher
from pydantic import BaseModel

from seer.configuration import AppConfig
from seer.dependency_injection import resolve
from seer.json_api import json_api


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

    response = test_client.post("/v0/some/url", json={"thing": "thing", "b": 12})
    assert response.status_code == 200
    assert response.get_json() == {"blah": "do it"}

    assert my_endpoint(DummyRequest(thing="thing", b=12)) == DummyResponse(blah="do it")


def test_json_api_signature_not_strict():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)
    headers = {}
    payload = {"thing": "thing", "b": 12}
    status_code_watcher = change_watcher(
        lambda: test_client.post("/v0/some/url", json=payload, headers=headers).status_code
    )

    with status_code_watcher as changed:
        os.environ["JSON_API_SHARED_SECRETS"] = "secret-one secret-two"
        headers["Authorization"] = "Rpcsignature rpc0:some-token"

    assert changed.result == 200

    with status_code_watcher as changed:
        headers["Authorization"] = (
            "Rpcsignature rpc0:96f23d5b3df807a9dc91f090078a46c00e17fe8b0bc7ef08c9391fa8b37a66b5"
        )

    assert changed.result == 200


def test_json_api_bearer_token_auth():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)

    app_config = resolve(AppConfig)
    app_config.ENFORCE_API_AUTH = True
    app_config.API_PUBLIC_KEY = "mock_public_key"

    with patch("seer.json_api.jwt.decode") as mock_jwt_decode:
        # Test valid token
        headers = {"Authorization": "Bearer valid_token"}
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 200
        mock_jwt_decode.assert_called_once_with(
            "valid_token", "mock_public_key", algorithms=["RS256"]
        )

        # Test invalid token
        mock_jwt_decode.side_effect = jwt.InvalidTokenError
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 401
        assert b"Invalid Bearer token" in response.data

        # Test missing Authorization header
        response = test_client.post("/v0/some/url", json={"thing": "thing", "b": 12})
        assert response.status_code == 401
        assert b"Authorization header is missing or invalid" in response.data

        # Test incorrect Authorization header format
        headers = {"Authorization": "InvalidFormat token"}
        response = test_client.post(
            "/v0/some/url", json={"thing": "thing", "b": 12}, headers=headers
        )
        assert response.status_code == 401
        assert b"Authorization header is missing or invalid" in response.data


def test_json_api_auth_not_enforced():
    app = Flask(__name__)
    blueprint = Blueprint("blueprint", __name__)
    test_client = app.test_client()

    @json_api(blueprint, "/v0/some/url")
    def my_endpoint(request: DummyRequest) -> DummyResponse:
        return DummyResponse(blah="do it")

    app.register_blueprint(blueprint)

    app_config = resolve(AppConfig)
    app_config.ENFORCE_API_AUTH = False

    # Test that request is allowed without any auth when ENFORCE_API_AUTH is False
    response = test_client.post("/v0/some/url", json={"thing": "thing", "b": 12})
    assert response.status_code == 200
    assert response.get_json() == {"blah": "do it"}


@pytest.mark.skip(reason="Waiting to validate configuration in production")
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
    status_code_watcher = change_watcher(
        lambda: test_client.post("/v0/some/url", json=payload, headers=headers).status_code
    )

    with status_code_watcher as changed:
        os.environ["JSON_API_SHARED_SECRETS"] = "secret-one secret-two"
        headers["Authorization"] = "Rpcsignature rpc0:some-token"

    assert changed.from_value(200)
    assert changed.to_value(401)

    with status_code_watcher as changed:
        headers["Authorization"] = (
            "Rpcsignature rpc0:96f23d5b3df807a9dc91f090078a46c00e17fe8b0bc7ef08c9391fa8b37a66b5"
        )

    assert changed.to_value(200)
