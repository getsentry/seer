from flask import Blueprint, Flask
from johen import change_watcher
from pydantic import BaseModel

from seer.configuration import AppConfig
from seer.dependency_injection import Module
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
