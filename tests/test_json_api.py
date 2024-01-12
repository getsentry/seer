from flask import Flask
from pydantic import BaseModel

from seer.json_api import json_api, register_json_api_views, view_functions


class DummyRequest(BaseModel):
    thing: str
    b: int


class DummyResponse(BaseModel):
    blah: str


def test_json_api_decorator():
    old_view_functions = [*view_functions]
    app = Flask(__name__)
    test_client = app.test_client()

    try:
        view_functions.clear()

        @json_api("/v0/some/url")
        def my_endpoint(request: DummyRequest) -> DummyResponse:
            assert request.thing == "thing"
            assert request.b == 12
            return DummyResponse(blah="do it")

        register_json_api_views(app)
        assert view_functions[0][2] == DummyRequest
        assert view_functions[0][3] == DummyResponse

        response = test_client.post("/v0/some/url", json={"thing": "thing", "b": 12})
        assert response.status_code == 200
        assert response.get_json() == {"blah": "do it"}

        assert my_endpoint(DummyRequest(thing="thing", b=12)) == DummyResponse(blah="do it")
    finally:
        view_functions.clear()
        view_functions.extend(old_view_functions)
