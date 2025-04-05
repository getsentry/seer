import dataclasses
import hashlib
import hmac
import logging
import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import requests
from requests import HTTPError

from seer.configuration import AppConfig
from seer.dependency_injection import Module, injected
from seer.utils import json_dumps

logger = logging.getLogger(__name__)

module = Module()
module.enable()

rpc_stub_module = Module()


class RpcClient(ABC):
    @abstractmethod
    def call(self, method: str, **kwargs) -> dict[str, Any] | None:
        pass


@dataclasses.dataclass
class FakeHttpResponse:
    status_code: int
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


@dataclasses.dataclass
class DummyRpcClient(RpcClient):
    """
    A mock RPCClient that forces (method, **kwargs) to search for method named "method" and
    invoke that for a response, or uses a default implementation that logs the interaction
    without providing a definitive response.

    Subclass this client to add unique behavior or add to it to control default dummy behavior.
    """

    should_log: bool = False
    dry_run: bool = False
    invocations: list[tuple[str, dict[str, Any]]] = dataclasses.field(default_factory=list)

    # use to force a specific consent value for get_organization_autofix_consent
    should_consent: bool = False

    # Use to force all requests to fail with a given http status code
    force_failure_status: int | None = None

    def call(self, method: str, **kwargs) -> dict[str, Any] | None:
        logger.warn(f"Call to Sentry autofix API {method} handled by DummyRpcClient")
        self.invocations.append((method, kwargs))

        if self.force_failure_status:
            raise HTTPError(
                response=FakeHttpResponse(status_code=self.force_failure_status, content=b"")
            )

        result = getattr(self, method, self._default_call)(method, kwargs)
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        status, msg = result
        raise HTTPError(response=FakeHttpResponse(status_code=status, content=msg.encode("utf-8")))

    def get_organization_autofix_consent(
        self, method: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        return {"consent": self.should_consent}

    def _default_call(
        self, method: str, kwargs: dict[str, Any]
    ) -> dict[str, Any] | tuple[int, str] | None:
        if self.dry_run:
            return None

        body_dict = {"args": kwargs}
        json_dump = json_dumps(body_dict, separators=(",", ":"))

        if self.should_log:
            print(f"Calling {method} with {json_dump}")
        return 404, "Not Found"


class SentryRpcClient(RpcClient):
    @cached_property
    def shared_secret(self) -> str:
        shared_secret = os.environ.get("RPC_SHARED_SECRET")
        if not shared_secret:
            raise RuntimeError("RPC_SHARED_SECRET must be set")
        return shared_secret

    @cached_property
    def base_url(self) -> str:
        base_url = os.environ.get("SENTRY_BASE_URL")
        if not base_url:
            raise RuntimeError("SENTRY_BASE_URL must be set")
        return base_url

    def _generate_request_signature(self, url_path: str, body: bytes) -> str:
        signature_input = body
        signature = hmac.new(
            self.shared_secret.encode("utf-8"), signature_input, hashlib.sha256
        ).hexdigest()
        return f"rpc0:{signature}"

    def call(self, method: str, **kwargs) -> dict[str, Any] | None:
        body_bytes, endpoint, headers = self._prepare_request(method, kwargs)
        response = requests.post(endpoint, headers=headers, data=body_bytes)
        response.raise_for_status()
        if response.headers.get("Content-Type", "") != "application/json":
            logger.warning("No application/json content type")
            return None
        return response.json()

    def _prepare_request(self, method, kwargs):
        url_path = f"/api/0/internal/seer-rpc/{method}/"
        endpoint = f"{self.base_url}{url_path}"
        body_dict = {"args": kwargs}
        body = json_dumps(body_dict, separators=(",", ":"))
        body_bytes = body.encode("utf-8")
        signature = self._generate_request_signature(url_path, body_bytes)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Rpcsignature {signature}",
        }
        return body_bytes, endpoint, headers


@module.provider
def get_sentry_client(config: AppConfig = injected) -> RpcClient:
    if config.NO_SENTRY_INTEGRATION:
        rpc_client: DummyRpcClient = DummyRpcClient()
        rpc_client.dry_run = True
        return rpc_client
    else:
        return SentryRpcClient()


# By using two providers by both these type names, you can access the same
@rpc_stub_module.provider
def get_sentry_dummy_client() -> DummyRpcClient:
    return DummyRpcClient()


@rpc_stub_module.provider
def get_sentry_stub_client(dummy_client: DummyRpcClient = injected) -> RpcClient:
    return dummy_client
