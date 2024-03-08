import contextlib
import hashlib
import hmac
import logging
import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable

import aiohttp
import requests

from seer.utils import json_dumps

logger = logging.getLogger(__name__)


class RpcClient(ABC):
    @abstractmethod
    def call(self, method: str, **kwargs) -> dict[str, Any]:
        pass

    @abstractmethod
    async def acall(self, method: str, **kwargs) -> dict[str, Any]:
        pass


class DummyRpcClient(RpcClient):
    handlers: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]]

    def __init__(self, should_log: bool = False):
        self.should_log = should_log
        self.handlers = {}

    def call(self, method: str, **kwargs) -> dict[str, Any]:
        return self.handlers.get(method, self._default_call)(method, kwargs)

    def _default_call(self, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        body_dict = {"args": kwargs}
        json_dump = json_dumps(body_dict, separators=(",", ":"))

        if self.should_log:
            print(f"Calling {method} with {json_dump}")
        return {}

    async def acall(self, method: str, **kwargs) -> dict[str, Any]:
        kwargs.pop("session", None)
        return self.call(method, **kwargs)


class SentryRpcClient(RpcClient):
    @cached_property
    def shared_secret(self) -> str:
        shared_secret = os.environ.get("SENTRY_BASE_URL")
        if not shared_secret:
            raise RuntimeError("SENTRY_BASE_URL must be set")
        return shared_secret

    @cached_property
    def base_url(self) -> str:
        base_url = os.environ.get("SENTRY_BASE_URL")
        if not base_url:
            raise RuntimeError("SENTRY_BASE_URL must be set")
        return base_url

    def _generate_request_signature(self, url_path: str, body: bytes) -> str:
        signature_input = b"%s:%s" % (url_path.encode("utf8"), body)
        signature = hmac.new(
            self.shared_secret.encode("utf-8"), signature_input, hashlib.sha256
        ).hexdigest()
        return f"rpc0:{signature}"

    def call(self, method: str, **kwargs) -> dict[str, Any]:
        body_bytes, endpoint, headers = self._prepare_request(method, kwargs)
        response = requests.post(endpoint, headers=headers, data=body_bytes)
        response.raise_for_status()
        return response.json()

    async def acall(
        self, method: str, session: aiohttp.ClientSession | None = None, **kwargs
    ) -> dict[str, Any]:
        body_bytes, endpoint, headers = self._prepare_request(method, kwargs)
        async with contextlib.AsyncExitStack() as stack:
            if session is None:
                session = aiohttp.ClientSession()
            await stack.enter_async_context(session)
            response = await stack.enter_async_context(
                session.post(endpoint, headers=headers, data=body_bytes)
            )
            response.raise_for_status()
            result = await response.json()
        return result

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
