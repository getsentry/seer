import hashlib
import hmac
import json
import logging
import os
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)


class RpcClient(ABC):
    @abstractmethod
    def call(self, method: str, **kwargs):
        pass


class DummyRpcClient(RpcClient):
    def __init__(self, should_log: bool = False):
        self.should_log = should_log

    def call(self, method: str, **kwargs):
        if self.should_log:
            print(f"Calling {method} with {kwargs}")
        return None


class SentryRpcClient(RpcClient):
    shared_secret: str

    def __init__(self, base_url: str):
        self.base_url = base_url
        shared_secret = os.environ.get("RPC_SHARED_SECRET")
        if not shared_secret:
            raise RuntimeError("RPC_SHARED_SECRET must be set")
        self.shared_secret = shared_secret

    def _generate_request_signature(self, url_path: str, body: bytes) -> str:
        signature_input = b"%s:%s" % (url_path.encode("utf8"), body)
        signature = hmac.new(
            self.shared_secret.encode("utf-8"), signature_input, hashlib.sha256
        ).hexdigest()
        return f"rpc0:{signature}"

    def call(self, method: str, **kwargs):
        url_path = f"/api/0/internal/seer-rpc/{method}/"
        endpoint = f"{self.base_url}{url_path}"
        body_dict = {"args": kwargs}
        body = json_dumps(body_dict, separators=(',', ':'))
        body_bytes = body.encode("utf-8")
        signature = self._generate_request_signature(url_path, body_bytes)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Rpcsignature {signature}",
        }
        response = requests.post(endpoint, headers=headers, json=body_dict)
        response.raise_for_status()

        if response.headers.get("Content-Type") == "application/json":
            return response.json()
        else:
            return None
