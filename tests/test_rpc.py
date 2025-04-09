import contextlib
import dataclasses
import http.server
import json
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from queue import Queue
from typing import Iterator, Type

import pytest
from johen.pytest import parametrize
from requests.models import HTTPError

from seer.automation.autofix.models import AutofixCompleteArgs, AutofixStepUpdateArgs
from seer.rpc import SentryRpcClient
from seer.utils import closing_queue


class QueueHandler(http.server.BaseHTTPRequestHandler):
    request_queue: Queue = Queue(maxsize=1)
    response_queue: Queue = Queue(maxsize=1)

    def do_POST(self):
        request_body = self.rfile.read(int(self.headers["Content-Length"]))
        self.request_queue.put((request_body, self.headers, self.path), timeout=5)

        response = self.response_queue.get(timeout=5)

        if isinstance(response, int):
            self.send_response(response)
            self.end_headers()
        else:
            self.send_response(HTTPStatus.OK)

            if response:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            self.wfile.flush()


def find_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@dataclasses.dataclass
class TestRpcHttpServer:
    server_addr: tuple[str, int] = dataclasses.field(default_factory=lambda: ("", find_free_port()))

    @contextlib.contextmanager
    def enabled(self) -> Iterator[Type[QueueHandler]]:
        old_base_url = os.environ["SENTRY_BASE_URL"]
        old_shared_secret = os.environ.get("RPC_SHARED_SECRET", None)

        server = http.server.HTTPServer(self.server_addr, lambda a, b, c: QueueHandler(a, b, c))
        thread = threading.Thread(target=lambda: server.serve_forever())
        thread.start()

        os.environ["SENTRY_BASE_URL"] = f"http://localhost:{server.server_port}"
        os.environ["RPC_SHARED_SECRET"] = "secret-sauce"
        try:
            with closing_queue(QueueHandler.request_queue, QueueHandler.response_queue):
                yield QueueHandler
        finally:
            server.shutdown()
            os.environ["SENTRY_BASE_URL"] = old_base_url
            if old_shared_secret:
                os.environ["SHARED_RPC_SECRET"] = old_shared_secret
            thread.join()


@parametrize
def test_rpc_call_200_json(
    test_server: TestRpcHttpServer,
    args: AutofixCompleteArgs | AutofixStepUpdateArgs,
    expected_result: dict[str, int],
):
    with test_server.enabled() as QueueHandler, ThreadPoolExecutor() as pool:

        def handle_request():
            body, headers, path = QueueHandler.request_queue.get(timeout=5)
            QueueHandler.response_queue.put(json.dumps(expected_result).encode("utf-8"))
            assert type(args).model_validate(json.loads(body)["args"]) == args
            assert headers["Content-Type"] == "application/json"
            assert path == "/api/0/internal/seer-rpc/method/"

        future = pool.submit(handle_request)

        client = SentryRpcClient()
        r = client.call("method", **args.model_dump(mode="json"))
        assert r == expected_result

        future.result()


@parametrize
def test_rpc_call_200_empty(test_server: TestRpcHttpServer):
    with test_server.enabled() as QueueHandler, ThreadPoolExecutor() as pool:

        client = SentryRpcClient()

        def handle_request():
            body, headers, path = QueueHandler.request_queue.get(timeout=5)
            QueueHandler.response_queue.put(b"")
            assert (
                headers["Authorization"]
                == f"Rpcsignature {client._generate_request_signature(body)}"
            )

        future = pool.submit(handle_request)

        r = client.call("method", issue_id=1)
        assert r is None

        future.result()


@parametrize(count=1)
def test_rpc_call_404(test_server: TestRpcHttpServer):
    with test_server.enabled() as QueueHandler, ThreadPoolExecutor() as pool:

        def handle_request():
            body, headers, path = QueueHandler.request_queue.get(timeout=5)
            QueueHandler.response_queue.put(404)

        future = pool.submit(handle_request)

        client = SentryRpcClient()
        with pytest.raises(HTTPError) as exc_info:
            client.call("method")

        assert exc_info.value.response.status_code == 404

        future.result()
