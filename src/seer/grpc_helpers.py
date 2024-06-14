import dataclasses
import struct
import time
from typing import Any, Callable, Iterable

import grpc
import requests
from flask import Flask, Response, abort, request
from google.protobuf.message import DecodeError, EncodeError
from sentry_services.seer.severity_pb2 import ScoreRequest
from sentry_services.seer.severity_pb2_grpc import SeverityStub

from seer.severity.severity_inference import SeverityRequest


@dataclasses.dataclass(frozen=True)
class HandlerCallDetails(grpc.HandlerCallDetails):
    method: str
    invocation_metadata: tuple[()] = ()


@dataclasses.dataclass(kw_only=True)
class ServicerContext(grpc.ServicerContext):
    headers: dict[str, str]
    details: str | None = None
    code: grpc.StatusCode = grpc.StatusCode.OK
    deadline: float = 0.0
    callbacks: list[Callable[[], None]] = dataclasses.field(default_factory=list)

    def set_code(self, code: grpc.StatusCode | int) -> None:
        if isinstance(code, grpc.StatusCode):
            self.code = code
            return

        for status_code in grpc.StatusCode:
            if status_code.value[0] == code:
                self.code = status_code
                break
        else:
            raise ValueError(f"Unknown StatusCode: {code}")

    def send_initial_metadata(self, initial_metadata: Any):
        raise NotImplementedError

    def set_trailing_metadata(self, trailing_metadata: Any):
        raise NotImplementedError

    def set_details(self, details: str):
        self.details = details

    def abort(self, code: grpc.StatusCode, details: str):
        if code == grpc.StatusCode.OK:
            raise ValueError()

        self.set_code(code)
        self.set_details(details)

        raise grpc.RpcError()

    def abort_with_status(self, status: grpc.StatusCode):
        if status == grpc.StatusCode.OK:
            raise ValueError()

        self.set_code(status)

        raise grpc.RpcError()

    def time_remaining(self):
        return max(self.deadline - time.monotonic(), 0)

    def invocation_metadata(self):
        return ()

    def peer(self) -> str:
        return ""

    def client_cert(self) -> dict[str, str]:
        return {
            k: v
            for segment in self.headers.get("X-Forwarded-Client-Cert", "").split(";")
            for (k, v) in (segment.split("=", 1),)
            if segment
        }

    def peer_identities(self) -> list[bytes] | None:
        return self.auth_context().get(self.peer_identity_key())

    def peer_identity_key(self) -> str:
        return "DNS"

    def auth_context(self) -> dict[str, list[bytes]]:
        return {k: [v.encode("ascii")] for k, v in self.client_cert().items()}

    def add_callback(self, cb: Callable[[], None]):
        self.callbacks.append(cb)

    def cancel(self):
        raise NotImplementedError()

    def is_active(self):
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class FlaskHandlerGrpcServer(grpc.Server):
    app: Flask
    handlers: list[grpc.GenericRpcHandler] = dataclasses.field(default_factory=list)

    def __call__(self) -> Response | None:
        if request.method != "POST":
            return None

        if request.content_type != "application/grpc":
            return None

        parts = request.path.split("/")
        if len(parts) != 3:
            return None

        for handler in self.handlers:
            method_handler = handler.service(HandlerCallDetails(request.path))
            if method_handler is None:
                continue

            if method_handler.request_streaming or method_handler.response_streaming:
                raise NotImplementedError

            unary_unary = method_handler.unary_unary
            if unary_unary is None:
                raise NotImplementedError

            deserializer = method_handler.request_deserializer
            serializer = method_handler.response_serializer
            if not deserializer or not serializer:
                raise NotImplementedError

            try:
                request_proto = deserializer(request.data)
            except DecodeError as e:
                self.app.logger.error(
                    f"Failed to deserializer parameters to {request.path}", exc_info=e
                )
                abort(400)

            context = ServicerContext(headers=request.headers)
            try:
                result = unary_unary(request_proto, context)
            except grpc.RpcError:
                abort(self.map_to_http_status(context.code))
            else:
                try:
                    serialized: bytes = serializer(result)
                except EncodeError as e:
                    self.app.logger.error(
                        f"Failed to serialize response to {request.path}", exc_info=e
                    )
                    abort(500)

            return Response(
                response=serialized,
                status=self.map_to_http_status(context.code),
                content_type="application/grpc",
                headers={
                    "Content-Length": str(len(serialized)),
                },
            )

        return None

    def map_to_http_status(self, code: grpc.StatusCode) -> int:
        if code == grpc.StatusCode.OK:
            return 200
        if code == grpc.StatusCode.NOT_FOUND:
            return 404
        if code == grpc.StatusCode.PERMISSION_DENIED:
            return 403
        if code == grpc.StatusCode.UNAUTHENTICATED:
            return 401
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            return 400
        if code == grpc.StatusCode.ALREADY_EXISTS:
            return 409
        return 500

    def add_generic_rpc_handlers(self, generic_rpc_handlers: Iterable[grpc.GenericRpcHandler]):
        self.handlers.extend(generic_rpc_handlers)

    def add_insecure_port(self, address: str):
        raise NotImplementedError()

    def add_secure_port(self, address: str, server_credentials: grpc.ServerCredentials) -> None:
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def stop(self, grace: float | None = None):
        raise NotImplementedError()


if __name__ == "__main__":
    # response = requests.post(
    #     "http://envoy:50051/v0/issues/severity-score",
    #     json=SeverityRequest(message="Oh not big problem!").model_dump(mode='json')
    # )
    #
    # response.raise_for_status()
    # print(response.json())

    with grpc.secure_channel(
        "envoy:50051",
        grpc.ssl_channel_credentials(
            root_certificates=open("/app/certs/ca/ca.pem", "rb").read(),
            private_key=open("/app/certs/client/client-key.pem", "rb").read(),
            certificate_chain=open("/app/certs/client/client.pem", "rb").read(),
        ),
    ) as channel:
        stub = SeverityStub(channel)
        response = stub.GetIssueScore(ScoreRequest(message="Oh no big problem!"))
        print("severity=", response.severity)
