import dataclasses
from collections import defaultdict
from concurrent import futures
from typing import Annotated, Callable, Iterable

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from sentry_protos.seer.v1 import summarize_pb2, summarize_pb2_grpc

from seer.bootup import module
from seer.configuration import AppConfig
from seer.dependency_injection import Labeled, inject, injected


@module.provider
def grpc_thread_pool(
    app_config: AppConfig,
) -> Annotated[futures.ThreadPoolExecutor, Labeled("grpc_thread_pool")]:
    return futures.ThreadPoolExecutor(max_workers=app_config.GRPC_THREAD_POOL_SIZE)


@module.provider
def grpc_health_service(app_config: AppConfig) -> health.HealthServicer:
    # Add health service
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(
            max_workers=app_config.GRPC_THREAD_POOL_SIZE
        ),
    )
    return health_servicer


@dataclasses.dataclass
# python grpc doesn't provide great interface for reflecting on services added during server creation.
# We create a wrapper proxy around a server that defers to an underlying implementation.
class ReflectableServer(grpc.Server):
    wrapped: grpc.Server
    service_handlers: dict[str, list[grpc.ServiceRpcHandler]] = dataclasses.field(
        default_factory=defaultdict
    )

    def add_generic_rpc_handlers(
        self,
        generic_rpc_handlers: Iterable[grpc.GenericRpcHandler],
    ):
        for handler in generic_rpc_handlers:
            if isinstance(handler, grpc.ServiceRpcHandler):
                self.service_handlers[handler.service_name()].append(handler)
        self.wrapped.add_generic_rpc_handlers(generic_rpc_handlers)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)


@dataclasses.dataclass
class HealthServicer(health.HealthServicer):
    wrapped: health.HealthServicer
    # Set of service names that will notify the health service when they are ready.
    checks: set[str] = dataclasses.field(default_factory=set)

    def register_async_check(
        self, service_name: str
    ) -> Callable[[health_pb2.HealthCheckResponse.ServingStatus], None]:
        self.checks.add(service_name)

        def set_response(health_response: health_pb2.HealthCheckResponse.ServingStatus):
            self.set(service_name, health_response)

        return set_response

    def ensure_health_check_names(self, server: ReflectableServer):
        for service_name in self.checks:
            if service_name not in server.service_handlers:
                raise ValueError(
                    f"Health check was registered for '{service_name}', but no such service exists.  Did you mean one of {', '.join(server.service_handlers.keys())}?"
                )

    def set_default_status(self, server: ReflectableServer):
        """
        Marks any services already added to the given server as SERVING if they do not have a registered
        a deferred health callback.
        """
        for service_name in server.service_handlers.keys():
            if service_name not in self.checks:
                self.set(service_name, health_pb2.HealthCheckResponse.SERVING)


@module.provider
def grpc_server(
    thread_pool: Annotated[futures.ThreadPoolExecutor, Labeled("grpc_thread_pool")] = injected,
    issue_summary: summarize_pb2_grpc.IssueSummaryServiceServicer = injected,
    health_servicer: HealthServicer = injected,
) -> grpc.Server:
    server = ReflectableServer(grpc.server(thread_pool))

    summarize_pb2_grpc.add_IssueSummaryServiceServicer_to_server(issue_summary, server)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    reflection.enable_server_reflection(list(server.service_handlers.keys()), server)
    health_servicer.ensure_health_check_names(server)
    health_servicer.set_default_status(server)

    return server


class DummyIssueSummaryService(summarize_pb2_grpc.IssueSummaryServiceServicer):
    def Summarize(self, request, context):
        return summarize_pb2.SummarizeResponse(group_id=2)


@inject
def run_server(server: grpc.Server):
    pass
