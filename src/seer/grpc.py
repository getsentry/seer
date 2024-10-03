import dataclasses
import logging
from collections import defaultdict
from concurrent import futures
from typing import Callable, Iterable

import grpc
import grpc.experimental
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from sentry_protos.seer.v1 import summarize_pb2, summarize_pb2_grpc
from sentry_sdk.integrations.grpc import GRPCIntegration

from celery_app.app import celery_app

# Use seer's module so that we get a flask object correctly.
from seer.app import app_module
from seer.bootup import bootup
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


@app_module.provider
def grpc_health_service(app_config: AppConfig = injected) -> health.HealthServicer:
    # Add health service
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(
            max_workers=app_config.GRPC_THREAD_POOL_SIZE
        ),
    )
    return health_servicer


class DummyIssueSummaryService(summarize_pb2_grpc.IssueSummaryServiceServicer):
    def Summarize(self, request, context):
        logger.info("Got summarize request!")
        return summarize_pb2.SummarizeResponse(group_id=2)


@app_module.provider
def issue_summary_service() -> summarize_pb2_grpc.IssueSummaryServiceServicer:
    return DummyIssueSummaryService()


@dataclasses.dataclass
# python grpc doesn't provide great interface for reflecting on services added during server creation.
# We create a wrapper proxy around a server that defers to an underlying implementation.
class ReflectableServer(grpc.Server):
    wrapped: grpc.Server
    service_handlers: dict[str, list[grpc.ServiceRpcHandler]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
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

    def add_insecure_port(self, *args, **kwds):
        return self.wrapped.add_insecure_port(*args, **kwds)

    def add_secure_port(self, *args, **kwds):
        return self.wrapped.add_secure_port(*args, **kwds)

    def start(self):
        self.wrapped.start()

    def stop(self, *args, **kwds):
        self.wrapped.stop(*args, **kwds)

    def wait_for_termination(self, *args, **kwds):
        return self.wrapped.wait_for_termination(*args, **kwds)


@app_module.provider
@dataclasses.dataclass
class HealthServicer(health.HealthServicer):
    wrapped: health.HealthServicer = injected
    # Set of service names that will notify the health service when they are ready.
    checks: set[str] = dataclasses.field(default_factory=set)

    def register_async_check(
        self, service_name: str
    ) -> Callable[[health_pb2.HealthCheckResponse.ServingStatus], None]:
        self.checks.add(service_name)

        def set_response(health_response: health_pb2.HealthCheckResponse.ServingStatus):
            self.wrapped.set(service_name, health_response)

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
                self.wrapped.set(service_name, health_pb2.HealthCheckResponse.SERVING)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)


@inject
def prepare_grpc_servers(
    config: AppConfig = injected,
    issue_summary: summarize_pb2_grpc.IssueSummaryServiceServicer = injected,
    health_servicer: HealthServicer = injected,
) -> list[grpc.Server]:
    server = ReflectableServer(
        grpc.server(futures.ThreadPoolExecutor(config.GRPC_THREAD_POOL_SIZE))
    )

    server_fallback_creds = grpc.insecure_server_credentials()
    server_creds = grpc.xds_server_credentials(server_fallback_creds)
    logger.info(f"Adding listener for service port {config.GRPC_SERVICE_PORT}")
    server.add_secure_port(f"0.0.0.0:{config.GRPC_SERVICE_PORT}", server_creds)

    maintenance_server = ReflectableServer(
        grpc.server(futures.ThreadPoolExecutor(config.GRPC_THREAD_POOL_SIZE))
    )
    logger.info(f"Adding listener for maintenance port {config.GRPC_MAINTENANCE_PORT}")
    maintenance_server.add_insecure_port(f"0.0.0.0:{config.GRPC_MAINTENANCE_PORT}")

    # Add new services right here.
    summarize_pb2_grpc.add_IssueSummaryServiceServicer_to_server(issue_summary, server)

    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, maintenance_server)
    reflection.enable_server_reflection(
        [*server.service_handlers.keys(), reflection.SERVICE_NAME, health.SERVICE_NAME],
        maintenance_server,
    )

    health_servicer.ensure_health_check_names(server)
    health_servicer.set_default_status(server)
    health_servicer.set_default_status(maintenance_server)

    return [server, maintenance_server]



@inject
def run_server(config: AppConfig = injected):
    app = bootup(
        start_model_loading=False,
        integrations=[GRPCIntegration()],
    )
    servers = prepare_grpc_servers()
    logger.info("Starting GRPC servers...")
    for server in servers:
        server.start()
    logger.info(
        f"Listening on maintenance {config.GRPC_MAINTENANCE_PORT} and service {config.GRPC_SERVICE_PORT}"
    )
    for server in servers:
        server.wait_for_termination()


@inject
def get_channel(config: AppConfig = injected):
    fallback_creds = grpc.experimental.insecure_channel_credentials()
    channel_creds = grpc.xds_channel_credentials(fallback_creds)
    return grpc.secure_channel(config.GRPC_TEST_ADDRESS, channel_creds)


@celery_app.task(time_limit=15)
def try_grpc_client():
    logger.info("Trying prepare channel")
    c = get_channel()
    with c:
        try:
            logger.info("Sending request")
            summarize_pb2_grpc.IssueSummaryServiceStub(c).Summarize(
                summarize_pb2.SummarizeRequest()
            )
        except Exception as e:
            logger.info("Failed!")
            raise e
        logger.info("Worked!!!")


if __name__ == "__main__":
    run_server()
