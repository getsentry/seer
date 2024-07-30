import logging
import sys
from typing import Annotated

import structlog
from gunicorn.glogging import Logger as GunicornBaseLogger  # type: ignore[import-untyped]
from structlog import get_logger

from seer.dependency_injection import Labeled, Module, inject, injected

DefaultLoggingHandlers = Annotated[list[logging.Handler], Labeled("default")]
LogLevel = Annotated[int, Labeled("log_level")]
LoggingPrefixes = Annotated[list[str], Labeled("logging_prefixes")]

logging_module = Module()

logging_module.constant(LogLevel, logging.INFO)
logging_module.constant(LoggingPrefixes, ["seer.", "celery_app."])


@logging_module.provider
def default_handlers() -> DefaultLoggingHandlers:
    return [StructLogHandler(sys.stdout)]


@inject
def initialize_logs(
    prefixes: list[str],
    handlers: DefaultLoggingHandlers = injected,
    log_level: LogLevel = injected,
):
    for log_name in logging.root.manager.loggerDict:
        if any(log_name.startswith(prefix) for prefix in prefixes):
            logger = logging.getLogger(log_name)
            logger.setLevel(log_level)
            for handler in handlers:
                logger.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    )


class StructLogHandler(logging.StreamHandler):
    def get_log_kwargs(self, record, logger):
        kwargs = {k: v for k, v in vars(record).items() if k not in throwaways and v is not None}
        kwargs.update({"level": record.levelno, "event": record.msg})

        if record.args:
            # record.args inside of LogRecord.__init__ gets unrolled
            # if it's the shape `({},)`, a single item dictionary.
            # so we need to check for this, and re-wrap it because
            # down the line of structlog, it's expected to be this
            # original shape.
            if isinstance(record.args, (tuple, list)):
                kwargs["positional_args"] = record.args
            else:
                kwargs["positional_args"] = (record.args,)

        return kwargs

    def emit(self, record, logger=None):
        # If anyone wants to use the 'extra' kwarg to provide context within
        # structlog, we have to strip all of the default attributes from
        # a record because the RootLogger will take the 'extra' dictionary
        # and just turn them into attributes.
        try:
            if logger is None:
                logger = get_logger()

            logger.log(**self.get_log_kwargs(record=record, logger=logger))
        except Exception:
            if logging.raiseExceptions:
                raise


class GunicornHealthCheckFilterLogger(GunicornBaseLogger):
    def access(self, resp, req, environ, request_time):
        if "/health" not in req.path:
            super().access(resp, req, environ, request_time)


throwaways = frozenset(
    (
        "threadName",
        "thread",
        "created",
        "process",
        "processName",
        "args",
        "module",
        "filename",
        "levelno",
        "exc_text",
        "msg",
        "pathname",
        "lineno",
        "funcName",
        "relativeCreated",
        "levelname",
        "msecs",
    )
)

logging_module.enable()
