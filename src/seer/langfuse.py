import logging

from langfuse import Langfuse
from langfuse.decorators import langfuse_context

from seer.configuration import AppConfig
from seer.dependency_injection import Module, inject, injected

logger = logging.getLogger(__name__)

langfuse_module = Module()


@langfuse_module.provider
def provide_langfuse(config: AppConfig = injected) -> Langfuse:
    return Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
        enabled=bool(config.LANGFUSE_HOST),
    )


@inject
def append_langfuse_trace_tags(new_tags: list[str], langfuse: Langfuse = injected):
    """
    Appends traces to the current trace in the context.
    MUST BE RUN WITHIN A LANGFUSE TRACE!
    """
    try:
        trace_id = langfuse_context.get_current_trace_id()
        if trace_id:
            trace = langfuse.get_trace(trace_id)
            langfuse_context.update_current_trace(
                tags=(trace.tags or []) + new_tags,
            )
    except Exception as e:
        logger.exception(e)


@inject
def append_langfuse_observation_metadata(new_metadata: dict, langfuse: Langfuse = injected):
    """
    Appends metadata to the current observation in the context.
    MUST BE RUN WITHIN A LANGFUSE OBSERVATION!
    """
    try:
        langfuse_context.update_current_observation(
            metadata=new_metadata,
        )
    except Exception as e:
        logger.exception(e)


langfuse_module.enable()
