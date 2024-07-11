import logging

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

from seer.handlers.severity_handler import SeverityHandler

logger = logging.getLogger(__name__)

# Initialize Sentry SDK
sentry_sdk.init(
    integrations=[
        LoggingIntegration(
            level=logging.DEBUG,  # Capture debug and above as breadcrumbs
        ),
    ]
)

_service = SeverityHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    return data
