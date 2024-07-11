import json
import logging

import numpy as np
import sentry_sdk
import torch
from ts.torch_handler.base_handler import BaseHandler

from seer.severity.severity_inference import (
    SeverityRequest,
    SeverityResponse,
    _init_classifier,
    _init_embeddings_model,
)

logger = logging.getLogger(__name__)


class SeverityHandler(BaseHandler):
    def initialize(self, context):
        self.context = context
        self.initialized = False

        # Load the model and other artifacts
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the embeddings model
        embeddings_path = f"{model_dir}/embeddings"
        print(embeddings_path)
        self.embeddings_model = _init_embeddings_model(embeddings_path)

        # Load the classifier model
        classifier_path = f"{model_dir}/classifier"
        self.classifier = _init_classifier(classifier_path)

        self.initialized = True

    def preprocess(self, data):
        """Preprocess the input data."""
        request = data[0].get("body")
        if isinstance(request, (bytes, bytearray)):
            request = request.decode("utf-8")
        request = json.loads(request)
        return SeverityRequest(**request)

    def inference(self, data):
        """Run the inference on the preprocessed data."""
        severity_request = data
        response = self.severity_score(severity_request)
        return response

    def postprocess(self, data):
        """Postprocess the inference output."""
        response = data
        return [json.dumps(response.dict())]

    def severity_score(self, data: SeverityRequest) -> SeverityResponse:
        """Predict the severity score for the given text using the pre-trained classifier."""
        with sentry_sdk.start_span(op="severity.embeddings"):
            embeddings = self.embeddings_model.encode(data.message, convert_to_numpy=True).reshape(
                -1
            )

        with sentry_sdk.start_span(op="severity.classification"):
            has_stacktrace = data.has_stacktrace

            handled = data.handled
            handled_true = 1 if handled is True else 0
            handled_false = 1 if handled is False else 0
            handled_unknown = 1 if handled is None else 0

            input_data = np.append(
                embeddings.reshape(1, -1),
                [[has_stacktrace, handled_true, handled_false, handled_unknown]],
                axis=1,
            )

            pred = self.classifier.predict_proba(input_data)[0][1]

        return SeverityResponse(severity=round(min(1.0, max(0.0, pred)), 2))


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
