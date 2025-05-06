from pathlib import Path
from unittest.mock import Mock

import pytest

# from google.api_core.exceptions import ClientError
from google.genai.errors import ClientError

# from google.genai.types import Respons
from requests import Response

from seer.automation.models import EAPTrace
from seer.automation.summarize.models import (
    SpanInsight,
    SummarizeTraceRequest,
    SummarizeTraceResponse,
)
from seer.automation.summarize.traces import summarize_trace


class TestSummarizeTrace:
    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def sample_request(self):
        traces_dir = Path(__file__).parent / "fixtures" / "traces"
        traces: list[dict] = []
        for path in traces_dir.glob("trace_to_summarize*.json"):
            with path.open() as f:
                traces.append(EAPTrace.model_validate_json(f.read()))

        return SummarizeTraceRequest(
            trace_id="123",
            only_transactions=False,
            trace=EAPTrace(
                trace_id="123",
                trace=traces[0].trace,
            ),
        )

    @pytest.mark.vcr()
    def test_summarize_trace_success(self, sample_request):
        res = summarize_trace(sample_request)

        expected_result = SummarizeTraceResponse(
            trace_id="123",
            summary="This trace represents an API transaction dominated by a single database query. The trace exhibits an unusual structure with deeply nested spans that appear identical, suggesting a potential instrumentation issue.",
            key_observations="The trace consists solely of a single database SELECT query repeated across multiple nested spans. The nested db spans have identical descriptions, durations, timestamps, and even span IDs, which is highly irregular. The depth of nesting (3 levels) for a single, fast database operation is unexpected.",
            performance_characteristics="The trace is extremely fast (1.236 ms), indicating the database query itself is performant. The unusual nested structure doesn't appear to cause significant performance degradation in this instance, but it represents inefficient span creation.",
            suggested_investigations=[
                SpanInsight(
                    explanation="Investigate instrumentation for db spans as identical nesting suggests a configuration error.",
                    span_id="9c39a42585c71c3d",
                    span_op="db",
                )
            ],
        )

        assert isinstance(res, SummarizeTraceResponse)

        assert res.trace_id == expected_result.trace_id
        assert res.summary == expected_result.summary
        assert res.key_observations == expected_result.key_observations
        assert (
            res.performance_characteristics == expected_result.performance_characteristics
        ), "performance_characteristics mismatch"
        assert (
            res.suggested_investigations == expected_result.suggested_investigations
        ), "suggested_investigations mismatch"

        assert res.model_dump() == expected_result.model_dump()

    def test_summarize_trace_client_error(self, sample_request, mock_llm_client):

        res = Response()
        res.status_code = 400
        res._content = b'{"error": {"code": 400, "message": "The input token count (2000000) exceeds the maximum number of tokens allowed (1000000).", "status": "INVALID_ARGUMENT"}}'

        mock_llm_client.generate_structured.side_effect = ClientError(
            code=400,
            response=res,
        )

        with pytest.raises(
            ClientError,
        ):
            summarize_trace(sample_request, mock_llm_client)

        mock_llm_client.generate_structured.side_effect = Exception("Some other test issue")

        with pytest.raises(Exception, match="Some other test issue"):
            summarize_trace(sample_request, mock_llm_client)
