from pathlib import Path
from unittest.mock import Mock

import pytest
from google.api_core.exceptions import ClientError

from seer.automation.models import EAPTrace
from seer.automation.summarize.models import SummarizeTraceRequest, SummarizeTraceResponse
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
            summary="**Trace: SentryApp Installation Retrieval**\n\n- A `db` transaction named `/api/0/organizations/{organization_id_or_slug}/repos/` initiates a database query.\n- The query retrieves `sentry_sentryappinstallation` data based on `api_token_id` and `status`.\n- The database operation is nested within another identical transaction.\n- The nested transaction contains a span that executes the same database query.\n",
            key_observations="- The trace involves **duplicate nested transactions** performing the same database query, which is unusual.\n- The database query targets the `sentry_sentryappinstallation` table, suggesting a focus on application installations.\n- The query filters by `api_token_id` and `status`, indicating a search for specific installations.",
            performance_characteristics="- The overall trace duration is **1.236ms**, which is relatively fast.\n- The database query itself takes **1.236ms**, indicating it's the primary operation.\n- The speed suggests the database query is efficient, but the duplicate execution may be unnecessary.",
            suggested_investigations='- Investigate why the `db` transaction `89239f27c64d1ac7` with description `SELECT "sentry_sentryappinstallation"."id", "sentry_sentryappinstallation"."date_deleted", "sentry_sentryappinstallation"."sentry_app_id", "sentry_sentryappinstallation"."organization_id", "sentry_sentryappinstallation"."api_grant_id", "sentry_sentryappinstallation"."api_token_id", "sentry_sentryappinstallation"."uuid", "sentry_sentryappinstallation"."status", "sentry_sentryappinstallation"."date_added", "sentry_sentryappinstallation"."date_updated" FROM "sentry_sentryappinstallation" WHERE ("sentry_sentryappinstallation"."date_deleted" IS NULL AND "sentry_sentryappinstallation"."api_token_id" = %s AND "sentry_sentryappinstallation"."status" = %s) LIMIT 21` is nested and duplicated. This could be an area for optimization.\n',
        )

        assert isinstance(res, SummarizeTraceResponse)

        assert res.trace_id == expected_result.trace_id
        assert res.summary == expected_result.summary, "summary mismatch"
        assert res.key_observations == expected_result.key_observations, "key_observations mismatch"
        assert (
            res.performance_characteristics == expected_result.performance_characteristics
        ), "performance_characteristics mismatch"
        assert (
            res.suggested_investigations == expected_result.suggested_investigations
        ), "suggested_investigations mismatch"

        assert res.model_dump() == expected_result.model_dump()

    def test_summarize_trace_client_error(self, sample_request, mock_llm_client):
        mock_llm_client.generate_structured.side_effect = ClientError(
            "The trace is too large to summarize. Please try a smaller trace."
        )

        with pytest.raises(
            ClientError, match="The trace is too large to summarize. Please try a smaller trace."
        ):
            summarize_trace(sample_request, mock_llm_client)
