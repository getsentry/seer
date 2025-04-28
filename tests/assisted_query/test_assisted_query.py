import unittest
from unittest.mock import MagicMock, patch

from seer.assisted_query.assisted_query import create_query_from_natural_language, translate_query
from seer.assisted_query.models import (
    Chart,
    ModelResponse,
    RelevantFieldsResponse,
    TranslateRequest,
    TranslateResponse,
)
from seer.automation.agent.client import LlmClient
from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Usage,
)
from seer.rpc import RpcClient


class TestAssistedQuery(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock(spec=LlmClient)
        self.mock_rpc_client = MagicMock(spec=RpcClient)

    @patch("seer.assisted_query.assisted_query.LlmClient")
    def test_translate_query_success(self, mock_llm_client_class):
        mock_llm_client = MagicMock()
        mock_llm_client.get_cache.return_value = "test-cache"
        mock_llm_client_class.return_value = mock_llm_client

        request = TranslateRequest(
            natural_language_query="Show me the slowest POST requests in the last 24 hours",
            organization_slug="test-org",
            project_ids=[1, 2],
        )

        with patch(
            "seer.assisted_query.assisted_query.create_query_from_natural_language"
        ) as mock_create_query:
            mock_create_query.return_value = TranslateResponse(
                query="error",
                stats_period="24h",
                group_by=["project"],
                visualization=Chart(
                    chart_type=1,
                    y_axes=[["Test y-axis 1", "Test y-axis 2"]],
                ),
                sort="-count",
            )

            response = translate_query(request)

            self.assertIsInstance(response, TranslateResponse)
            self.assertEqual(response.query, "error")
            self.assertEqual(response.stats_period, "24h")
            self.assertEqual(response.group_by, ["project"])
            self.assertEqual(
                response.visualization,
                Chart(
                    chart_type=1,
                    y_axes=[["Test y-axis 1", "Test y-axis 2"]],
                ),
            )
            self.assertEqual(response.sort, "-count")

    @patch("seer.assisted_query.assisted_query.LlmClient")
    def test_translate_query_cache_not_found(self, mock_llm_client_class):
        mock_llm_client = MagicMock()
        mock_llm_client.get_cache.return_value = None
        mock_llm_client_class.return_value = mock_llm_client

        request = TranslateRequest(
            natural_language_query="Show me errors in the last 24 hours",
            organization_slug="test-org",
            project_ids=[1, 2],
        )

        with self.assertRaises(ValueError) as context:
            translate_query(request)
        self.assertEqual(str(context.exception), "Cache not found")

    @patch("seer.assisted_query.assisted_query.RpcClient")
    @patch("seer.assisted_query.assisted_query.LlmClient")
    def test_create_query_from_natural_language_success(
        self, mock_rpc_client_class, mock_llm_client_class
    ):
        mock_rpc_client = MagicMock()
        mock_llm_client = MagicMock()

        first_call = LlmGenerateStructuredResponse(
            parsed=RelevantFieldsResponse(fields=["span.op", "span.description"]),
            metadata=LlmResponseMetadata(
                model="gemini-2.0-flash-001",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        second_call = LlmGenerateStructuredResponse(
            parsed=ModelResponse(
                explanation="This is a test explanation",
                query="error",
                stats_period="24h",
                group_by=["project"],
                visualization=Chart(
                    chart_type=1,
                    y_axes=[["Test y-axis 1", "Test y-axis 2"]],
                ),
                sort="-count",
                confidence_score=0.95,
            ),
            metadata=LlmResponseMetadata(
                model="gemini-2.0-flash-001",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        mock_llm_client.generate_structured.side_effect = [first_call, second_call]

        mock_rpc_client.call.return_value = {
            "field_values": {
                "span.op": [{"value": "db", "count": 100}, {"value": "function", "count": 200}],
                "span.description": [
                    {"value": "GET /api/v1/users", "count": 100},
                    {"value": "POST /api/v1/products", "count": 200},
                ],
            }
        }

        mock_llm_client_class.return_value = mock_llm_client
        mock_rpc_client_class.return_value = mock_rpc_client

        response = create_query_from_natural_language(
            natural_language_query="Show me the slowest database operations in the last 24 hours",
            cache_name="test-cache",
            organization_slug="test-org",
            project_ids=[1, 2],
            rpc_client=mock_rpc_client,
            llm_client=mock_llm_client,
        )

        self.assertIsInstance(response, LlmGenerateStructuredResponse)
        self.assertEqual(
            response.parsed,
            ModelResponse(
                explanation="This is a test explanation",
                query="error",
                stats_period="24h",
                group_by=["project"],
                visualization=Chart(
                    chart_type=1,
                    y_axes=[["Test y-axis 1", "Test y-axis 2"]],
                ),
                sort="-count",
                confidence_score=0.95,
            ),
        )
        self.assertEqual(mock_llm_client.generate_structured.call_count, 2)

        mock_rpc_client.call.assert_called_once_with(
            "get_field_values",
            organization_slug="test-org",
            fields=["span.op", "span.description"],
            project_ids=[1, 2],
            stats_period="48h",
            k=150,
        )
