import unittest
from unittest.mock import MagicMock, patch

from seer.assisted_query.assisted_query import translate_query
from seer.assisted_query.models import Chart, TranslateRequest, TranslateResponse
from seer.automation.agent.client import LlmClient
from seer.rpc import RpcClient

# import pytest


class TestAssistedQuery(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock(spec=LlmClient)
        self.mock_rpc_client = MagicMock(spec=RpcClient)

    @patch("seer.assisted_query.assisted_query.LlmClient")
    def test_translate_query_success(self, mock_llm_client_class):
        # Setup
        mock_llm_client = MagicMock()
        mock_llm_client.get_cache.return_value = "test-cache"
        mock_llm_client_class.return_value = mock_llm_client

        request = TranslateRequest(
            natural_language_query="Show me the slowest POST requests in the last 24 hours",
            organization_slug="test-org",
            project_ids=[1, 2],
        )

        # Mock the create_query_from_natural_language function
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

            # Execute
            response = translate_query(request)

            # Assert
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
        # Setup
        mock_llm_client = MagicMock()
        mock_llm_client.get_cache.return_value = None
        mock_llm_client_class.return_value = mock_llm_client

        request = TranslateRequest(
            natural_language_query="Show me errors in the last 24 hours",
            organization_slug="test-org",
            project_ids=[1, 2],
        )

        # Execute and Assert
        with self.assertRaises(ValueError) as context:
            translate_query(request)
        self.assertEqual(str(context.exception), "Cache not found")

    # @patch("seer.assisted_query.assisted_query.RpcClient")
    # @pytest.mark.vcr()
    # def test_create_query_from_natural_language_success(
    #     self, mock_rpc_client_class, mock_llm_client_class
    # ):
    #     # Setup
    #     mock_rpc_client = MagicMock()
    #     mock_llm_client = MagicMock()

    #     # Mock the LLM client responses
    #     # mock_llm_client.generate_structured.side_effect = [
    #     #     RelevantFieldsResponse(fields=["error", "project"]),
    #     #     ModelResponse(
    #     #         query="error",
    #     #         stats_period="24h",
    #     #         group_by=["project"],
    #     #         visualization="table",
    #     #         sort="-count",
    #     #     ),
    #     # ]

    #     # Mock the RPC client response
    #     mock_rpc_client.call.return_value = {
    #         "field_values": {
    #             "error": ["error1", "error2"],
    #             "project": ["project1", "project2"],
    #         }
    #     }

    #     # mock_llm_client_class.return_value = mock_llm_client
    #     mock_rpc_client_class.return_value = mock_rpc_client

    #     # Execute
    #     response = create_query_from_natural_language(
    #         natural_language_query="Show me errors in the last 24 hours",
    #         cache_name="test-cache",
    #         organization_slug="test-org",
    #         project_ids=[1, 2],
    #         rpc_client=mock_rpc_client,
    #     )

    #     # Assert
    #     self.assertIsInstance(response, ModelResponse)
    #     self.assertEqual(response.query, "error")
    #     self.assertEqual(response.stats_period, "24h")
    #     self.assertEqual(response.group_by, ["project"])
    #     self.assertEqual(
    #         response.visualization,
    #         {
    #             "chart_type": 1,
    #             "y_axes": [["Test y-axis 1", "Test y-axis 2"]],
    #         },
    #     )
    #     self.assertEqual(response.sort, "-count")

    #     # Verify LLM client calls
    #     # self.assertEqual(mock_llm_client.generate_structured.call_count, 2)

    #     # Verify RPC client call
    #     mock_rpc_client.call.assert_called_once_with(
    #         "get_field_values",
    #         organization_slug="test-org",
    #         fields=["error", "project"],
    #         project_ids=[1, 2],
    #         stats_period="48h",
    #         k=150,
    #     )
