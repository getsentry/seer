import unittest
from unittest.mock import MagicMock, Mock, patch

from seer.assisted_query.create_cache import create_cache
from seer.assisted_query.models import CreateCacheRequest, CreateCacheResponse
from seer.automation.agent.client import GeminiProvider


@patch("seer.rpc.DummyRpcClient.call")
class TestCreateCache(unittest.TestCase):
    def setUp(self):
        self.organization_id = 1
        self.project_ids = [1, 2, 3]
        self.request = CreateCacheRequest(
            organization_id=self.organization_id, project_ids=self.project_ids
        )

    @patch("seer.assisted_query.create_cache.LlmClient")
    def test_create_cache_existing_cache(self, mock_llm_client, mock_rpc_client_call: Mock):
        """Test creating cache when it already exists"""
        mock_llm_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.get_cache.return_value = "existing-cache-name"

        response = create_cache(self.request)

        self.assertIsInstance(response, CreateCacheResponse)
        self.assertTrue(response.success)
        self.assertEqual(response.message, "Cache already exists")
        self.assertEqual(response.cache_name, "existing-cache-name")

        mock_llm_instance.get_cache.assert_called_once_with(
            display_name="test-org-1-2-3",
            model=GeminiProvider.model("gemini-2.0-flash-001"),
        )
        mock_rpc_client_call.assert_not_called()

    @patch("seer.assisted_query.create_cache.LlmClient")
    def test_create_cache_new_cache(self, mock_llm_client, mock_rpc_client_call: Mock):
        """Test creating a new cache"""
        mock_llm_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.get_cache.return_value = None
        mock_llm_instance.create_cache.return_value = "new-cache-name"

        mock_rpc_client_call.return_value = {
            "fields": ["field1", "field2"],
            "field_values": {"field1": ["value1"], "field2": ["value2"]},
        }

        response = create_cache(self.request)

        self.assertIsInstance(response, CreateCacheResponse)
        self.assertTrue(response.success)
        self.assertEqual(response.message, "Cache created successfully")
        self.assertEqual(response.cache_name, "new-cache-name")

        mock_llm_instance.get_cache.assert_called_once_with(
            display_name="test-org-1-2-3",
            model=GeminiProvider.model("gemini-2.0-flash-001"),
        )

        mock_rpc_client_call.assert_any_call(
            "get_fields",
            organization_id=self.organization_id,
            project_ids=self.project_ids,
            stats_period="48h",
        )
        mock_rpc_client_call.assert_any_call(
            "get_field_values",
            organization_id=self.organization_id,
            fields=["field1", "field2"],
            project_ids=self.project_ids,
            stats_period="48h",
            k=5,
        )

        mock_llm_instance.create_cache.assert_called_once()
