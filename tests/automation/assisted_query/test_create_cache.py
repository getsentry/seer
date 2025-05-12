import unittest
from unittest.mock import MagicMock, Mock, patch

from seer.automation.agent.client import GeminiProvider
from seer.automation.assisted_query.create_cache import create_cache
from seer.automation.assisted_query.models import CreateCacheRequest, CreateCacheResponse


@patch("seer.rpc.DummyRpcClient.call")
class TestCreateCache(unittest.TestCase):
    def setUp(self):
        self.org_id = 1
        self.project_ids = [1, 2, 3]
        self.request = CreateCacheRequest(org_id=self.org_id, project_ids=self.project_ids)

    @patch("seer.automation.assisted_query.create_cache.LlmClient")
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
            display_name="1_1-2-3",
            model=GeminiProvider.model("gemini-2.0-flash-001"),
        )
        mock_rpc_client_call.assert_not_called()

    @patch("seer.automation.assisted_query.create_cache.LlmClient")
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
            display_name="1_1-2-3",
            model=GeminiProvider.model("gemini-2.0-flash-001"),
        )

        mock_rpc_client_call.assert_any_call(
            "get_attribute_names",
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
        )
        mock_rpc_client_call.assert_any_call(
            "get_attribute_values",
            fields=["field1", "field2"],
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
            limit=5,
        )

        mock_llm_instance.create_cache.assert_called_once()

    @patch("seer.automation.assisted_query.create_cache.LlmClient")
    def test_create_cache_field_filtering(self, mock_llm_client, mock_rpc_client_call: Mock):
        """Test creating cache with field filtering when there are too many fields"""
        mock_llm_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.get_cache.return_value = None
        mock_llm_instance.create_cache.return_value = "new-cache-name"

        fields = []
        fields.extend([f"span.field{i}" for i in range(25)])
        fields.extend([f"transaction.field{i}" for i in range(25)])
        fields.extend([f"user.field{i}" for i in range(25)])

        # These fields should be filtered out
        fields.extend([f"tags[field{i},number]" for i in range(50)])
        fields.extend([f"other.field{i}" for i in range(150)])

        mock_rpc_client_call.return_value = {
            "fields": fields,
            "field_values": {field: ["value1"] for field in fields},
        }

        response = create_cache(self.request)

        assert isinstance(response, CreateCacheResponse)
        assert response.success is True
        assert response.message == "Cache created successfully"
        assert response.cache_name == "new-cache-name"

        mock_rpc_client_call.assert_any_call(
            "get_attribute_names",
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
        )

        get_attribute_values_call = next(
            call
            for call in mock_rpc_client_call.call_args_list
            if call[0][0] == "get_attribute_values"
        )

        filtered_fields = get_attribute_values_call[1]["fields"]

        assert len(filtered_fields) <= 125
        assert any(field.startswith("span.") for field in filtered_fields)
        assert any(field.startswith("transaction.") for field in filtered_fields)
        assert any(field.startswith("user.") for field in filtered_fields)
        assert not any("tags[" in field and ",number]" in field for field in filtered_fields)

        mock_llm_instance.create_cache.assert_called_once()
