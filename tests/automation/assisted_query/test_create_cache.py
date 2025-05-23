import unittest
from unittest.mock import MagicMock

from seer.automation.agent.client import GeminiProvider
from seer.automation.assisted_query.create_cache import create_cache
from seer.automation.assisted_query.models import CreateCacheRequest, CreateCacheResponse


class TestCreateCache(unittest.TestCase):
    def setUp(self):
        self.org_id = 1
        self.project_ids = [1, 2, 3]
        self.request = CreateCacheRequest(org_id=self.org_id, project_ids=self.project_ids)
        self.mock_llm_client = MagicMock()
        self.mock_rpc_client = MagicMock()

    def test_create_cache_existing_cache(self):
        """Test creating cache when it already exists"""
        self.mock_llm_client.get_cache.return_value = "existing-cache-name"

        response = create_cache(
            self.request, llm_client=self.mock_llm_client, rpc_client=self.mock_rpc_client
        )

        assert isinstance(response, CreateCacheResponse)
        assert response.success is True
        assert response.message == "Cache already exists"
        assert response.cache_name == "existing-cache-name"

        self.mock_llm_client.get_cache.assert_called_once_with(
            display_name="1_1-2-3",
            model=GeminiProvider.model("gemini-2.5-flash-preview-05-20"),
        )
        self.mock_rpc_client.call.assert_not_called()

    def test_create_cache_new_cache(self):
        """Test creating a new cache"""
        self.mock_llm_client.get_cache.return_value = None
        self.mock_llm_client.create_cache.return_value = "new-cache-name"

        self.mock_rpc_client.call.return_value = {
            "fields": ["field1", "field2"],
            "field_values": {"field1": ["value1"], "field2": ["value2"]},
        }

        response = create_cache(
            self.request, llm_client=self.mock_llm_client, rpc_client=self.mock_rpc_client
        )

        assert isinstance(response, CreateCacheResponse)
        assert response.success is True
        assert response.message == "Cache created successfully"
        assert response.cache_name == "new-cache-name"

        self.mock_llm_client.get_cache.assert_called_once_with(
            display_name="1_1-2-3",
            model=GeminiProvider.model("gemini-2.5-flash-preview-05-20"),
        )

        self.mock_rpc_client.call.assert_any_call(
            "get_attribute_names",
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
        )
        self.mock_rpc_client.call.assert_any_call(
            "get_attribute_values",
            fields=["field1", "field2"],
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
            limit=15,
        )

        self.mock_llm_client.create_cache.assert_called_once()

    def test_create_cache_field_filtering(self):
        """Test creating cache with field filtering when there are too many fields"""
        self.mock_llm_client.get_cache.return_value = None
        self.mock_llm_client.create_cache.return_value = "new-cache-name"

        fields = []
        fields.extend([f"span.field{i}" for i in range(25)])
        fields.extend([f"transaction.field{i}" for i in range(25)])
        fields.extend([f"user.field{i}" for i in range(25)])

        # These fields should be filtered out
        fields.extend([f"tags[field{i},number]" for i in range(50)])
        fields.extend([f"other.field{i}" for i in range(150)])

        self.mock_rpc_client.call.return_value = {
            "fields": fields,
            "field_values": {field: ["value1"] for field in fields},
        }

        response = create_cache(
            self.request, llm_client=self.mock_llm_client, rpc_client=self.mock_rpc_client
        )

        assert isinstance(response, CreateCacheResponse)
        assert response.success is True
        assert response.message == "Cache created successfully"
        assert response.cache_name == "new-cache-name"

        self.mock_rpc_client.call.assert_any_call(
            "get_attribute_names",
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
        )

        get_attribute_values_call = next(
            call
            for call in self.mock_rpc_client.call.call_args_list
            if call[0][0] == "get_attribute_values"
        )

        filtered_fields = get_attribute_values_call[1]["fields"]

        assert len(filtered_fields) <= 125
        assert any(field.startswith("span.") for field in filtered_fields)
        assert any(field.startswith("transaction.") for field in filtered_fields)
        assert any(field.startswith("user.") for field in filtered_fields)
        assert not any("tags[" in field and ",number]" in field for field in filtered_fields)

        self.mock_llm_client.create_cache.assert_called_once()

    def test_create_cache_no_attribute_names_response(self):
        """Test creating cache when get_attribute_names returns None"""
        self.mock_llm_client.get_cache.return_value = None
        self.mock_llm_client.create_cache.return_value = "new-cache-name"

        self.mock_rpc_client.call.return_value = None

        response = create_cache(
            self.request, llm_client=self.mock_llm_client, rpc_client=self.mock_rpc_client
        )

        assert isinstance(response, CreateCacheResponse)
        assert response.success is True
        assert response.message == "Cache created successfully"
        assert response.cache_name == "new-cache-name"

        self.mock_rpc_client.call.assert_any_call(
            "get_attribute_names",
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
        )

    def test_create_cache_missing_fields_key(self):
        """Test creating cache when get_attribute_names response is missing 'fields' key"""
        self.mock_llm_client.get_cache.return_value = None
        self.mock_llm_client.create_cache.return_value = "new-cache-name"

        self.mock_rpc_client.call.return_value = {"some_other_key": "value"}

        response = create_cache(
            self.request, llm_client=self.mock_llm_client, rpc_client=self.mock_rpc_client
        )

        assert isinstance(response, CreateCacheResponse)
        assert response.success is True
        assert response.message == "Cache created successfully"
        assert response.cache_name == "new-cache-name"

        self.mock_rpc_client.call.assert_any_call(
            "get_attribute_names",
            org_id=self.org_id,
            project_ids=self.project_ids,
            stats_period="48h",
        )
