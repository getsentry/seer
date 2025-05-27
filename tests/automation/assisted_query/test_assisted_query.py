from unittest.mock import MagicMock, patch

from seer.automation.agent.client import LlmClient
from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Usage,
)
from seer.automation.assisted_query.assisted_query import (
    create_query_from_natural_language,
    translate_query,
)
from seer.automation.assisted_query.models import (
    Chart,
    CreateCacheResponse,
    ModelResponse,
    RelevantFieldsResponse,
    TranslateRequest,
    TranslateResponses,
)
from seer.rpc import RpcClient


class TestAssistedQuery:
    def setup_method(self):
        self.mock_llm_client = MagicMock(spec=LlmClient)
        self.mock_rpc_client = MagicMock(spec=RpcClient)

    @patch("seer.automation.assisted_query.assisted_query.LlmClient")
    def test_translate_query_success(self, mock_llm_client_class):
        mock_llm_client = MagicMock()
        mock_llm_client.get_cache.return_value = "test-cache"
        mock_llm_client_class.return_value = mock_llm_client

        request = TranslateRequest(
            natural_language_query="Show me the slowest POST requests in the last 24 hours",
            org_id=1,
            project_ids=[1, 2],
        )

        with patch(
            "seer.automation.assisted_query.assisted_query.create_query_from_natural_language"
        ) as mock_create_query:
            mock_create_query.return_value = LlmGenerateStructuredResponse(
                [
                    ModelResponse(
                        explanation="This is a test explanation",
                        query="error",
                        stats_period="24h",
                        group_by=["project"],
                        visualization=[
                            Chart(
                                chart_type=1,
                                y_axes=["Test y-axis 1", "Test y-axis 2"],
                            )
                        ],
                        sort="-count",
                        confidence_score=0.95,
                    ),
                ],
                metadata=LlmResponseMetadata(
                    model="gemini-2.5-flash-preview-05-20",
                    provider_name=LlmProviderType.GEMINI,
                    usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                ),
            )

            response = translate_query(request)

            assert isinstance(response, TranslateResponses)
            assert response.responses[0].query == "error"
            assert response.responses[0].stats_period == "24h"
            assert response.responses[0].group_by == ["project"]
            assert response.responses[0].visualization == [
                Chart(
                    chart_type=1,
                    y_axes=["Test y-axis 1", "Test y-axis 2"],
                )
            ]
            assert response.responses[0].sort == "-count"

    @patch("seer.automation.assisted_query.assisted_query.create_query_from_natural_language")
    @patch("seer.automation.assisted_query.assisted_query.LlmClient")
    @patch("seer.automation.assisted_query.assisted_query.create_cache")
    def test_translate_query_cache_not_found(
        self, mock_create_cache, mock_llm_client_class, mock_create_query
    ):
        mock_llm_client = MagicMock()
        mock_llm_client.get_cache.return_value = None
        mock_llm_client_class.return_value = mock_llm_client

        org_id = 1
        project_ids = [1, 2]
        cache_display_name = f"{org_id}_{'-'.join(map(str, project_ids))}"

        mock_create_cache.return_value = CreateCacheResponse(
            success=True, message="Cache created successfully", cache_name=cache_display_name
        )

        request = TranslateRequest(
            natural_language_query="Show me errors in the last 24 hours",
            org_id=org_id,
            project_ids=project_ids,
        )

        mock_create_query.return_value = LlmGenerateStructuredResponse(
            [
                ModelResponse(
                    explanation="This is a test explanation",
                    query="error",
                    stats_period="24h",
                    group_by=["project"],
                    visualization=[Chart(chart_type=1, y_axes=["Test y-axis 1", "Test y-axis 2"])],
                    sort="-count",
                    confidence_score=0.95,
                )
            ],
            metadata=LlmResponseMetadata(
                model="gemini-2.5-flash-preview-05-20",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        response = translate_query(request)
        assert isinstance(response, TranslateResponses)
        assert response.responses[0].query == "error"
        assert response.responses[0].stats_period == "24h"
        assert response.responses[0].group_by == ["project"]
        assert response.responses[0].visualization == [
            Chart(chart_type=1, y_axes=["Test y-axis 1", "Test y-axis 2"])
        ]
        assert response.responses[0].sort == "-count"

    @patch("seer.automation.assisted_query.assisted_query.RpcClient")
    @patch("seer.automation.agent.client.LlmClient")
    def test_create_query_from_natural_language_success(
        self, mock_rpc_client_class, mock_llm_client_class
    ):
        mock_rpc_client = MagicMock()
        mock_llm_client = MagicMock()

        first_call = LlmGenerateStructuredResponse(
            RelevantFieldsResponse(fields=["span.op", "span.description"]),
            metadata=LlmResponseMetadata(
                model="gemini-2.5-flash-preview-05-20",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        second_call = LlmGenerateStructuredResponse(
            ModelResponse(
                explanation="This is a test explanation",
                query="error",
                stats_period="24h",
                group_by=["project"],
                visualization=[
                    Chart(
                        chart_type=1,
                        y_axes=["Test y-axis 1", "Test y-axis 2"],
                    )
                ],
                sort="-count",
                confidence_score=0.95,
            ),
            metadata=LlmResponseMetadata(
                model="gemini-2.5-flash-preview-05-20",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )
        mock_llm_client.generate_structured.side_effect = [first_call, second_call]

        mock_rpc_client.call.return_value = {
            "field_values": {
                "span.op": [{"value": "db"}, {"value": "function"}],
                "span.description": [
                    {"value": "GET /api/v1/users"},
                    {"value": "POST /api/v1/products"},
                ],
            }
        }

        mock_llm_client_class.return_value = mock_llm_client
        mock_rpc_client_class.return_value = mock_rpc_client

        response = create_query_from_natural_language(
            natural_language_query="Show me the slowest database operations in the last 24 hours",
            cache_name="test-cache",
            org_id=1,
            project_ids=[1, 2],
            rpc_client=mock_rpc_client,
            llm_client=mock_llm_client,
        )

        assert isinstance(response, LlmGenerateStructuredResponse)
        assert response.parsed == ModelResponse(
            explanation="This is a test explanation",
            query="error",
            stats_period="24h",
            group_by=["project"],
            visualization=[
                Chart(
                    chart_type=1,
                    y_axes=["Test y-axis 1", "Test y-axis 2"],
                )
            ],
            sort="-count",
            confidence_score=0.95,
        )
        assert mock_llm_client.generate_structured.call_count == 2

        mock_rpc_client.call.assert_called_once_with(
            "get_attribute_values",
            org_id=1,
            fields=["span.op", "span.description", "transaction"],
            project_ids=[1, 2],
            stats_period="48h",
            limit=200,
        )

    @patch("seer.automation.assisted_query.assisted_query.RpcClient")
    @patch("seer.automation.agent.client.LlmClient")
    def test_create_query_from_natural_language_no_field_values(
        self, mock_rpc_client_class, mock_llm_client_class
    ):
        mock_rpc_client = MagicMock()
        mock_llm_client = MagicMock()

        # Create responses for both test cases
        first_call_response = LlmGenerateStructuredResponse(
            RelevantFieldsResponse(fields=["span.op", "span.description"]),
            metadata=LlmResponseMetadata(
                model="gemini-2.5-flash-preview-05-20",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        second_call_response = LlmGenerateStructuredResponse(
            ModelResponse(
                explanation="This is a test explanation",
                query="error",
                stats_period="24h",
                group_by=["project"],
                visualization=[
                    Chart(
                        chart_type=1,
                        y_axes=["Test y-axis 1", "Test y-axis 2"],
                    )
                ],
                sort="-count",
                confidence_score=0.95,
            ),
            metadata=LlmResponseMetadata(
                model="gemini-2.5-flash-preview-05-20",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            ),
        )

        # Set up responses for both test cases (4 total calls)
        mock_llm_client.generate_structured.side_effect = [
            first_call_response,
            second_call_response,
            first_call_response,
            second_call_response,
        ]

        mock_llm_client_class.return_value = mock_llm_client
        mock_rpc_client_class.return_value = mock_rpc_client

        # Test case 1: field_values_response is None
        mock_rpc_client.call.return_value = None

        response = create_query_from_natural_language(
            natural_language_query="Show me the slowest database operations in the last 24 hours",
            cache_name="test-cache",
            org_id=1,
            project_ids=[1, 2],
            rpc_client=mock_rpc_client,
            llm_client=mock_llm_client,
        )

        assert isinstance(response, LlmGenerateStructuredResponse)
        assert response.parsed == ModelResponse(
            explanation="This is a test explanation",
            query="error",
            stats_period="24h",
            group_by=["project"],
            visualization=[
                Chart(
                    chart_type=1,
                    y_axes=["Test y-axis 1", "Test y-axis 2"],
                )
            ],
            sort="-count",
            confidence_score=0.95,
        )

        # Verify that get_attribute_values was called with correct parameters
        mock_rpc_client.call.assert_any_call(
            "get_attribute_values",
            org_id=1,
            fields=["span.op", "span.description", "transaction"],
            project_ids=[1, 2],
            stats_period="48h",
            limit=200,
        )

        # Test case 2: field_values_response is missing 'values' key
        mock_rpc_client.call.return_value = {"some_other_key": "value"}

        response = create_query_from_natural_language(
            natural_language_query="Show me the slowest database operations in the last 24 hours",
            cache_name="test-cache",
            org_id=1,
            project_ids=[1, 2],
            rpc_client=mock_rpc_client,
            llm_client=mock_llm_client,
        )

        assert isinstance(response, LlmGenerateStructuredResponse)
        assert response.parsed == ModelResponse(
            explanation="This is a test explanation",
            query="error",
            stats_period="24h",
            group_by=["project"],
            visualization=[
                Chart(
                    chart_type=1,
                    y_axes=["Test y-axis 1", "Test y-axis 2"],
                )
            ],
            sort="-count",
            confidence_score=0.95,
        )
