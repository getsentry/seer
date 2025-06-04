from unittest.mock import patch

import pytest
from pydantic import BaseModel

from seer.automation.agent.client import (
    AnthropicProvider,
    GeminiProvider,
    LlmClient,
    OpenAiProvider,
)
from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmGenerateTextResponse,
    LlmProviderDefaults,
    LlmProviderType,
)
from seer.automation.agent.tools import FunctionTool


class TestParameterResolution:
    """Test parameter resolution functionality comprehensively"""

    def test_parameter_resolution_no_defaults(self):
        """Test parameter resolution with no defaults provided"""
        llm_client = LlmClient()

        resolved = llm_client._resolve_parameters(
            defaults=None,
            temperature=0.8,
            max_tokens=500,
            seed=123,
            reasoning_effort="medium",
            timeout=60.0,
            first_token_timeout=50.0,
            inactivity_timeout=25.0,
        )

        assert resolved.temperature == 0.8
        assert resolved.max_tokens == 500
        assert resolved.seed == 123
        assert resolved.reasoning_effort == "medium"
        assert resolved.timeout == 60.0
        assert resolved.first_token_timeout == 50.0
        assert resolved.inactivity_timeout == 25.0

    def test_parameter_resolution_empty_defaults(self):
        """Test parameter resolution with empty defaults"""
        llm_client = LlmClient()

        empty_defaults = LlmProviderDefaults()

        resolved = llm_client._resolve_parameters(
            defaults=empty_defaults,
            temperature=0.9,
        )

        assert resolved.temperature == 0.9
        assert resolved.max_tokens is None
        assert resolved.seed is None
        assert resolved.reasoning_effort is None
        assert resolved.timeout is None
        assert resolved.first_token_timeout == 40.0  # Default constant
        assert resolved.inactivity_timeout == 20.0  # Default constant

    def test_parameter_resolution_all_defaults(self):
        """Test parameter resolution when all parameters have defaults"""
        llm_client = LlmClient()

        full_defaults = LlmProviderDefaults(
            temperature=0.3,
            max_tokens=2000,
            seed=999,
            reasoning_effort="low",
            timeout=120.0,
            first_token_timeout=100.0,
            inactivity_timeout=40.0,
        )

        # Don't override any parameters - should use all defaults
        resolved = llm_client._resolve_parameters(defaults=full_defaults)

        assert resolved.temperature == 0.3
        assert resolved.max_tokens == 2000
        assert resolved.seed == 999
        assert resolved.reasoning_effort == "low"
        assert resolved.timeout == 120.0
        assert resolved.first_token_timeout == 100.0
        assert resolved.inactivity_timeout == 40.0

    def test_parameter_resolution_selective_override(self):
        """Test parameter resolution with selective overrides"""
        llm_client = LlmClient()

        defaults = LlmProviderDefaults(
            temperature=0.5,
            max_tokens=1000,
            seed=42,
            reasoning_effort="medium",
            timeout=30.0,
            first_token_timeout=60.0,
            inactivity_timeout=20.0,
        )

        # Override only some parameters
        resolved = llm_client._resolve_parameters(
            defaults=defaults,
            temperature=0.1,  # Override
            # max_tokens not provided - use default
            seed=None,  # Explicitly None - use default
            reasoning_effort="high",  # Override
            # timeout not provided - use default
            first_token_timeout=80.0,  # Override
            # inactivity_timeout not provided - use default
        )

        assert resolved.temperature == 0.1  # Overridden
        assert resolved.max_tokens == 1000  # From defaults
        assert resolved.seed == 42  # From defaults (None means use default)
        assert resolved.reasoning_effort == "high"  # Overridden
        assert resolved.timeout == 30.0  # From defaults
        assert resolved.first_token_timeout == 80.0  # Overridden
        assert resolved.inactivity_timeout == 20.0  # From defaults

    def test_parameter_resolution_zero_values(self):
        """Test parameter resolution with zero/falsy values"""
        llm_client = LlmClient()

        defaults = LlmProviderDefaults(
            temperature=0.5,
            max_tokens=1000,
            seed=42,
            first_token_timeout=60.0,
            inactivity_timeout=20.0,
        )

        # Test that zero values are respected (not treated as None)
        resolved = llm_client._resolve_parameters(
            defaults=defaults,
            temperature=0.0,  # Zero should override default
            max_tokens=0,  # Zero should override default
            seed=0,  # Zero should override default
            first_token_timeout=0.0,  # Zero should override default
            inactivity_timeout=0.0,  # Zero should override default
        )

        assert resolved.temperature == 0.0
        assert resolved.max_tokens == 0
        assert resolved.seed == 0
        assert resolved.first_token_timeout == 0.0
        assert resolved.inactivity_timeout == 0.0

    def test_timeout_parameter_resolution_with_custom_values(self):
        """Test timeout parameter resolution with custom values"""
        llm_client = LlmClient()

        defaults = LlmProviderDefaults(
            first_token_timeout=90.0,
            inactivity_timeout=30.0,
        )

        resolved = llm_client._resolve_parameters(
            defaults=defaults,
            first_token_timeout=120.0,  # Override default
            inactivity_timeout=None,  # Use default
        )

        assert resolved.first_token_timeout == 120.0  # Overridden
        assert resolved.inactivity_timeout == 30.0  # From defaults


class TestProviderParameterResolution:
    """Test parameter resolution specific to different providers"""

    def test_openai_provider_parameter_resolution(self):
        """Test parameter resolution works correctly with OpenAI provider model creation"""
        # Test O-model defaults are preserved in resolution
        o1_model = OpenAiProvider.model(
            "o1-mini",
            temperature=None,  # Should use O-model default
            max_tokens=1500,  # Should override
            seed=123,  # Should set
        )

        # The model() method should have applied the defaults correctly
        assert o1_model.model_name == "o1-mini"
        assert o1_model.defaults.temperature == 1.0  # O-model default
        assert o1_model.defaults.max_tokens == 1500  # Overridden
        assert o1_model.defaults.seed == 123  # Set

        # Test regular model defaults
        gpt4_model = OpenAiProvider.model(
            "gpt-4",
            temperature=None,  # Should use regular default
            max_tokens=2000,  # Should override
        )

        assert gpt4_model.model_name == "gpt-4"
        assert gpt4_model.defaults.temperature == 0.0  # Regular default
        assert gpt4_model.defaults.max_tokens == 2000  # Overridden

    def test_anthropic_provider_parameter_resolution(self):
        """Test parameter resolution works correctly with Anthropic provider model creation"""
        claude_model = AnthropicProvider.model(
            "claude-3-5-sonnet@20240620",
            region="us-east-1",
            temperature=0.7,
            max_tokens=4000,
            timeout=90.0,
        )

        assert claude_model.model_name == "claude-3-5-sonnet@20240620"
        assert claude_model.region == "us-east-1"
        assert claude_model.defaults.temperature == 0.7
        assert claude_model.defaults.max_tokens == 4000
        assert claude_model.defaults.timeout == 90.0

    def test_gemini_provider_parameter_resolution(self):
        """Test parameter resolution works correctly with Gemini provider model creation"""
        gemini_model = GeminiProvider.model(
            "gemini-2.0-flash-001",
            region="us-central1",
            temperature=0.4,
            max_tokens=3000,
            seed=456,
            local_regions_only=True,
        )

        assert gemini_model.model_name == "gemini-2.0-flash-001"
        assert gemini_model.region == "us-central1"
        assert gemini_model.defaults.temperature == 0.4
        assert gemini_model.defaults.max_tokens == 3000
        assert gemini_model.defaults.seed == 456
        assert gemini_model.local_regions_only is True

    def test_provider_specific_parameters(self):
        """Test that provider-specific parameters are handled correctly"""
        # OpenAI-specific parameters
        openai_model = OpenAiProvider.model(
            "gpt-4",
            reasoning_effort="high",  # OpenAI-specific
            seed=123,
        )
        assert openai_model.defaults.reasoning_effort == "high"
        assert openai_model.defaults.seed == 123

        # Anthropic-specific parameters
        anthropic_model = AnthropicProvider.model(
            "claude-3-5-sonnet@20240620",
            region="us-west-2",  # Anthropic-specific region handling
            timeout=120.0,  # Anthropic-specific timeout
        )
        assert anthropic_model.region == "us-west-2"
        assert anthropic_model.defaults.timeout == 120.0

        # Gemini-specific parameters
        gemini_model = GeminiProvider.model(
            "gemini-2.0-flash-001",
            region="europe-west1",  # Gemini-specific region handling
            seed=789,  # Gemini supports seed
            local_regions_only=True,  # Gemini-specific parameter
        )
        assert gemini_model.region == "europe-west1"
        assert gemini_model.defaults.seed == 789
        assert gemini_model.local_regions_only is True


class TestFallbackBehavior:
    """Test fallback functionality and parameter resolution during fallback"""

    def test_parameter_resolution_fallback_behavior(self):
        """Test parameter resolution behavior during fallback scenarios"""
        llm_client = LlmClient()

        # Test that each model in a fallback list can have different parameter defaults
        models = [
            OpenAiProvider.model("gpt-4", temperature=0.1, max_tokens=100),
            OpenAiProvider.model("gpt-3.5-turbo", temperature=0.5, max_tokens=200),
            AnthropicProvider.model("claude-3-5-sonnet@20240620", temperature=0.3, max_tokens=150),
        ]

        # Each model should maintain its own defaults
        assert models[0].defaults.temperature == 0.1
        assert models[0].defaults.max_tokens == 100

        assert models[1].defaults.temperature == 0.5
        assert models[1].defaults.max_tokens == 200

        assert models[2].defaults.temperature == 0.3
        assert models[2].defaults.max_tokens == 150

        # Test parameter resolution for each provider type
        for model in models:
            if isinstance(model, OpenAiProvider):
                resolved = llm_client._resolve_parameters(
                    defaults=model.defaults,
                    temperature=0.9,  # Override
                )
                assert resolved.temperature == 0.9  # Should be overridden
                assert resolved.max_tokens == model.defaults.max_tokens  # Should use model default
            elif isinstance(model, AnthropicProvider):
                resolved = llm_client._resolve_parameters(
                    defaults=model.defaults,
                    max_tokens=500,  # Override
                )
                assert (
                    resolved.temperature == model.defaults.temperature
                )  # Should use model default
                assert resolved.max_tokens == 500  # Should be overridden

    def test_fallback_model_list_validation(self):
        """Test validation of fallback model lists"""

        # Valid fallback list
        models = [
            OpenAiProvider.model("gpt-4"),
            OpenAiProvider.model("gpt-3.5-turbo"),
        ]

        # Should not raise an exception when validating
        # This tests the internal validation logic
        assert len(models) > 0
        assert all(hasattr(model, "model_name") for model in models)
        assert all(hasattr(model, "provider_name") for model in models)

    def test_mixed_provider_fallback_list(self):
        """Test fallback behavior with mixed providers"""
        # Create models from different providers
        models = [
            OpenAiProvider.model("gpt-4", temperature=0.1),
            AnthropicProvider.model("claude-3-5-sonnet@20240620", temperature=0.3),
            GeminiProvider.model("gemini-2.0-flash-001", temperature=0.5),
        ]

        # Each should maintain its provider-specific settings
        assert models[0].provider_name == LlmProviderType.OPENAI
        assert models[1].provider_name == LlmProviderType.ANTHROPIC
        assert models[2].provider_name == LlmProviderType.GEMINI

        # Each should have its own temperature
        assert models[0].defaults.temperature == 0.1
        assert models[1].defaults.temperature == 0.3
        assert models[2].defaults.temperature == 0.5

    def test_region_fallback_behavior(self):
        """Test that fallback tries all regions for a model before moving to next model"""
        from unittest.mock import Mock, patch

        llm_client = LlmClient()

        # Create a model that should have region preferences
        base_model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

        # Mock the get_region_preference to return multiple regions
        with patch.object(
            base_model, "get_region_preference", return_value=["region1", "region2", "region3"]
        ):
            call_count = 0
            attempted_regions = []

            def mock_operation(model):
                nonlocal call_count, attempted_regions
                call_count += 1
                attempted_regions.append(model.region)

                # Fail for first two regions, succeed on third
                if call_count <= 2:
                    from anthropic import RateLimitError

                    raise RateLimitError(
                        message="Rate limit exceeded", body={}, response=Mock(status_code=429)
                    )
                return "success"

            # Should try all regions and succeed on the third
            result = llm_client._execute_with_fallback(
                models=[base_model], operation_name="Test Operation", operation_func=mock_operation
            )

            assert result == "success"
            assert call_count == 3
            assert attempted_regions == ["region1", "region2", "region3"]

    def test_region_fallback_explicit_region_no_fallback(self):
        """Test that when a region is explicitly set, no region fallback occurs"""

        llm_client = LlmClient()

        # Create a model with explicit region
        base_model = AnthropicProvider.model("claude-3-5-sonnet@20240620", region="explicit-region")

        call_count = 0
        attempted_regions = []

        def mock_operation(model):
            nonlocal call_count, attempted_regions
            call_count += 1
            attempted_regions.append(model.region)
            return "success"

        result = llm_client._execute_with_fallback(
            models=[base_model], operation_name="Test Operation", operation_func=mock_operation
        )

        assert result == "success"
        assert call_count == 1
        assert attempted_regions == ["explicit-region"]

    def test_region_fallback_multiple_models(self):
        """Test region fallback across multiple models"""
        from unittest.mock import Mock, patch

        llm_client = LlmClient()

        # Create multiple models
        model1 = AnthropicProvider.model("claude-3-5-sonnet@20240620")
        model2 = GeminiProvider.model("gemini-2.0-flash-001")

        call_count = 0
        operation_calls = []

        def mock_operation(model):
            nonlocal call_count, operation_calls
            call_count += 1
            operation_calls.append((model.model_name, model.region))

            # Fail for first model's regions, succeed on second model's first region
            if model.model_name == "claude-3-5-sonnet@20240620":
                from anthropic import RateLimitError

                raise RateLimitError(
                    message="Rate limit exceeded", body={}, response=Mock(status_code=429)
                )
            return "success"

        # Mock region preferences for both models
        with (
            patch.object(model1, "get_region_preference", return_value=["us-east5", "global"]),
            patch.object(model2, "get_region_preference", return_value=["us-central1", "global"]),
        ):

            result = llm_client._execute_with_fallback(
                models=[model1, model2],
                operation_name="Test Operation",
                operation_func=mock_operation,
            )

            assert result == "success"
            assert call_count == 3  # 2 calls for model1 regions + 1 call for model2's first region

            # Should have tried both regions of model1, then first region of model2
            expected_calls = [
                ("claude-3-5-sonnet@20240620", "us-east5"),
                ("claude-3-5-sonnet@20240620", "global"),
                ("gemini-2.0-flash-001", "us-central1"),
            ]
            assert operation_calls == expected_calls

    def test_region_fallback_no_region_preferences(self):
        """Test fallback behavior when model has no region preferences"""

        llm_client = LlmClient()

        # Create a model
        base_model = AnthropicProvider.model("claude-3-5-sonnet@20240620")

        call_count = 0
        attempted_regions = []

        def mock_operation(model):
            nonlocal call_count, attempted_regions
            call_count += 1
            attempted_regions.append(model.region)
            return "success"

        # Mock no region preferences
        with patch.object(base_model, "get_region_preference", return_value=None):
            result = llm_client._execute_with_fallback(
                models=[base_model], operation_name="Test Operation", operation_func=mock_operation
            )

            assert result == "success"
            assert call_count == 1
            assert attempted_regions == [None]  # Should try with None region


class TestParameterResolutionIntegration:
    """Integration tests for parameter resolution with actual LLM calls"""

    @pytest.mark.vcr()
    def test_parameter_resolution_integration_single_model(self):
        """Test that parameter resolution works in actual LLM generation with single model"""
        llm_client = LlmClient()

        # Create a model with specific defaults
        model = OpenAiProvider.model(
            "gpt-3.5-turbo",
            temperature=0.1,  # Very low temperature for deterministic output
            max_tokens=50,  # Limited tokens
            seed=42,  # Fixed seed
        )

        response = llm_client.generate_text(
            prompt="Say hello",
            model=model,
            # Don't override any parameters - should use model defaults
        )

        assert isinstance(response, LlmGenerateTextResponse)
        assert response.message.content is not None
        assert response.metadata.model == "gpt-3.5-turbo"
        assert response.metadata.usage.total_tokens > 0

    @pytest.mark.vcr()
    def test_parameter_resolution_integration_models_list(self):
        """Test that parameter resolution works in actual LLM generation with models list"""
        llm_client = LlmClient()

        # Create models with different defaults
        models = [
            OpenAiProvider.model(
                "gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=100,
                seed=123,
            ),
            OpenAiProvider.model(
                "gpt-4o-mini",
                temperature=0.3,
                max_tokens=150,
                seed=456,
            ),
        ]

        response = llm_client.generate_text(
            prompt="Say hello",
            models=models,
            # Override some parameters at call time
            temperature=0.0,  # Should override model defaults
            seed=999,  # Should override model defaults
        )

        assert isinstance(response, LlmGenerateTextResponse)
        assert response.message.content is not None
        assert response.metadata.model == "gpt-3.5-turbo"  # First model should be used
        assert response.metadata.usage.total_tokens > 0

    @pytest.mark.vcr()
    def test_parameter_resolution_with_structured_generation(self):
        """Test parameter resolution with structured generation"""
        llm_client = LlmClient()

        class TestStructure(BaseModel):
            name: str
            age: int

        # Create model with specific defaults
        model = OpenAiProvider.model(
            "gpt-4o-mini-2024-07-18",
            temperature=0.1,
            seed=42,
        )

        response = llm_client.generate_structured(
            prompt="Generate a person named Alice aged 25",
            model=model,
            response_format=TestStructure,
            # max_tokens override at call time
            max_tokens=100,
        )

        assert isinstance(response, LlmGenerateStructuredResponse)
        assert response.parsed.name == "Alice"
        assert response.parsed.age == 25
        assert response.metadata.model == "gpt-4o-mini-2024-07-18"

    @pytest.mark.vcr()
    def test_parameter_resolution_with_tools(self):
        """Test parameter resolution with tool usage"""
        llm_client = LlmClient()

        tools = [
            FunctionTool(
                name="test_function",
                description="A test function",
                parameters=[
                    {
                        "name": "x",
                        "type": "string",
                        "description": 'The string "Hello"',
                    },
                ],
                fn=lambda x: x,
            )
        ]

        # Create model with specific defaults
        model = OpenAiProvider.model(
            "gpt-3.5-turbo",
            temperature=0.1,
            seed=42,
        )

        response = llm_client.generate_text(
            prompt="Invoke test_function please!",
            model=model,
            tools=tools,
        )

        assert isinstance(response, LlmGenerateTextResponse)
        assert response.message.tool_calls is not None
        assert len(response.message.tool_calls) > 0
        assert response.message.tool_calls[0].function == "test_function"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in parameter resolution"""

    def test_parameter_resolution_with_conflicting_configs(self):
        """Test parameter resolution when there are conflicting configurations"""
        # Test that call-time parameters always take precedence
        model = OpenAiProvider.model(
            "gpt-4",
            temperature=0.1,  # Model default
            max_tokens=100,  # Model default
        )

        llm_client = LlmClient()
        resolved = llm_client._resolve_parameters(
            defaults=model.defaults,
            temperature=0.9,  # Call-time override
            max_tokens=500,  # Call-time override
        )

        assert resolved.temperature == 0.9  # Call-time should win
        assert resolved.max_tokens == 500  # Call-time should win

    def test_parameter_resolution_boundary_values(self):
        """Test parameter resolution with boundary values"""
        llm_client = LlmClient()

        # Test extreme values
        resolved = llm_client._resolve_parameters(
            defaults=None,
            temperature=0.0,  # Minimum
            max_tokens=1,  # Minimum practical value
            seed=-1,  # Negative value
            first_token_timeout=0.1,  # Very small timeout
            inactivity_timeout=0.1,  # Very small timeout
        )

        assert resolved.temperature == 0.0
        assert resolved.max_tokens == 1
        assert resolved.seed == -1
        assert resolved.first_token_timeout == 0.1
        assert resolved.inactivity_timeout == 0.1

    def test_parameter_resolution_with_none_values(self):
        """Test parameter resolution handles None values correctly"""
        llm_client = LlmClient()

        defaults = LlmProviderDefaults(
            temperature=0.5,
            max_tokens=1000,
        )

        # None values should fallback to defaults
        resolved = llm_client._resolve_parameters(
            defaults=defaults,
            temperature=None,  # Should use default
            max_tokens=None,  # Should use default
            seed=None,  # Should remain None (no default)
        )

        assert resolved.temperature == 0.5  # From defaults
        assert resolved.max_tokens == 1000  # From defaults
        assert resolved.seed is None  # No default, remains None
