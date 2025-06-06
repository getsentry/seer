import datetime
from unittest.mock import MagicMock

import anthropic
import pytest

from seer.automation.agent.blacklist import LlmRegionBlacklistService
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.configuration import AppConfig
from seer.db import DbLlmRegionBlacklist, Session
from seer.dependency_injection import Module


class TestLlmRegionBlacklistService:
    """Test the LLM region blacklist service functionality"""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up database for each test"""
        # Clean up any existing blacklist entries
        with Session() as session:
            session.query(DbLlmRegionBlacklist).delete()
            session.commit()

    @pytest.fixture
    def mock_config_enabled(self):
        """Mock config with blacklisting enabled"""
        config = MagicMock(spec=AppConfig)
        config.LLM_REGION_BLACKLIST_ENABLED = True
        config.LLM_REGION_BLACKLIST_DURATION_SECONDS = 300  # 5 minutes
        config.SENTRY_REGION = "us"
        return config

    @pytest.fixture
    def mock_config_disabled(self):
        """Mock config with blacklisting disabled"""
        config = MagicMock(spec=AppConfig)
        config.LLM_REGION_BLACKLIST_ENABLED = False
        config.LLM_REGION_BLACKLIST_DURATION_SECONDS = 300
        config.SENTRY_REGION = "us"
        return config

    def test_add_to_blacklist_creates_new_entry(self, mock_config_enabled):
        """Test that adding to blacklist creates a new entry"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                failure_reason="Rate limit exceeded",
            )

        # Verify entry was created
        with Session() as session:
            entry = (
                session.query(DbLlmRegionBlacklist)
                .filter_by(
                    provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
                )
                .first()
            )

            assert entry is not None
            assert entry.failure_reason == "Rate limit exceeded"
            assert entry.failure_count == 1
            # Compare with UTC datetime, but handle timezone differences
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            # If the database entry is timezone-naive, make comparison compatible
            if entry.expires_at.tzinfo is None:
                now_utc = now_utc.replace(tzinfo=None)
            assert entry.expires_at > now_utc

    def test_add_to_blacklist_updates_existing_entry(self, mock_config_enabled):
        """Test that adding an existing entry updates it and increments failure count"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            # Add first time
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                failure_reason="First failure",
            )

            # Add second time
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                failure_reason="Second failure",
            )

        # Verify entry was updated
        with Session() as session:
            entries = (
                session.query(DbLlmRegionBlacklist)
                .filter_by(
                    provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
                )
                .all()
            )

            assert len(entries) == 1  # Should be only one entry
            entry = entries[0]
            assert entry.failure_reason == "Second failure"
            assert entry.failure_count == 2

    def test_add_to_blacklist_disabled_does_nothing(self, mock_config_disabled):
        """Test that adding to blacklist does nothing when disabled"""
        module = Module()
        module.constant(AppConfig, mock_config_disabled)

        with module:
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

        # Verify no entry was created
        with Session() as session:
            count = session.query(DbLlmRegionBlacklist).count()
            assert count == 0

    def test_is_region_blacklisted_returns_true_for_blacklisted(self, mock_config_enabled):
        """Test that is_region_blacklisted returns True for blacklisted regions"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            # Add to blacklist
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            # Check if blacklisted
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            assert is_blacklisted is True

    def test_is_region_blacklisted_returns_false_for_non_blacklisted(self, mock_config_enabled):
        """Test that is_region_blacklisted returns False for non-blacklisted regions"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            assert is_blacklisted is False

    def test_is_region_blacklisted_returns_false_for_expired(self, mock_config_enabled):
        """Test that is_region_blacklisted returns False for expired entries"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        # Manually create an expired entry
        past_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        with Session() as session:
            entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                blacklisted_at=past_time,
                expires_at=past_time + datetime.timedelta(minutes=1),  # Already expired
                failure_count=1,
            )
            session.add(entry)
            session.commit()

        with module:
            # Check if blacklisted (should be False due to expiry)
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            assert is_blacklisted is False

        # Verify expired entry was cleaned up
        with Session() as session:
            count = session.query(DbLlmRegionBlacklist).count()
            assert count == 0

    def test_is_region_blacklisted_disabled_returns_false(self, mock_config_disabled):
        """Test that is_region_blacklisted returns False when disabled"""
        module = Module()
        module.constant(AppConfig, mock_config_disabled)

        # Manually add an entry to DB (bypassing the service)
        with Session() as session:
            entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                blacklisted_at=datetime.datetime.now(datetime.timezone.utc),
                expires_at=datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(hours=1),
                failure_count=1,
            )
            session.add(entry)
            session.commit()

        with module:
            # Should return False because service is disabled
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            assert is_blacklisted is False

    def test_get_non_blacklisted_regions_filters_correctly(self, mock_config_enabled):
        """Test that get_non_blacklisted_regions filters out blacklisted regions"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            # Add some regions to blacklist
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic", model_name="claude-3-sonnet", region="eu-west-1"
            )

            candidate_regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
            filtered_regions = LlmRegionBlacklistService.get_non_blacklisted_regions(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                candidate_regions=candidate_regions,
            )

            assert filtered_regions == ["us-west-2", "ap-southeast-1"]

    def test_get_non_blacklisted_regions_empty_candidates(self, mock_config_enabled):
        """Test that get_non_blacklisted_regions handles empty candidate list"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            filtered_regions = LlmRegionBlacklistService.get_non_blacklisted_regions(
                provider_name="anthropic", model_name="claude-3-sonnet", candidate_regions=[]
            )

            assert filtered_regions == []

    def test_get_non_blacklisted_regions_disabled_returns_all(self, mock_config_disabled):
        """Test that get_non_blacklisted_regions returns all regions when disabled"""
        module = Module()
        module.constant(AppConfig, mock_config_disabled)

        with module:
            candidate_regions = ["us-east-1", "us-west-2", "eu-west-1"]
            filtered_regions = LlmRegionBlacklistService.get_non_blacklisted_regions(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                candidate_regions=candidate_regions,
            )

            assert filtered_regions == candidate_regions

    def test_clear_blacklist_for_region_removes_entry(self, mock_config_enabled):
        """Test that clear_blacklist_for_region removes the specified entry"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            # Add to blacklist first
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            # Clear the blacklist
            was_removed = LlmRegionBlacklistService.clear_blacklist_for_region(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            assert was_removed is True

        # Verify entry was removed
        with Session() as session:
            count = session.query(DbLlmRegionBlacklist).count()
            assert count == 0

    def test_clear_blacklist_for_region_returns_false_if_not_found(self, mock_config_enabled):
        """Test that clear_blacklist_for_region returns False if entry not found"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            was_removed = LlmRegionBlacklistService.clear_blacklist_for_region(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            assert was_removed is False

    def test_clear_blacklist_for_region_disabled_returns_false(self, mock_config_disabled):
        """Test that clear_blacklist_for_region returns False when disabled"""
        module = Module()
        module.constant(AppConfig, mock_config_disabled)

        with module:
            was_removed = LlmRegionBlacklistService.clear_blacklist_for_region(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

            assert was_removed is False

    def test_get_blacklist_status_returns_correct_info(self, mock_config_enabled):
        """Test that get_blacklist_status returns correct information"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            # Add multiple entries
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                failure_reason="Rate limit",
            )
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="eu-west-1",
                failure_reason="Timeout",
            )

            status = LlmRegionBlacklistService.get_blacklist_status(
                provider_name="anthropic", model_name="claude-3-sonnet"
            )

            assert len(status) == 2
            regions = {entry["region"] for entry in status}
            assert regions == {"us-east-1", "eu-west-1"}

            # Check structure of returned data
            for entry in status:
                assert "region" in entry
                assert "blacklisted_at" in entry
                assert "expires_at" in entry
                assert "failure_reason" in entry
                assert "failure_count" in entry
                assert "time_remaining_minutes" in entry
                assert entry["failure_count"] >= 1

    def test_get_blacklist_status_excludes_expired(self, mock_config_enabled):
        """Test that get_blacklist_status excludes expired entries"""
        module = Module()
        module.constant(AppConfig, mock_config_enabled)

        with module:
            # Add current entry
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
            )

        # Add expired entry manually
        past_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        with Session() as session:
            expired_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="eu-west-1",
                blacklisted_at=past_time,
                expires_at=past_time + datetime.timedelta(minutes=1),
                failure_count=1,
            )
            session.add(expired_entry)
            session.commit()

        with module:
            status = LlmRegionBlacklistService.get_blacklist_status(
                provider_name="anthropic", model_name="claude-3-sonnet"
            )

            # Should only return the non-expired entry
            assert len(status) == 1
            assert status[0]["region"] == "us-east-1"

    def test_get_blacklist_status_disabled_returns_empty(self, mock_config_disabled):
        """Test that get_blacklist_status returns empty list when disabled"""
        module = Module()
        module.constant(AppConfig, mock_config_disabled)

        with module:
            status = LlmRegionBlacklistService.get_blacklist_status(
                provider_name="anthropic", model_name="claude-3-sonnet"
            )

            assert status == []


class TestLlmClientBlacklistIntegration:
    """Test integration of blacklisting with LlmClient"""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up database for each test"""
        with Session() as session:
            session.query(DbLlmRegionBlacklist).delete()
            session.commit()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=AppConfig)
        config.LLM_REGION_BLACKLIST_ENABLED = True
        config.LLM_REGION_BLACKLIST_DURATION_SECONDS = 300
        config.SENTRY_REGION = "us"
        config.DEV = False
        config.GOOGLE_CLOUD_PROJECT = "test-project"
        return config

    @pytest.fixture
    def anthropic_provider(self):
        """Create an Anthropic provider with multiple regions"""
        provider = AnthropicProvider.model(
            model_name="claude-3-sonnet-20240620", region=None  # Will use region preference
        )
        return provider

    @pytest.fixture
    def gemini_provider(self):
        """Create a Gemini provider with multiple regions"""
        provider = GeminiProvider.model(
            model_name="gemini-1.5-pro", region=None  # Will use region preference
        )
        return provider

    def test_get_regions_to_try_filters_blacklisted(self, mock_config, anthropic_provider):
        """Test that _get_regions_to_try filters out blacklisted regions"""
        module = Module()
        module.constant(AppConfig, mock_config)

        with module:
            client = LlmClient()

            # Blacklist one region
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name=anthropic_provider.provider_name,
                model_name=anthropic_provider.model_name,
                region="europe-west4",
            )

            # Get regions to try
            regions = client._get_regions_to_try(anthropic_provider)

            # Should not include the blacklisted region
            assert "europe-west4" not in regions
            # Should still have other regions available
            assert len(regions) > 0

    def test_get_regions_to_try_with_single_region_not_filtered(self, mock_config):
        """Test that single region providers are not filtered by blacklist"""
        module = Module()
        module.constant(AppConfig, mock_config)

        with module:
            client = LlmClient()

            # Create provider with single specific region
            provider = AnthropicProvider.model(
                model_name="claude-3-sonnet-20240620", region="us-east-1"
            )

            # Blacklist that region
            LlmRegionBlacklistService.add_to_blacklist(
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                region="us-east-1",
            )

            # Should still return the region (single region not filtered)
            regions = client._get_regions_to_try(provider)
            assert regions == ["us-east-1"]

    def test_get_regions_to_try_with_no_regions_available(self, mock_config, anthropic_provider):
        """Test behavior when all regions are blacklisted"""
        module = Module()
        module.constant(AppConfig, mock_config)

        with module:
            client = LlmClient()

            # Get all available regions first
            original_regions = client._get_regions_to_try(anthropic_provider)

            # Blacklist all regions
            for region in original_regions:
                if region:  # Skip None values
                    LlmRegionBlacklistService.add_to_blacklist(
                        provider_name=anthropic_provider.provider_name,
                        model_name=anthropic_provider.model_name,
                        region=region,
                    )

            # Should return empty list or only None values
            regions = client._get_regions_to_try(anthropic_provider)
            non_none_regions = [r for r in regions if r is not None]
            assert len(non_none_regions) == 0

    def test_fallback_adds_to_blacklist_on_retryable_exception(self, mock_config):
        """Test that fallback mechanism adds regions to blacklist on retryable exceptions"""
        module = Module()
        module.constant(AppConfig, mock_config)

        with module:
            client = LlmClient()

            # Create a provider that will fail
            provider = AnthropicProvider.model(
                model_name="claude-3-sonnet-20240620", region="us-east-1"
            )

            # Mock the operation to raise a retryable exception
            def failing_operation(model_to_use):
                raise anthropic.RateLimitError(
                    "Rate limit exceeded", response=MagicMock(), body=None
                )

            # This should fail and add to blacklist
            with pytest.raises(anthropic.RateLimitError):
                client._execute_with_fallback(
                    models=[provider],
                    operation_name="Test operation",
                    operation_func=failing_operation,
                )

            # Verify region was added to blacklist
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                region="us-east-1",
            )
            assert is_blacklisted is True

    def test_fallback_does_not_add_to_blacklist_on_non_retryable_exception(self, mock_config):
        """Test that fallback mechanism doesn't add regions to blacklist on non-retryable exceptions"""
        module = Module()
        module.constant(AppConfig, mock_config)

        with module:
            client = LlmClient()

            provider = AnthropicProvider.model(
                model_name="claude-3-sonnet-20240620", region="us-east-1"
            )

            # Mock the operation to raise a non-retryable exception
            def failing_operation(model_to_use):
                raise ValueError("Invalid input")

            # This should fail but not add to blacklist
            with pytest.raises(ValueError):
                client._execute_with_fallback(
                    models=[provider],
                    operation_name="Test operation",
                    operation_func=failing_operation,
                )

            # Verify region was NOT added to blacklist
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                region="us-east-1",
            )
            assert is_blacklisted is False

    def test_multiple_providers_fallback_with_blacklisting(self, mock_config):
        """Test fallback through multiple providers with blacklisting"""
        module = Module()
        module.constant(AppConfig, mock_config)

        with module:
            client = LlmClient()

            provider1 = AnthropicProvider.model(
                model_name="claude-3-sonnet-20240620", region="us-east-1"
            )
            provider2 = GeminiProvider.model(model_name="gemini-1.5-pro", region="us-central1")

            call_count = 0

            def failing_then_succeeding_operation(model_to_use):
                nonlocal call_count
                call_count += 1

                if call_count == 1:
                    # First call (provider1) fails with retryable error
                    raise anthropic.RateLimitError("Rate limit", response=MagicMock(), body=None)
                else:
                    # Second call (provider2) succeeds
                    return "success"

            # Should succeed on second provider
            result = client._execute_with_fallback(
                models=[provider1, provider2],
                operation_name="Test operation",
                operation_func=failing_then_succeeding_operation,
            )

            assert result == "success"
            assert call_count == 2

            # Verify first provider's region was blacklisted
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name=provider1.provider_name,
                model_name=provider1.model_name,
                region="us-east-1",
            )
            assert is_blacklisted is True

            # Verify second provider's region was NOT blacklisted
            is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
                provider_name=provider2.provider_name,
                model_name=provider2.model_name,
                region="us-central1",
            )
            assert is_blacklisted is False
