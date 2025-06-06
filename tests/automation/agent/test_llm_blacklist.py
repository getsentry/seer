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

    def test_add_to_blacklist_creates_new_entry(self):
        """Test that adding to blacklist creates a new entry"""
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

    def test_add_to_blacklist_updates_existing_entry(self):
        """Test that adding an existing non-expired entry extends it and increments failure count"""
        # Add first time
        LlmRegionBlacklistService.add_to_blacklist(
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            region="us-east-1",
            failure_reason="First failure",
        )

        # Add second time (should extend the existing entry)
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

    def test_add_to_blacklist_creates_new_entry_after_expiry(self):
        """Test that adding to blacklist creates a new entry when previous one has expired"""
        # Manually create an expired entry
        past_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        with Session() as session:
            expired_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                blacklisted_at=past_time,
                expires_at=past_time + datetime.timedelta(minutes=1),  # Already expired
                failure_count=5,  # Had multiple failures before
                failure_reason="Old failure",
            )
            session.add(expired_entry)
            session.commit()

        # Add to blacklist again (should create new entry since old one expired)
        LlmRegionBlacklistService.add_to_blacklist(
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            region="us-east-1",
            failure_reason="New failure",
        )

        # Verify new entry was created with fresh failure count
        with Session() as session:
            entries = (
                session.query(DbLlmRegionBlacklist)
                .filter_by(
                    provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
                )
                .all()
            )

            # Should have only one entry (expired one should be cleaned up)
            assert len(entries) == 1
            entry = entries[0]
            assert entry.failure_reason == "New failure"
            assert entry.failure_count == 1  # Should start fresh, not continue from expired entry
            # Verify it's a new entry (not expired)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if entry.expires_at.tzinfo is None:
                now_utc = now_utc.replace(tzinfo=None)
            assert entry.expires_at > now_utc

    def test_is_region_blacklisted_returns_true_for_blacklisted(self):
        """Test that is_region_blacklisted returns True for blacklisted regions"""
        # Add to blacklist
        LlmRegionBlacklistService.add_to_blacklist(
            provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
        )

        # Check if blacklisted
        is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
            provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
        )

        assert is_blacklisted is True

    def test_is_region_blacklisted_returns_false_for_non_blacklisted(self):
        """Test that is_region_blacklisted returns False for non-blacklisted regions"""
        is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
            provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
        )

        assert is_blacklisted is False

    def test_is_region_blacklisted_returns_false_for_expired(self):
        """Test that is_region_blacklisted returns False for expired entries"""
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

        # Check if blacklisted (should be False due to expiry)
        is_blacklisted = LlmRegionBlacklistService.is_region_blacklisted(
            provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
        )

        assert is_blacklisted is False

        # Verify expired entry is still present (not cleaned up automatically)
        with Session() as session:
            count = session.query(DbLlmRegionBlacklist).count()
            assert count == 1

    def test_get_non_blacklisted_regions_filters_correctly(self):
        """Test that get_non_blacklisted_regions filters out blacklisted regions"""
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

    def test_get_non_blacklisted_regions_empty_candidates(self):
        """Test that get_non_blacklisted_regions handles empty candidate list"""
        filtered_regions = LlmRegionBlacklistService.get_non_blacklisted_regions(
            provider_name="anthropic", model_name="claude-3-sonnet", candidate_regions=[]
        )

        assert filtered_regions == []

    def test_cleanup_expired_entries_removes_only_expired(self):
        """Test that cleanup_expired_entries removes only expired entries"""
        # Create both expired and non-expired entries
        now = datetime.datetime.now(datetime.timezone.utc)
        past_time = now - datetime.timedelta(hours=2)
        future_time = now + datetime.timedelta(hours=1)

        with Session() as session:
            # Add expired entry
            expired_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-east-1",
                blacklisted_at=past_time,
                expires_at=past_time + datetime.timedelta(minutes=1),  # Already expired
                failure_count=1,
            )
            # Add non-expired entry
            active_entry = DbLlmRegionBlacklist(
                provider_name="anthropic",
                model_name="claude-3-sonnet",
                region="us-west-2",
                blacklisted_at=now,
                expires_at=future_time,  # Not expired yet
                failure_count=1,
            )
            session.add(expired_entry)
            session.add(active_entry)
            session.commit()

        # Run cleanup
        with Session() as session:
            deleted_count = LlmRegionBlacklistService.cleanup_expired_entries(session, now)

        # Verify only expired entry was removed
        assert deleted_count == 1

        with Session() as session:
            remaining_entries = session.query(DbLlmRegionBlacklist).all()
            assert len(remaining_entries) == 1
            assert remaining_entries[0].region == "us-west-2"  # Non-expired entry should remain

    def test_concurrent_blacklist_additions_same_region(self):
        """Test that concurrent additions to blacklist for same region work correctly"""
        # This test simulates what could happen if multiple processes try to blacklist
        # the same region simultaneously

        # First, create an active blacklist
        LlmRegionBlacklistService.add_to_blacklist(
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            region="us-east-1",
            failure_reason="First failure",
        )

        # Get the current entry to check blacklisted_at time
        with Session() as session:
            first_entry = (
                session.query(DbLlmRegionBlacklist)
                .filter_by(
                    provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
                )
                .first()
            )
            first_blacklisted_at = first_entry.blacklisted_at

        # Add to blacklist again (should extend existing)
        LlmRegionBlacklistService.add_to_blacklist(
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            region="us-east-1",
            failure_reason="Second failure",
        )

        # Verify only one entry exists and it was extended
        with Session() as session:
            entries = (
                session.query(DbLlmRegionBlacklist)
                .filter_by(
                    provider_name="anthropic", model_name="claude-3-sonnet", region="us-east-1"
                )
                .all()
            )

            assert len(entries) == 1
            entry = entries[0]
            assert entry.failure_reason == "Second failure"
            assert entry.failure_count == 2
            # Should have same blacklisted_at (same blacklist period)
            assert entry.blacklisted_at == first_blacklisted_at


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
            regions = client._get_regions_to_try(anthropic_provider, num_models_to_try=1)

            # Should not include the blacklisted region
            assert "europe-west4" not in regions
            # Should still have other regions available
            assert len(regions) > 0

    def test_get_regions_to_try_with_single_region_not_filtered(self, mock_config):
        """Test that single region providers are not filtered by blacklist when only one model to try"""
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

            # Should still return the region (single region not filtered when only one model to try)
            regions = client._get_regions_to_try(provider, num_models_to_try=1)
            assert regions == ["us-east-1"]

    def test_get_regions_to_try_with_no_regions_available(self, mock_config, anthropic_provider):
        """Test behavior when all regions are blacklisted"""
        module = Module()
        module.constant(AppConfig, mock_config)

        with module:
            client = LlmClient()

            # Get all available regions first
            original_regions = client._get_regions_to_try(anthropic_provider, num_models_to_try=2)

            # Blacklist all regions
            for region in original_regions:
                if region:  # Skip None values
                    LlmRegionBlacklistService.add_to_blacklist(
                        provider_name=anthropic_provider.provider_name,
                        model_name=anthropic_provider.model_name,
                        region=region,
                    )

            # Should return empty list or only None values
            regions = client._get_regions_to_try(anthropic_provider, num_models_to_try=2)
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
