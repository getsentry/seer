import datetime
import logging
from typing import List, Optional

from sqlalchemy import and_, delete, select
from sqlalchemy.dialects.postgresql import insert

from seer.automation.codebase.utils import ensure_timezone_aware
from seer.configuration import AppConfig
from seer.db import DbLlmRegionBlacklist, Session
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class LlmRegionBlacklistService:
    """Service for managing temporarily blacklisted LLM provider regions"""

    @staticmethod
    @inject
    def is_region_blacklisted(
        provider_name: str,
        model_name: str,
        region: str,
        config: AppConfig = injected,
    ) -> bool:
        """Check if a specific provider/model/region combination is currently blacklisted"""
        if not config.LLM_REGION_BLACKLIST_ENABLED:
            return False

        with Session() as session:
            now = datetime.datetime.now(datetime.UTC)

            # First, clean up expired entries
            LlmRegionBlacklistService._cleanup_expired_entries(session, now)

            # Check if the region is blacklisted
            result = session.scalar(
                select(DbLlmRegionBlacklist.id).where(
                    and_(
                        DbLlmRegionBlacklist.provider_name == provider_name,
                        DbLlmRegionBlacklist.model_name == model_name,
                        DbLlmRegionBlacklist.region == region,
                        DbLlmRegionBlacklist.expires_at > now,
                    )
                )
            )

            return result is not None

    @staticmethod
    @inject
    def add_to_blacklist(
        provider_name: str,
        model_name: str,
        region: str,
        failure_reason: Optional[str] = None,
        config: AppConfig = injected,
    ) -> None:
        """Add a provider/model/region combination to the blacklist"""
        if not config.LLM_REGION_BLACKLIST_ENABLED:
            return

        with Session() as session:
            now = datetime.datetime.now(datetime.UTC)
            expires_at = now + datetime.timedelta(
                seconds=config.LLM_REGION_BLACKLIST_DURATION_SECONDS
            )

            # Use upsert to handle the case where the entry already exists
            stmt = insert(DbLlmRegionBlacklist).values(
                provider_name=provider_name,
                model_name=model_name,
                region=region,
                blacklisted_at=now,
                expires_at=expires_at,
                failure_reason=failure_reason,
                failure_count=1,
            )

            # If the entry already exists, update it with new expiry and increment failure count
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    DbLlmRegionBlacklist.provider_name,
                    DbLlmRegionBlacklist.model_name,
                    DbLlmRegionBlacklist.region,
                ],
                set_={
                    DbLlmRegionBlacklist.blacklisted_at: now,
                    DbLlmRegionBlacklist.expires_at: expires_at,
                    DbLlmRegionBlacklist.failure_reason: failure_reason,
                    DbLlmRegionBlacklist.failure_count: DbLlmRegionBlacklist.failure_count + 1,
                },
            )

            session.execute(stmt)
            session.commit()

            logger.warning(
                f"Added {provider_name} model '{model_name}' region '{region}' to blacklist "
                f"until {expires_at.isoformat()} due to: {failure_reason or 'unknown error'}"
            )

    @staticmethod
    @inject
    def get_non_blacklisted_regions(
        provider_name: str,
        model_name: str,
        candidate_regions: List[str],
        config: AppConfig = injected,
    ) -> List[str]:
        """Filter a list of regions to remove any that are currently blacklisted"""
        if not config.LLM_REGION_BLACKLIST_ENABLED or not candidate_regions:
            return candidate_regions

        with Session() as session:
            now = datetime.datetime.now(datetime.UTC)

            # Clean up expired entries
            LlmRegionBlacklistService._cleanup_expired_entries(session, now)

            # Get all blacklisted regions for this provider/model
            blacklisted_regions = set(
                session.scalars(
                    select(DbLlmRegionBlacklist.region).where(
                        and_(
                            DbLlmRegionBlacklist.provider_name == provider_name,
                            DbLlmRegionBlacklist.model_name == model_name,
                            DbLlmRegionBlacklist.region.in_(candidate_regions),
                            DbLlmRegionBlacklist.expires_at > now,
                        )
                    )
                ).all()
            )

            # Return only non-blacklisted regions
            filtered_regions = [r for r in candidate_regions if r not in blacklisted_regions]

            if blacklisted_regions:
                logger.info(
                    f"Filtered out blacklisted regions for {provider_name} '{model_name}': "
                    f"{blacklisted_regions}. Available regions: {filtered_regions}"
                )

            return filtered_regions

    @staticmethod
    def _cleanup_expired_entries(session: Session, now: datetime.datetime) -> None:
        """Remove expired blacklist entries"""
        deleted_count = session.execute(
            delete(DbLlmRegionBlacklist).where(DbLlmRegionBlacklist.expires_at <= now)
        ).rowcount

        if deleted_count > 0:
            logger.debug(f"Cleaned up {deleted_count} expired blacklist entries")
            session.commit()

    @staticmethod
    @inject
    def clear_blacklist_for_region(
        provider_name: str,
        model_name: str,
        region: str,
        config: AppConfig = injected,
    ) -> bool:
        """Manually remove a specific region from the blacklist. Returns True if entry was found and removed."""
        if not config.LLM_REGION_BLACKLIST_ENABLED:
            return False

        with Session() as session:
            deleted_count = session.execute(
                delete(DbLlmRegionBlacklist).where(
                    and_(
                        DbLlmRegionBlacklist.provider_name == provider_name,
                        DbLlmRegionBlacklist.model_name == model_name,
                        DbLlmRegionBlacklist.region == region,
                    )
                )
            ).rowcount

            session.commit()

            if deleted_count > 0:
                logger.info(
                    f"Manually cleared blacklist for {provider_name} '{model_name}' region '{region}'"
                )
                return True

            return False

    @staticmethod
    @inject
    def get_blacklist_status(
        provider_name: str,
        model_name: str,
        config: AppConfig = injected,
    ) -> List[dict]:
        """Get current blacklist status for a provider/model combination"""
        if not config.LLM_REGION_BLACKLIST_ENABLED:
            return []

        with Session() as session:
            now = datetime.datetime.now(datetime.UTC)

            # Clean up expired entries first
            LlmRegionBlacklistService._cleanup_expired_entries(session, now)

            # Get active blacklist entries
            entries = session.scalars(
                select(DbLlmRegionBlacklist)
                .where(
                    and_(
                        DbLlmRegionBlacklist.provider_name == provider_name,
                        DbLlmRegionBlacklist.model_name == model_name,
                        DbLlmRegionBlacklist.expires_at > now,
                    )
                )
                .order_by(DbLlmRegionBlacklist.expires_at)
            ).all()

            return [
                {
                    "region": entry.region,
                    "blacklisted_at": entry.blacklisted_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat(),
                    "failure_reason": entry.failure_reason,
                    "failure_count": entry.failure_count,
                    "time_remaining_minutes": int(
                        (
                            ensure_timezone_aware(entry.expires_at)
                            - datetime.datetime.now(datetime.timezone.utc)
                        ).total_seconds()
                        / 60
                    ),
                }
                for entry in entries
            ]
