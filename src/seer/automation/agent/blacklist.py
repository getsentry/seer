import datetime
import logging

from sqlalchemy import and_, delete, select
from sqlalchemy.dialects.postgresql import insert

from seer.configuration import AppConfig
from seer.db import DbLlmRegionBlacklist, Session, SQLAlchemySession
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
        failure_reason: str | None = None,
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

            current_failure_count = session.scalar(
                select(DbLlmRegionBlacklist.failure_count).where(
                    and_(
                        DbLlmRegionBlacklist.provider_name == provider_name,
                        DbLlmRegionBlacklist.model_name == model_name,
                        DbLlmRegionBlacklist.region == region,
                    )
                )
            )

            if current_failure_count is not None and current_failure_count > 3:
                logger.error(
                    f"Region {region} for {provider_name} model '{model_name}' has failed for more than 3 times. "
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
        candidate_regions: list[str | None],
        config: AppConfig = injected,
    ) -> list[str | None]:
        """Filter a list of regions to remove any that are currently blacklisted"""
        if not config.LLM_REGION_BLACKLIST_ENABLED or not candidate_regions:
            return candidate_regions

        if None in candidate_regions:
            # If none is in there, regions probably aren't that important
            return candidate_regions

        with Session() as session:
            now = datetime.datetime.now(datetime.UTC)

            string_regions = [r for r in candidate_regions if r is not None]
            if not string_regions:
                return []

            # Get all blacklisted regions for this provider/model
            blacklisted_regions = set(
                session.scalars(
                    select(DbLlmRegionBlacklist.region).where(
                        and_(
                            DbLlmRegionBlacklist.provider_name == provider_name,
                            DbLlmRegionBlacklist.model_name == model_name,
                            DbLlmRegionBlacklist.region.in_(string_regions),
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
    def cleanup_expired_entries(session: SQLAlchemySession, now: datetime.datetime) -> int:
        """Remove expired blacklist entries"""
        deleted_count = session.execute(
            delete(DbLlmRegionBlacklist).where(DbLlmRegionBlacklist.expires_at <= now)
        ).rowcount

        if deleted_count > 0:
            logger.debug(f"Cleaned up {deleted_count} expired blacklist entries")
            session.commit()

        return deleted_count
