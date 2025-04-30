import logging

import sentry_sdk
from langfuse.decorators import observe

from seer.assisted_query import prompts
from seer.assisted_query.create_cache import create_cache
from seer.assisted_query.models import (
    CreateCacheRequest,
    ModelResponse,
    RelevantFieldsResponse,
    TranslateRequest,
    TranslateResponse,
)
from seer.assisted_query.utils import get_cache_display_name, get_model_provider
from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient

logger = logging.getLogger(__name__)


def translate_query(request: TranslateRequest) -> TranslateResponse:

    natural_language_query = request.natural_language_query

    org_id = request.organization_id
    project_ids = request.project_ids

    # Cache key will be based off the organization id and project ids
    cache_display_name = get_cache_display_name(org_id, project_ids)

    cache_name = LlmClient().get_cache(cache_display_name, get_model_provider())

    if not cache_name:
        # Will result in cold start
        logger.info("Creating cached prompt as not available upon translation request`.")
        create_cache(CreateCacheRequest(organization_id=org_id, project_ids=project_ids))

    sentry_query = create_query_from_natural_language(
        natural_language_query,
        cache_display_name,
        org_id,
        project_ids,
    )

    return TranslateResponse(
        query=sentry_query.query,
        stats_period=sentry_query.stats_period,
        group_by=sentry_query.group_by,
        visualization=sentry_query.visualization,
        sort=sentry_query.sort,
    )


@inject
@observe(name="Create query from natural language")
@sentry_sdk.trace
def create_query_from_natural_language(
    natural_language_query: str,
    cache_name: str,
    organization_slug: str,
    project_ids: list[int],
    llm_client: LlmClient = injected,
    rpc_client: RpcClient = injected,
) -> ModelResponse:

    # Step 1: Figure out relevant fields
    relevant_fields_prompt = prompts.select_relevant_fields_prompt(natural_language_query)
    relevant_fields_response = llm_client.generate_structured(
        prompt=relevant_fields_prompt,
        model=get_model_provider(),
        cache_name=cache_name,
        response_format=RelevantFieldsResponse,
    )

    relevant_fields = (
        relevant_fields_response.parsed.fields if relevant_fields_response.parsed else []
    )

    # Step 2: Fetch values for relevant fields
    field_values_response = rpc_client.call(
        "get_field_values",
        organization_slug=organization_slug,
        fields=relevant_fields,
        project_ids=project_ids,
        stats_period="48h",
        k=150,
    )
    field_values = field_values_response.get("field_values", {}) if field_values_response else {}

    # Step 3: Generate final prompt based off of relevant fields and values
    fields_and_values_prompt = prompts.get_fields_and_values_prompt(
        natural_language_query, relevant_fields, field_values
    )
    generated_query = llm_client.generate_structured(
        prompt=fields_and_values_prompt,
        model=GeminiProvider.model("gemini-2.0-flash-001"),
        cache_name=cache_name,
        response_format=ModelResponse,
    )

    # XXX: Step 3a/b: Create 3-5 query options and select the best one

    return generated_query
