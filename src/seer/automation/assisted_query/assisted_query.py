import logging

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import LlmClient
from seer.automation.agent.models import LlmGenerateStructuredResponse
from seer.automation.assisted_query import prompts
from seer.automation.assisted_query.create_cache import create_cache
from seer.automation.assisted_query.models import (
    CreateCacheRequest,
    ModelResponse,
    RelevantFieldsResponse,
    TranslateRequest,
    TranslateResponse,
)
from seer.automation.assisted_query.utils import get_cache_display_name, get_model_provider
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient

logger = logging.getLogger(__name__)


def translate_query(request: TranslateRequest) -> TranslateResponse:

    natural_language_query = request.natural_language_query
    org_id = request.org_id
    project_ids = request.project_ids

    # Cache key will be based off the org id and project ids
    cache_display_name = get_cache_display_name(org_id, project_ids)
    cache_name = LlmClient().get_cache(display_name=cache_display_name, model=get_model_provider())

    if not cache_name:
        # Will result in cold start
        logger.info(f"Cache miss for {cache_display_name}, creating new cache")
        res = create_cache(CreateCacheRequest(org_id=org_id, project_ids=project_ids))
        cache_name = res.cache_name

    sentry_query = create_query_from_natural_language(
        natural_language_query,
        cache_name,
        org_id,
        project_ids,
    )

    sentry_query = sentry_query.parsed

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
    org_id: int,
    project_ids: list[int],
    llm_client: LlmClient = injected,
    rpc_client: RpcClient = injected,
) -> LlmGenerateStructuredResponse:

    model = get_model_provider()

    # Step 1: Figure out relevant fields
    relevant_fields_prompt = prompts.select_relevant_fields_prompt(natural_language_query)
    relevant_fields_response = llm_client.generate_structured(
        prompt=relevant_fields_prompt,
        model=model,
        cache_name=cache_name,
        response_format=RelevantFieldsResponse,
        thinking_budget=0,
        use_local_endpoint=True,
    )

    relevant_fields = (
        relevant_fields_response.parsed.fields if relevant_fields_response.parsed else []
    )

    # Step 2: Fetch values for relevant fields
    field_values_response = rpc_client.call(
        "get_attribute_values",
        org_id=org_id,
        fields=relevant_fields,
        project_ids=project_ids,
        stats_period="48h",
        limit=150,
    )
    field_values = field_values_response.get("field_values", {}) if field_values_response else {}

    # Step 3: Generate final prompt based off of relevant fields and values
    fields_and_values_prompt = prompts.get_fields_and_values_prompt(
        natural_language_query, relevant_fields, field_values
    )
    generated_query = llm_client.generate_structured(
        prompt=fields_and_values_prompt,
        model=model,
        cache_name=cache_name,
        response_format=ModelResponse,
        thinking_budget=0,
        use_local_endpoint=True,
    )

    # XXX: Step 3a/b: Create 3-5 query options and select the best one

    return generated_query
