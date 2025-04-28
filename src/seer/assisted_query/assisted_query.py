import sentry_sdk
from langfuse.decorators import observe

from seer.assisted_query import prompts
from seer.assisted_query.models import (
    ModelResponse,
    RelevantFieldsResponse,
    TranslateRequest,
    TranslateResponse,
)
from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient


def translate_query(request: TranslateRequest) -> TranslateResponse:

    natural_language_query = request.natural_language_query

    # Cache key will be based off the organization slug and project ids
    cache_display_name = f"{request.organization_slug}-{'-'.join(map(str, request.project_ids))}"

    cache_name = LlmClient().get_cache(
        cache_display_name, GeminiProvider.model("gemini-2.0-flash-001")
    )

    if not cache_name:
        # XXX: If cache is not found, should we just cold start the query and create a new cache?
        raise ValueError("Cache not found")

    sentry_query = create_query_from_natural_language(
        natural_language_query,
        cache_display_name,
        request.organization_slug,
        request.project_ids,
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
        model=GeminiProvider.model("gemini-2.0-flash-001"),
        cache_name=cache_name,
        response_format=RelevantFieldsResponse,
    )

    relevant_fields = relevant_fields_response.parsed.fields

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
