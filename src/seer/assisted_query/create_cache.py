from seer.assisted_query.models import CreateCacheRequest, CreateCacheResponse
from seer.assisted_query.prompts import get_cache_prompt
from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient


@inject
def create_cache(data: CreateCacheRequest, client: RpcClient = injected) -> CreateCacheResponse:

    organization_slug = data.organization_slug
    project_ids = data.project_ids

    cache_diplay_name = f"{organization_slug}-{'-'.join(map(str, project_ids))}"

    cache_name = LlmClient().get_cache(
        display_name=cache_diplay_name, model=GeminiProvider.model("gemini-2.0-flash-001")
    )

    if cache_name:
        return CreateCacheResponse(
            success=True, message="Cache already exists", cache_name=cache_name
        )

    fields_response = client.call(
        "get_fields",
        organization_slug=organization_slug,
        project_ids=project_ids,
        stats_period="48h",
    )

    fields = fields_response.get("fields", []) if fields_response else []

    field_values_response = client.call(
        "get_field_values",
        organization_slug=organization_slug,
        fields=fields,
        project_ids=project_ids,
        stats_period="48h",
        k=5,
    )

    field_values = field_values_response.get("field_values", {}) if field_values_response else {}

    cache_prompt = get_cache_prompt(fields=fields, field_values=field_values)

    cache_name = LlmClient().create_cache(
        display_name=cache_diplay_name,
        contents=cache_prompt,
        model=GeminiProvider.model("gemini-2.0-flash-001"),
    )

    return CreateCacheResponse(
        success=True, message="Cache created successfully", cache_name=cache_name
    )
