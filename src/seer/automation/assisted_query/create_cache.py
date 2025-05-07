from seer.automation.agent.client import LlmClient
from seer.automation.assisted_query.models import CreateCacheRequest, CreateCacheResponse
from seer.automation.assisted_query.prompts import get_cache_prompt
from seer.automation.assisted_query.utils import get_cache_display_name, get_model_provider
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient


@inject
def create_cache(data: CreateCacheRequest, client: RpcClient = injected) -> CreateCacheResponse:

    org_id = data.org_id
    project_ids = data.project_ids

    cache_diplay_name = get_cache_display_name(org_id, project_ids)

    cache_name = LlmClient().get_cache(display_name=cache_diplay_name, model=get_model_provider())

    if cache_name:
        return CreateCacheResponse(
            success=True, message="Cache already exists", cache_name=cache_name
        )

    fields_response = client.call(
        "get_attribute_names", org_id=org_id, project_ids=project_ids, stats_period="48h"
    )

    fields = fields_response.get("fields", []) if fields_response else []

    field_values_response = client.call(
        "get_attribute_values",
        fields=fields,
        org_id=org_id,
        project_ids=project_ids,
        stats_period="48h",
        limit=5,
    )

    field_values = field_values_response.get("field_values", {}) if field_values_response else {}

    cache_prompt = get_cache_prompt(fields=fields, field_values=field_values)

    cache_name = LlmClient().create_cache(
        display_name=cache_diplay_name, contents=cache_prompt, model=get_model_provider()
    )

    return CreateCacheResponse(
        success=True, message="Cache created successfully", cache_name=cache_name
    )
