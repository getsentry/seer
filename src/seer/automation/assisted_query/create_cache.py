import logging

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import LlmClient
from seer.automation.assisted_query.models import CreateCacheRequest, CreateCacheResponse
from seer.automation.assisted_query.prompts import get_cache_prompt
from seer.automation.assisted_query.utils import get_cache_display_name, get_model_provider
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient

logger = logging.getLogger(__name__)

REQUEST_VALUES_LIMIT = 125
REQUIRED_FIELD_PREFIXES = [
    "span.",
    "transaction",
    "user.",
    "http.",
    "request.",
    "db.",
    "project.",
    "trace.",
    "organization.",
    "user",
    "release",
    "ai.",
]


@inject
@observe(name="Create assisted query cache")
@sentry_sdk.trace
def create_cache(
    data: CreateCacheRequest, llm_client: LlmClient = injected, rpc_client: RpcClient = injected
) -> CreateCacheResponse:

    org_id = data.org_id
    project_ids = data.project_ids
    no_values = data.no_values

    cache_diplay_name = get_cache_display_name(org_id, project_ids, no_values)

    cache_name = llm_client.get_cache(display_name=cache_diplay_name, model=get_model_provider())

    if cache_name:
        return CreateCacheResponse(
            success=True, message="Cache already exists", cache_name=cache_name
        )

    fields_response = rpc_client.call(
        "get_attribute_names", org_id=org_id, project_ids=project_ids, stats_period="48h"
    )

    all_fields = []
    if not fields_response:
        logger.warning("No response received from get_attribute_names call")
    elif "fields" not in fields_response:
        logger.warning(
            "Response from get_attribute_names missing 'fields' key. Response: %s",
            fields_response,
        )
    else:
        all_fields = fields_response["fields"]

    filtered_field_values = None
    if not no_values:
        # Filter out numeric tags
        string_fields = [
            field for field in all_fields if not (field.startswith("tags[") and ",number]" in field)
        ]
        filtered_fields = string_fields

        # Include the most important fields for this portion of the query until we reach the limit
        if len(string_fields) > REQUEST_VALUES_LIMIT:
            filtered_fields = []
            for prefix in REQUIRED_FIELD_PREFIXES:
                filtered_fields.extend(
                    [field for field in string_fields if field.startswith(prefix)]
                )
            for field in string_fields:
                if len(filtered_fields) >= REQUEST_VALUES_LIMIT:
                    break
                if field not in filtered_fields:
                    filtered_fields.append(field)

        filtered_field_values_response = rpc_client.call(
            "get_attribute_values",
            fields=filtered_fields,
            org_id=org_id,
            project_ids=project_ids,
            stats_period="48h",
            limit=15,
        )

        filtered_field_values = {}
        if not filtered_field_values_response:
            logger.warning("No response received from get_attribute_values call")
        elif "values" not in filtered_field_values_response:
            logger.warning(
                "Response from get_attribute_values missing 'values' key. Response: %s",
                filtered_field_values_response,
            )
        else:
            filtered_field_values = filtered_field_values_response["values"]

    cache_prompt = get_cache_prompt(
        fields=all_fields, field_values=filtered_field_values, no_values=no_values
    )

    cache_name = llm_client.create_cache(
        display_name=cache_diplay_name, contents=cache_prompt, model=get_model_provider()
    )

    return CreateCacheResponse(
        success=True, message="Cache created successfully", cache_name=cache_name
    )
