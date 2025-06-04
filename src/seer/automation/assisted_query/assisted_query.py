import logging

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import LlmClient
from seer.automation.agent.models import LlmGenerateStructuredResponse
from seer.automation.assisted_query import prompts
from seer.automation.assisted_query.create_cache import create_cache
from seer.automation.assisted_query.models import (  # QueryOrFieldsResponse,; RelevantFieldsResponse,
    CreateCacheRequest,
    ModelResponse,
    TestResponse,
    TranslateRequest,
    TranslateResponse,
    TranslateResponses,
)

# from seer.automation.assisted_query.tools import SearchTools
from seer.automation.assisted_query.utils import get_cache_display_name, get_model_provider
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["span.op", "span.description", "transaction"]


def translate_query(
    request: TranslateRequest,
) -> TranslateResponses:
    natural_language_query = request.natural_language_query
    org_id = request.org_id
    project_ids = request.project_ids

    # Cache key will be based off the org id and project ids
    cache_display_name = get_cache_display_name(org_id, project_ids)
    cache_name = LlmClient().get_cache(display_name=cache_display_name, model=get_model_provider())

    if not cache_name:
        sentry_sdk.set_tag("cache-miss-name", cache_display_name)
        res = create_cache(
            CreateCacheRequest(org_id=org_id, project_ids=project_ids, no_values=True)
        )
        cache_name = res.cache_name

    sentry_queries = create_query_from_natural_language(
        natural_language_query, cache_name, org_id, project_ids
    )

    queries = sentry_queries.parsed

    responses = [
        TranslateResponse(
            query=query.query,
            stats_period=query.stats_period,
            group_by=query.group_by,
            visualization=query.visualization,
            sort=query.sort,
        )
        for query in queries
    ]

    return TranslateResponses(responses=responses)


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
    # tools = SearchTools(org_id, project_ids)
    model = get_model_provider()

    # Step 1: Try to generate query directly OR request specific fields
    # query_or_fields_prompt = prompts.get_query_or_fields_prompt(natural_language_query)
    # initial_response = llm_client.generate_structured(
    #     prompt=query_or_fields_prompt,
    #     model=model,
    #     cache_name=cache_name,
    #     response_format=QueryOrFieldsResponse,
    #     temperature=0.2,
    #     thinking_budget=0,
    #     use_local_endpoint=True,
    #     # tools=SearchTools(org_id, project_ids).get_tools(),
    # )

    one_shot_prompt = prompts.get_one_shot_prompt(natural_language_query)

    one_shot_response = llm_client.generate_structured(
        prompt=one_shot_prompt,
        model=model,
        cache_name=cache_name,
        response_format=TestResponse,
        temperature=0.2,
        thinking_budget=0,
        use_local_endpoint=True,
    )

    # print("--------------------------------")
    # print("--------------initial_response------------------")
    # print(initial_response)
    # print()
    # print(initial_response.parsed)
    # print("--------------------------------")
    # print("--------------------------------")

    if one_shot_response.parsed and one_shot_response.parsed.queries:
        return LlmGenerateStructuredResponse(
            one_shot_response.parsed.queries, metadata=one_shot_response.metadata
        )

    fields_call_prompt = prompts.get_fields_call_prompt(natural_language_query)

    tool_call_response = llm_client.generate_text(
        prompt=fields_call_prompt,
        model=model,
        cache_name=cache_name,
    )

    # Check for tool calls

    # TODO: Turn the general field search into a tool call? ----------------------------------------------------------------
    # requested_fields = []
    # tool_calls = []
    if tool_call_response.parsed:
        # tool_calls = tool_call_response.tool_calls  # TODO: fix
        pass
    else:
        # Fallback: use the original field selection logic
        logger.info(
            "No direct queries or requested fields for query '%s', falling back to original field selection",
            natural_language_query,
        )
        relevant_fields_prompt = prompts.select_relevant_fields_prompt(natural_language_query)
        relevant_fields_response = llm_client.generate_structured(
            prompt=relevant_fields_prompt,
            model=model,
            cache_name=cache_name,
            response_format=TestResponse,
            thinking_budget=0,
            use_local_endpoint=True,
        )
        requested_fields = (
            relevant_fields_response.parsed.requested_fields
            if relevant_fields_response.parsed
            else []
        )

    # tool_results = []
    # available_tools = {tool.name: tool for tool in SearchTools(org_id, project_ids).get_tools()}

    # for tool_call in tool_calls:
    #     if tool_call.function in available_tools:
    #         tool = available_tools[tool_call.function]
    #         try:
    #             # Parse arguments and call tools
    #             kwargs = json.loads(tool_call.args)
    #             result = tool.call(**kwargs)

    #             tool_results.append(result)
    #             logger.info(f"Executed tool {tool_call.function} with result: {result}")
    #         except Exception as e:
    #             logger.error(f"Error executing tool {tool_call.function}: {e}")
    #     else:
    #         logger.warning(f"Tool {tool_call.function} not found in available tools")

    for field in REQUIRED_FIELDS:
        if field not in requested_fields:
            requested_fields.append(field)

    # -----------------------------------------------------------------------------------------------------

    # Step 2: Fetch values for requested fields
    field_values_response = rpc_client.call(
        "get_attribute_values",
        org_id=org_id,
        fields=requested_fields,
        project_ids=project_ids,
        stats_period="48h",
        limit=200,
    )

    field_values = {}
    if not field_values_response:
        logger.warning("No response received from get_attribute_values call")
    elif "values" not in field_values_response:
        logger.warning(
            "Response from get_attribute_values missing 'values' key. Response: %s",
            field_values_response,
        )
    else:
        field_values = field_values_response["values"]

    # Step 3: Generate final prompt(s) based off of relevant fields and values
    final_query_prompt = prompts.get_final_query_prompt(
        natural_language_query, requested_fields, field_values
    )
    generated_query = llm_client.generate_structured(
        prompt=final_query_prompt,
        model=model,
        cache_name=cache_name,
        response_format=list[ModelResponse],
        temperature=0.2,
        thinking_budget=0,
        use_local_endpoint=True,
    )
    return generated_query
