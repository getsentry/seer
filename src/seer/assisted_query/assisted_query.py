from seer.assisted_query.models import (
    Chart,
    ModelProvider,
    ModelResponse,
    TranslateRequest,
    TranslateResponse,
)
from seer.automation.agent.client import GeminiProvider, LlmClient


def translate_query(request: TranslateRequest) -> TranslateResponse:

    natural_language_query = request.natural_language_query

    sentry_query = create_query_from_natural_language(natural_language_query)

    return TranslateResponse(
        query=sentry_query.query,
        stats_period=sentry_query.stats_period,
        group_by=sentry_query.group_by,
        visualization=sentry_query.visualization,
        sort=sentry_query.sort,
    )


def create_query_from_natural_language(
    natural_language_query: str, model_provider: ModelProvider = ModelProvider.GEMINI
) -> ModelResponse:

    if model_provider == ModelProvider.GEMINI:
        client = LlmClient(
            model=GeminiProvider.model("gemini-2.0-flash-001"),
            system_prompt="",
        )

        # TODO: Step 0: Check if system prompt (field/values) is in cache
        client.caches.list()

    # TODO: Step 1: Figure out relevant fields

    # TODO: Step 2: Fetch values for relevant fields

    # TODO: Step 2.a: Create 3-5 queries

    # TODO: Step 2.b: Select best query

    # TODO: Step 3: Create and return model response

    return ModelResponse(
        explanation="",
        query="",
        stats_period="",
        group_by="",
        visualization=Chart(chart_type=1, y_axes=[]),
        sort="",
        confidence_score=0.0,
    )
