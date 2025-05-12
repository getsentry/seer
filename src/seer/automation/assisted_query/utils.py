from seer.automation.agent.client import GeminiProvider


def get_cache_display_name(org_id: int, project_ids: list[int]):
    return f"{org_id}_{'-'.join(map(str, project_ids))}"


def get_model_provider():
    return GeminiProvider.model("gemini-2.5-flash-preview-04-17")
