from seer.automation.agent.client import GeminiProvider


def get_cache_display_name(org_id: int, project_ids: list[int], no_values: bool = False):
    if no_values:
        return f"{org_id}_{'-'.join(map(str, project_ids))}_no_values"
    return f"{org_id}_{'-'.join(map(str, project_ids))}"


def get_model_provider():
    return GeminiProvider.model("gemini-2.5-flash-preview-05-20", local_regions_only=True)
