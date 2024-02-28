import functools
import os
from typing import Any, Callable

from seer.grouping.grouping import GroupingLookup
from seer.severity.severity_inference import SeverityInference

root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))


def model_path(subpath: str) -> str:
    return os.path.join(root, "models", subpath)


@functools.cache
def embeddings_model() -> SeverityInference:
    return SeverityInference(
        model_path("issue_severity_v0/embeddings"), model_path("issue_severity_v0/classifier")
    )


@functools.cache
def grouping_lookup() -> GroupingLookup:
    return GroupingLookup(
        model_path=model_path("issue_grouping_v0/embeddings"),
        data_path=model_path("issue_grouping_v0/data.pkl"),
    )


function_env_config = {
    "embeddings_model": "SEVERITY_ENABLED",
    "grouping_lookup": "GROUPING_ENABLED",
}

cached: list[Callable[..., Any]] = [
    globals()[function_name]
    for function_name, env_var in function_env_config.items()
    if os.environ.get(env_var).lower() in ("true", "1", "t")
    if hasattr(globals()[function_name], "cache_info")
]
