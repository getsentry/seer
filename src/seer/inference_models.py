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
    if os.environ.get("GROUPING_ENABLED") != "true":
        raise ValueError("Grouping is not enabled")
    return GroupingLookup(
        model_path=model_path("issue_grouping_v0/embeddings"),
        data_path=model_path("issue_grouping_v0/data.pkl"),
    )


cached: list[Callable[..., Any]] = [v for k, v in globals().items() if hasattr(v, "cache_info")]
