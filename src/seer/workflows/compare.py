from seer.workflows.models import CompareCohortsRequest, CompareCohortsResponse


def compare_cohorts(data: CompareCohortsRequest) -> CompareCohortsResponse:
    mock_results = {
        "results": [
            {"attributeName": "browser", "attributeValues": ["unknown", "safari", "edge"]},
            {"attributeName": "country", "attributeValues": ["fr", "uk", "us"]},
        ],
    }
    return CompareCohortsResponse(**mock_results)
