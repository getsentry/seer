from seer.workflows.cohorts_data_processor import DataProcessor
from seer.workflows.models import CompareCohortsRequest, CompareCohortsResponse


def compare_cohorts(data: CompareCohortsRequest) -> CompareCohortsResponse:
    data_processor = DataProcessor()
    dataset = data_processor.prepare_data(data)
    print(dataset)
    mock_results = {
        "results": [
            {"attributeName": "browser", "attributeValues": ["unknown", "safari", "edge"]},
            {"attributeName": "country", "attributeValues": ["fr", "uk", "us"]},
        ],
    }
    return CompareCohortsResponse(**mock_results)
