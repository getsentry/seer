from seer.workflows.cohorts_data_processor import DataProcessor
from seer.workflows.cohorts_metrics_scorer import CohortsMetricsScorer
from seer.workflows.models import CompareCohortsRequest, CompareCohortsResponse


def compare_cohorts(data: CompareCohortsRequest) -> CompareCohortsResponse:
    dataset = DataProcessor().prepare_data(data)
    scored_dataset = CohortsMetricsScorer().compute_metrics(dataset, data.options.metric_weights)
    print(scored_dataset)
    mock_results = {
        "results": [
            {"attributeName": "browser", "attributeValues": ["unknown", "safari", "edge"]},
            {"attributeName": "country", "attributeValues": ["fr", "uk", "us"]},
        ],
    }
    return CompareCohortsResponse(**mock_results)
