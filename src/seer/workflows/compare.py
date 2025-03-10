from seer.workflows.cohorts_data_processor import DataProcessor
from seer.workflows.cohorts_metrics_scorer import CohortsMetricsScorer
from seer.workflows.models import CompareCohortsRequest, CompareCohortsResponse


def compare_cohorts(data: CompareCohortsRequest) -> CompareCohortsResponse:
    dataset = DataProcessor().prepare_data(data)
    scored_dataset = CohortsMetricsScorer().compute_metrics(dataset, data.options.metric_weights)
    results = [
        {
            "attributeName": rows["attribute_name"],
            "attributeValues": list(rows["distribution_selection"].keys())[
                : data.options.top_k_buckets
            ],
        }
        for _, rows in scored_dataset.head(data.options.top_k_attributes).iterrows()
    ]
    return CompareCohortsResponse(results=results)
