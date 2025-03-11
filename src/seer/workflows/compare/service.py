from seer.workflows.compare.models import (
    CompareCohortsRequest,
    CompareCohortsResponse,
    MetricWeights,
    Options,
)
from seer.workflows.compare.processor import DataProcessor
from seer.workflows.compare.scorer import CohortsMetricsScorer


class CompareService:
    def __init__(self, processor: DataProcessor = None, scorer: CohortsMetricsScorer = None):
        self.processor = processor or DataProcessor()
        self.scorer = scorer or CohortsMetricsScorer()

    def compare_cohorts(self, request: CompareCohortsRequest) -> CompareCohortsResponse:
        if request.options is None:
            request.options = Options()

        dataset = self.processor.prepare_data(request)
        scored_dataset = self.scorer.compute_metrics(
            dataset, request.options.metric_weights or MetricWeights()
        )

        results = [
            {
                "attributeName": rows["attribute_name"],
                "attributeValues": list(rows["distribution_selection"].keys())[
                    : request.options.top_k_buckets
                ],
            }
            for _, rows in scored_dataset.head(request.options.top_k_attributes).iterrows()
        ]
        return CompareCohortsResponse(results=results)


def compare_cohorts(data):
    return CompareService().compare_cohorts(data)
