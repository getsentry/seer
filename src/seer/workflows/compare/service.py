from seer.workflows.compare.models import (
    CompareCohortsRequest,
    CompareCohortsResponse,
    MetricWeights,
    Options,
)
from seer.workflows.compare.processor import DataProcessor
from seer.workflows.compare.scorer import CohortsMetricsScorer


class CompareService:
    """
    Service class for comparing cohorts using attribute distributions.

    This service orchestrates the process of comparing two cohorts by:
    1. Processing and normalizing their attribute distributions
    2. Computing comparison metrics (KL divergence and entropy)
    3. Ranking attributes by their differences

    The service uses dependency injection for its processor and scorer components,
    allowing for flexible configuration and easier testing.
    """

    _instance = None  # Class-level instance for singleton pattern

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance of CompareService"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Private constructor - use get_instance() instead."""
        if self._instance is not None:
            raise RuntimeError("Use get_instance() to access CompareService")
        self.processor = DataProcessor()
        self.scorer = CohortsMetricsScorer()

    def compare_cohorts(self, request: CompareCohortsRequest) -> CompareCohortsResponse:
        """
        Compare two cohorts and identify the most interesting attribute differences.

        Args:
            request: CompareCohortsRequest containing:
                - baseline cohort data
                - selection cohort data
                - options for comparison (weights, limits)

        Returns:
            CompareCohortsResponse containing:
                - list of top attributes and their most distinctive values,
                  ranked by their RRF scores

        Process:
            1. Ensures options are set (uses defaults if none provided)
            2. Processes cohort data to normalize distributions
            3. Computes comparison metrics and ranks attributes
            4. Extracts top K attributes and their most distinctive values
        """
        if request.options is None:
            request.options = Options()

        dataset = self.processor.prepare_cohorts_data(request)
        scored_dataset = self.scorer.compute_metrics(
            dataset, request.options.metric_weights or MetricWeights()
        )

        results = [
            {
                "attributeName": row["attribute_name"],
                "attributeValues": list(row["distribution_selection"].keys())[
                    : request.options.top_k_buckets
                ],
                "attributeScore": row["RRF_score"],
            }
            for _, row in scored_dataset.head(request.options.top_k_attributes).iterrows()
        ]
        return CompareCohortsResponse(results=results)


def compare_cohorts(data):
    """
    Function used by the API to compare cohorts.

    Args:
        data: CompareCohortsRequest containing cohort data and comparison options

    Returns:
        CompareCohortsResponse containing ranked attributes and their distinctive values

    Note:
        This is a simplified entry point that creates a new service instance for each call.
        The service is implemented as a singleton, so the same instance will be reused across calls.
    """
    return CompareService.get_instance().compare_cohorts(data)
