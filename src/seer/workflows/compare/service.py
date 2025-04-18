import logging

from seer.events import SeerEventNames, log_seer_event
from seer.workflows.compare.models import CompareCohortsRequest, CompareCohortsResponse
from seer.workflows.compare.processor import DataProcessor
from seer.workflows.compare.scorer import CohortsMetricsScorer

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        processor: DataProcessor | None = None,
        scorer: CohortsMetricsScorer | None = None,
    ):
        self.processor = processor or DataProcessor()
        self.scorer = scorer or CohortsMetricsScorer()

    def compare_cohort(self, request: CompareCohortsRequest) -> CompareCohortsResponse:
        """
        Compare two cohorts and identify the most suspcious attribute differences.

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

        dataset = self.processor.prepare_cohort_data(request)
        scoredDataset = self.scorer.compute_metrics(dataset, request.config)
        results = [
            {
                "attributeName": row["attribute_name"],
                "attributeValues": list(row["distribution_selection"].keys())[
                    : request.config.topKBuckets
                ],
                "attributeScore": row["rrf_score"],
            }
            for _, row in scoredDataset.head(request.config.topKAttributes).iterrows()
        ]
        return CompareCohortsResponse(results=results)


def compare_cohort(request: CompareCohortsRequest) -> CompareCohortsResponse:
    """
    Function used by the API to compare cohorts.

    Args:
        request: CompareCohortsRequest containing cohort data and comparison options

    Returns:
        CompareCohortsResponse containing ranked attributes and their distinctive values

    Note:
        This is a simplified entry point that creates a new service instance for each call, since the service is cheap to create.
        In the future, if the service becomes more complex, we can consider implementing a singleton pattern.
    """
    log_seer_event(
        SeerEventNames.COMPARATIVE_WORKFLOWS_STARTED,
        {
            "config": request.config.model_dump(mode="json"),
            "meta": request.meta.model_dump(mode="json"),
        },
    )
    return CompareService().compare_cohort(request)
