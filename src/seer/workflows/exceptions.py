class WorkflowException(Exception):
    """Base exception for workflow errors"""

    pass


class DataProcessingError(WorkflowException):
    """Raised when there's an error processing the data"""

    pass


class ScoringError(WorkflowException):
    """Raised when there's an error during scoring"""

    pass
