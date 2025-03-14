class WorkflowException(Exception):
    """Base exception for workflow errors"""


class DataProcessingError(WorkflowException):
    """Raised when there's an error processing the data"""


class ScoringError(WorkflowException):
    """Raised when there's an error during scoring"""
