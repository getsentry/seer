from pydantic import BaseModel


class CleanupPredictConfig(BaseModel):
    num_old_points: int
    timestamp_threshold: float
    num_acceptable_points: int
    num_predictions_remaining: int
    num_acceptable_predictions: int
