from pydantic import BaseModel


class CleanupConfig(BaseModel):
    num_old_points: int
    timestamp_threshold: float
    num_acceptable_points: int
