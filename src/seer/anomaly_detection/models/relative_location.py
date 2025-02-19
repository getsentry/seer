from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict


class PointLocation(Enum):
    UP = 1
    DOWN = 2
    NONE = 3


class ThresholdType(Enum):
    TREND = 1
    PREDICTION = 2
    MP_DIST_IQR = 3
    LOW_VARIANCE_THRESHOLD = 4
    BOX_COX_THRESHOLD = 5


class Threshold(BaseModel):
    type: ThresholdType
    timestamp: float
    upper: float
    lower: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RelativeLocation(BaseModel):
    location: PointLocation
    thresholds: List[Threshold]

    model_config = ConfigDict(arbitrary_types_allowed=True)
