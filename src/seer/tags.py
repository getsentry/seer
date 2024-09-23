from enum import StrEnum


class AnomalyDetectionTags(StrEnum):
    ALERT_ID = "alert_id"
    MODE = "mode"
    LOW_VARIANCE_TS = "low_variance_ts"
    WINDOW_SEARCH_FAILED = "window_search_failed"


class AnomalyDetectionModes(StrEnum):
    STREAMING_ALERT = "streaming.alert"
    STREAMING_TS_WITH_HISTORY = "streaming.ts_with_history"
    BATCH_TS_FULL = "batch.ts_full"
