import numpy as np

from seer.anomaly_detection.models.dynamic_alert import DynamicAlert
from seer.anomaly_detection.models.external import StoreDataRequest, TimeSeriesPoint
from seer.anomaly_detection.models.internal import TimeSeries


def convert_external_ts_to_internal(external_ts: list[TimeSeriesPoint]) -> TimeSeries:
    values = []
    timestamps = []
    for point in external_ts:
        values.append(np.float64(point.value))
        timestamps.append(np.float64(point.timestamp))

    return TimeSeries(values=np.array(values), timestamps=np.array(timestamps))


def store_request_to_dynamic_alert(external_store_request: StoreDataRequest) -> DynamicAlert:
    return DynamicAlert(
        organization_id=external_store_request.organization_id,
        project_id=external_store_request.project_id,
        external_alert_id=external_store_request.alert.id,
        config=external_store_request.config.model_dump(),
    )
