import os

SNUBA_URL = os.environ.get("SNUBA", "http://127.0.0.1:1218")
SNUBA_TIMEOUT = 30
SNUBA_REFERER = "timeseries-analysis-event-count"
