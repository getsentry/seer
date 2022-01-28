import logging
import time
import json
import settings
import pandas as pd
import numpy as np

from datetime import datetime
from flask import Flask, request
from urllib3 import connection_from_url, Retry
from prophet_detector import ProphetDetector, ProphetParams
from snuba_sdk import (
    Column,
    Condition,
    Direction,
    Entity,
    Function,
    Granularity,
    Offset,
    Op,
    OrderBy,
    Query,
    Limit,
)

log = logging.getLogger()
app = Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    timings = {}
    s = time.time()
    if request.method == "GET":
        start, end = args.get("start"), args.get("end")
        query_start, query_end, granularity = _map_snuba_queries(start, end)
        dataset = "transactions"
        query = _build_snuba_query(
            dataset,
            query_start,
            query_end,
            granularity,
            args.get("project"),
            args.get("transaction"),
        )
        data = _submit_snuba_query(query)
        start, end = datetime.fromtimestamp(start), datetime.fromtimestamp(end)
    elif request.method == "POST":
        data = request.get_json()
        start, end = data["start"], data["end"]
        granularity = data["granularity"]

    params = ProphetParams(
        interval_width=0.95,
        changepoint_prior_scale=0.01,
        weekly_seasonality=14,
        daily_seasonality=False,
        uncertainty_samples=None,
    )
    m = ProphetDetector(pd.DataFrame(data["data"]), start, end, params)
    m.pre_process_data()
    timings["pre_process"] = time.time() - s

    s = time.time()
    m.fit()
    timings["train"] = time.time() - s

    s = time.time()
    fcst = m.predict()
    timings["predict"] = time.time() - s

    s = time.time()
    m.add_prophet_uncertainty(fcst)
    timings["uncertainty"] = time.time() - s

    s = time.time()
    fcst = m.scale_scores(fcst)
    # convert datetime to unix seconds
    fcst["ds"] = fcst["ds"].astype(np.int64) * 1e-9
    timings["gen_scores"] = time.time() - s

    s = time.time()
    output = process_output(fcst, granularity)
    timings["format_output"] = time.time() - s

    logging.info(timings)

    return output


def _map_snuba_queries(start, end):
    """
    Takes visualization start/end timestamps
    and returns the start/end/granularity
    of the snuba query that we should execute

    Attributes:
    start: unix timestamp representing start of visualization window
    end: unix timestamp representing end of visualization window

    Returns:
    results: dictionary containing
        query_start: datetime representing start of query window
        query_end: datetime representing end of query window
        granularity: granularity to use (in seconds)
    """

    def days(n):
        return 60 * 60 * 24 * n

    if end - start <= days(2):
        granularity = 300
        query_start = end - days(7)
    elif end - start <= days(7):
        granularity = 600
        query_start = end - days(14)
    elif end - start <= days(14):
        granularity = 1200
        query_start = end - days(28)
    else:
        granularity = 3600
        query_start = end - days(90)
    query_end = end

    return (
        datetime.fromtimestamp(query_start),
        datetime.fromtimestamp(query_end),
        granularity,
    )


def _build_snuba_query(dataset, query_start, query_end, granularity, project_id, transaction):
    """
    dataset: dataset name
    query_start: starting datetime
    query_end: ending datetime
    granularity: data granularity in seconds
    project_id: project_id
    transaction: transaction name
    """
    query = Query(
        dataset=dataset,
        match=Entity("transactions"),
        select=[Function("count", [], "transaction_count")],
        groupby=[Column("time")],
        where=[Condition(Column("finish_ts"), Op.GTE, query_start),
               Condition(Column("finish_ts"), Op.LT, query_end),
               Condition(Column("project_id"), Op.EQ, project_id),
               Condition(Column("transaction"), Op.EQ, transaction)],
        orderby=[OrderBy(Column("time"), Direction.ASC)],
        offset=Offset(0),
        limit=Limit(2200),  # max possible rows based on _map_snuba_queries()
        granularity=Granularity(granularity),
    )

    query.set_parent_api("<unknown>")\
        .set_turbo(False)\
        .set_consistent(False)\
        .set_debug(False)\
        .set_dry_run(False)\
        .set_legacy(False)

    return query


def _submit_snuba_query(query):
    snuba_connection_pool = connection_from_url(
        settings.SNUBA_URL,
        retries=Retry(
            total=5,
            # Our calls to Snuba fail frequently due to network issues. We want to automatically
            # retry most requests. Some of our POSTs and all of our DELETEs do cause mutations,
            # but we have other things in place to handle duplicate mutations.
            method_whitelist={"POST"},
            status_forcelist=[408, 429, 502, 503, 504],
        ),
        timeout=settings.SNUBA_TIMEOUT,
        maxsize=1,  # since this service is not multithreaded, yet
    )

    response = snuba_connection_pool.urlopen(
        "POST", f"/{query.dataset}/snql",
        body=query.snuba(),
        headers={"referer": settings.SNUBA_REFERER})

    return response.json()


def aggregate_anomalies(data, granularity):
    """
    Format data for frontend

    Attributes:
    data: the input dataframe (with anomalies added)
    granularity: data granularity (in seconds)

    Returns:
    results: list of dictionaries containing anomaly information
        start: when anomaly started
        end: when anomaly ended
        confidence: anomaly confidence
        received: actual count for metric
        expected: expected count for metric (from yhat)
        id: "unique" id for each anomaly
    """
    anomalies = []
    last_score = None
    anomaly_index = -1
    sum_expected, sum_actual = 0, 0
    for ds_time, score, y, yhat in data[~data["anomalies"].isna()][
        ["ds", "anomalies", "y", "yhat"]
    ].itertuples(index=False):
        if score == last_score:
            anomalies[anomaly_index]["end"] = ds_time + granularity
            anomalies[anomaly_index]["received"] += y
            anomalies[anomaly_index]["expected"] += yhat
        else:
            sum_expected = yhat
            sum_actua = y
            anomaly_index += 1
            anomalies.append(
                {
                    "start": int(ds_time),
                    "end": int(ds_time + granularity),
                    "confidence": score,
                    "received": y,
                    "expected": yhat,
                    "id": anomaly_index,
                }
            )
    last_score = score

    return anomalies


def process_output(data, granularity):
    """
    Format data for frontend

    Attributes:
    data: the input dataframe (with anomalies added)
    granularity: data granularity (seconds)

    Returns:
    results: dictionary containing
        y: input timeseries
        yhat_upper: upper confidence bound
        yhat_lower: lower confidence bound
        anomalies: all detected anomalies
    """

    def convert_ts(ts, value_col):
        """
        Format a timeseries for the frontend
        """
        data = zip(
            list(map(int, ts["ds"])), [[{"count": x}] for x in list(ts[value_col])]
        )
        start = int(ts["ds"].iloc[0])
        end = int(ts["ds"].iloc[-1])
        return {"data": list(data), "start": start, "end": end}

    results = {
        "y": convert_ts(data, "y"),
        "yhat_upper": convert_ts(data, "yhat_upper"),
        "yhat_lower": convert_ts(data, "yhat_lower"),
        "anomalies": aggregate_anomalies(data, granularity),
    }
    return results
