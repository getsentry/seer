"""
Trend Detection Logic:

- Get the most recent change point (as long as its 5 datapoints from the left and 1 from the right)
    - otherwise if the change points are on the edges use midpoint analysis
- Calculate the trend percentage before/after the breakpoint
- Use Welch’s T-test to determine if there is a significant difference in means
    - Welch’s t-test: finds significant in means between two sample groups with unequal variances and different sample sizes
- If p-value > 0.01 and trend percentage > 5%, then the trend is surfaced

"""

import pandas as pd
import numpy as np
import scipy
import datetime

from seer.trend_detection.detectors.cusum_detection import CUSUMDetector


def find_trends(txns_data, sort_function, zerofilled, pval=0.01, trend_perc=0.05):
    trend_percentage_list = []

    # defined outside of for loop so error won't throw for empty data
    transaction_list = txns_data.keys()

    for txn in transaction_list:

        keys = list(txns_data[txn].keys())

        old_format = 'count()' in keys

        # data without zero-filling
        timestamps = []
        metrics = []
        counts = []

        if old_format:
            count_data = txns_data[txn]['count()']['data']

            if keys[0] == 'count()':
                ts_data = txns_data[txn][keys[1]]['data']
            else:
                ts_data = txns_data[txn][keys[0]]['data']

            timestamps_zero_filled = [x[0] for x in ts_data]
            start = txns_data[txn][keys[0]]['start']
            end = txns_data[txn][keys[0]]['end']

            # create lists for time/metric lists without 0 values for more accurate breakpoint analysis
            for i in range(len(ts_data)):
                count = count_data[i][1][0]['count']

                if count != 0:
                    counts.append(count)
                    timestamps.append(ts_data[i][0])
                    metrics.append(ts_data[i][1][0]['count'])

        else:

            if zerofilled:
                ts_data = txns_data[txn]['data']

                for i in range(len(ts_data)):
                    metric = ts_data[i][1][0]['count']

                    if metric != 0:
                        timestamps.append(ts_data[i][0])
                        metrics.append(metric)
            else:

                timestamps = [ts_data[x][0] for x in range(len(ts_data))]
                metrics = [ts_data[x][1][0]['count'] for x in range(len(ts_data))]

            start = txns_data[txn]['start']
            end = txns_data[txn]['end']
            timestamps_pandas = [pd.Timestamp(datetime.datetime.fromtimestamp(x)) for x in timestamps]

        # snuba query limit was hit and we won't have complete data for this transaction so disregard this txn
        if None in metrics:
            continue

        timeseries = pd.DataFrame(
            {
                'time': timestamps_pandas,
                'y': metrics
            }
        )

        # don't include transaction if there are less than three datapoints
        if len(metrics) < 3:
            continue

        #NEW CODE (Once we get timeseries data for 7 days prior)
        # change_points = CUSUMDetector(timeseries).detector(interest_window = [168, len(ts_data)-1], magnitude_quantile=1.0)

        #adding in this parameter will make sure the algorithm will only look for change-points in one direction
        #this will cut the time of cusum detection in half
        if sort_function == 'trend_percentage()':
            change_points = CUSUMDetector(timeseries).detector(change_directions=["decrease"])
        elif sort_function == '-trend_percentage()':
            change_points = CUSUMDetector(timeseries).detector(change_directions=["increase"])
        else:
            CUSUMDetector(timeseries).detector()

        # sort change points by start time to get most recent one
        change_points.sort(key=lambda x: x.start_time)
        num_breakpoints = len(change_points)

        # if breakpoints are detected, get most recent changepoint
        if num_breakpoints != 0:
            change_point = change_points[-1].start_time
            change_index = timestamps.index(change_point)

        # if breakpoint is in the very beginning or no breakpoints are detected, use midpoint analysis instead
        elif num_breakpoints == 0 or change_index <= 10 or change_index == len(timestamps) - 5:
            change_point = (start + end) // 2

        first_half = [metrics[i] for i in range(len(metrics)) if timestamps[i] < change_point]
        second_half = [metrics[i] for i in range(len(metrics)) if timestamps[i] >= change_point]

        # if either of the halves don't have any data to compare to then move on to the next txn
        if len(first_half) == 0 or len(second_half) == 0:
            continue

        mu0 = np.average(first_half)
        mu1 = np.average(second_half)

        # calculate t-value between both groups
        scipy_t_test = scipy.stats.ttest_ind(first_half, second_half, equal_var=False)

        if mu0 == 0:
            trend_percentage = mu1
        else:
            trend_percentage = mu1 / mu0

        txn_names = txn.split(",")
        output_dict = {
            "project": txn_names[0],
            "transaction": txn_names[1],
            "aggregate_range_1": mu0,
            "aggregate_range_2": mu1,
            "unweighted_t_value": scipy_t_test.statistic,
            "unweighted_p_value": round(scipy_t_test.pvalue, 10),
            "trend_percentage": trend_percentage,
            "trend_difference": mu1 - mu0,
            "breakpoint": change_point
        }

        if old_format:
            # get the non-zero counts for the first and second halves
            counts_first_half = [counts[i] for i in range(len(counts)) if timestamps[i] < change_point]
            counts_second_half = [counts[i] for i in range(len(counts)) if timestamps[i] >= change_point]

            # sum of counts before/after changepoint
            count_range_1 = sum(counts_first_half)
            count_range_2 = sum(counts_second_half)

            output_dict['count_range_1'] = count_range_1
            output_dict['count_range_2'] = count_range_2
            output_dict['count_percentage'] = count_range_2 / count_range_1

        # TREND LOGIC:
        #  1. p-value of t-test is less than passed in threshold (default = 0.01)
        #  2. trend percentage is greater than passed in threshold (default = 5%)

        # most improved - get only negatively significant trending txns
        if (
                sort_function == 'trend_percentage()' or sort_function == 'no_sort') and mu1 <= mu0 and scipy_t_test.pvalue < pval and abs(
                trend_percentage - 1) > trend_perc:
            output_dict['change'] = 'improvement'
            trend_percentage_list.append((trend_percentage, output_dict))

        # if most regressed - get only positively signiificant txns
        elif (
                sort_function == '-trend_percentage()' or sort_function == 'no_sort') and mu0 <= mu1 and scipy_t_test.pvalue < pval and abs(
                trend_percentage - 1) > trend_perc:
            output_dict['change'] = 'regression'
            trend_percentage_list.append((trend_percentage, output_dict))


    return trend_percentage_list