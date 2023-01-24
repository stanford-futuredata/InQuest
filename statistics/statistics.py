"""
Define statistic functions used by oracle and proxy.
"""
from typing import List

import pandas as pd


def at_least_one(frame_df: pd.DataFrame, count_col: str=None) -> int:
    """
    This function returns 1 if the frame is non-empty (i.e. there's at least
    one prediction in the frame) and 0 otherwise. In the case where count_col
    is provided, this means that there's a single row per-frame with a column
    specifying the count of the object being queried for. As a result, we
    simply return whether the value in that column for the first (and only)
    element of the dataframe is >= 1.
    """
    return (
        int(not frame_df.empty)
        if count_col is None
        else int(frame_df[count_col].iloc[0] >= 1)
    )


def count(frame_df: pd.DataFrame, count_col: str=None) -> int:
    """
    This function returns the count of frames that satisfy the predicate.
    In the case where count_col is provided, this means that there's a single
    row per-frame with a column specifying the count of the object being
    queried for. As a result, we simply return the value in that column
    for the first (and only) element of the dataframe.
    """
    return (
        frame_df.shape[0]
        if count_col is None
        else frame_df[count_col].iloc[0]
    )


def windowed_fcn(predictions: List[float], window_agg: str, window_len: int) -> float:
    """
    This function applies a window-based aggregation with the specified
    window length to the input predictions. 
    """
    # convert to pandas.Series and create rolling object
    rolling_predictions = pd.Series(predictions).rolling(window=window_len)
    if window_agg == "sum":
        rolling_predictions = rolling_predictions.sum()

    elif window_agg == "mean" or window_agg == "avg":
        rolling_predictions = rolling_predictions.mean()

    elif window_agg == "median":
        rolling_predictions = rolling_predictions.median()

    elif window_agg == "max":
        rolling_predictions = rolling_predictions.max()
    
    return rolling_predictions
