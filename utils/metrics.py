import numpy as np


def compute_metrics(prediction, targets, config, oracle_limit, aggregation_fcn):
    """
    This function takes in a prediction or list of predictions and a target
    or list of targets and computes a set of metrics that we use to evaluate
    our algorithm's performance. In the aggregation setting the targets
    represent the per-frame groundtruth values that the proxy should predict.
    We also pass in an aggregation function which takes in the list of
    predictions and targets to compute the predicted and groudtruth aggregation
    values, respectively.
    """
    results = None

    # compute aggregation over predictions and targets
    target = aggregation_fcn(targets)

    # compute final aggregation if we're applying one after windowed aggregation
    agg_config = config['aggregation']
    if agg_config.get('final_agg') == "sum":
        prediction = np.sum(prediction)
        target = np.sum(target)

    elif agg_config.get('final_agg') == "mean":
        prediction = np.mean(prediction)
        target = np.mean(target)

    elif agg_config.get('final_agg') == "median":
        prediction = np.nanmedian(prediction)
        target = np.nanmedian(target)

    # compute L1 and L2 error on aggregation
    results = {
        "l1_error": abs(prediction - target),
        "l2_error": (prediction - target)**2,
        "prediction": prediction,
        "target": target,
        "oracle_limit": oracle_limit,
    }

    return results
