import numpy as np

import json


def compute_metrics(prediction, segment_predictions, targets, target_frames, segment_end_frames, oracle_limit, config, aggregation_fcn=None):
    """
    This function takes in a prediction or list of predictions and a target
    or list of targets and computes a set of metrics that we use to evaluate
    our algorithm's performance. In the selection and aggregation settings
    the targets represent the per-frame groundtruth values that the proxy
    should predict. For aggregation queries we also pass in an aggregation
    function which takes in the list of predictions and targets to compute
    the predicted and groudtruth aggregation values, respectively.
    """
    results = None
    num_segments = config["sampling"]["num_segments"]

    # compute aggregation over all targets
    target = aggregation_fcn(targets)

    # compute aggregations over targets within each segment
    segment_targets = []
    target_tuples = list(zip(targets, target_frames))
    for segment in range(num_segments):
        segment_start_frame = segment_end_frames[segment - 1] if segment > 0 else 0
        segment_end_frame = segment_end_frames[segment]
        segment_target_tuples = list(filter(
            lambda tup: (segment_start_frame <= tup[1]) and (tup[1] < segment_end_frame),
            target_tuples,
        ))
        segment_target = aggregation_fcn(list(map(lambda tup: tup[0], segment_target_tuples)))
        segment_targets.append(segment_target)

    # compute L1 and L2 error on aggregation
    results = {
        "l1_error": abs(prediction - target),
        "l2_error": (prediction - target)**2,
        "prediction": prediction,
        "target": target,
        "oracle_limit": oracle_limit,
    }

    for segment in range(num_segments):
        segment_prediction = segment_predictions[segment]
        segment_target = segment_targets[segment]
        results[f"prediction_segment_{segment}"] = segment_prediction
        results[f"target_segment_{segment}"] = segment_target
        results[f"l1_error_segment_{segment}"] = abs(segment_prediction - segment_target)
        results[f"l2_error_segment_{segment}"] = (segment_prediction - segment_target)**2

    return results
