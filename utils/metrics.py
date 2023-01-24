import numpy as np

import json


def compute_metrics(prediction, targets, config, aggregation_fcn=None):
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

    # # selection queries
    # if config['query']['type'] == "selection":
    #     # turn into numpy arrays
    #     selections_arr = np.array(predictions)
    #     targets_arr = np.array(targets)

    #     # compute recall and precision
    #     tps = ((selections_arr == 1) & (targets_arr == 1)).sum()
    #     fns = ((selections_arr == 0) & (targets_arr == 1)).sum()
    #     fps = ((selections_arr == 1) & (targets_arr == 0)).sum()

    #     recall = tps / (tps + fns) if tps + fns > 0 else 0 
    #     precision = tps / (fps + tps) if fps + tps > 0 else 0

    #     # TODO: if it's not binary selection: results = {"l2_error": , "l1_error": }

    #     results = {"recall": round(recall, 4), "precision": round(precision, 4)}

    # # aggregation queries
    # else:

    # compute aggregation over predictions and targets
    # prediction = aggregation_fcn(predictions)
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

    elif agg_config.get('final_agg') == "max":
        prediction = np.max(prediction)
        target = np.max(target)
    
    # get the total number of frames in the query
    num_preds = config['query']['time_limit'] + 1

    # # write out samples
    # experiment = "uniform"
    # filename = f"results/samples_{trial_idx}_{experiment}.npy"
    # np.save(filename, np.array(samples))
    # # filename = f"results/sample_indices_{trial_idx}_{experiment}.npy"
    # # np.save(filename, np.array(sample_indices))
    # filename = f"results/counts_{trial_idx}_{experiment}.npy"
    # np.save(filename, counts)
    # filename = f"results/weights_{trial_idx}_{experiment}.npy"
    # np.save(filename, weights)

    # compute L1 and L2 error on aggregation
    results = {
        # "l1_error": round(abs(prediction - target) / num_preds, 4),
        # "l2_error": round((prediction - target)**2 / num_preds, 4),
        "l1_error": abs(prediction - target), # / num_preds,
        "l2_error": (prediction - target)**2, # / num_preds,
        "prediction": prediction,
        "target": target,
    }

    return results
