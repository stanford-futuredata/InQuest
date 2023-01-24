from oracle.oracle import Oracle
from proxy.proxy import OracleProxy, PrecomputedProxy, RandomProxy, RandomIntervalProxy
from query.query import AggregationQuery
from sampling.sampling import DynamicSampling, StaticSampling, UniformSampling
from statistics.statistics import at_least_one, count, windowed_fcn
from utils.io import write_json
from utils.metrics import compute_metrics

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

import argparse
import json
import os
import random
import time


def construct_query(query_config, oracle_idx):
    """
    Construct the query class from the specified configuration.
    """
    return AggregationQuery(query_config, oracle_idx)


def construct_statistic_fcn(statistic_config):
    """
    Construct the statistic function from the specified configuration.
    """
    statistic_fcn = None
    if statistic_config['function'] == "at_least_one":
        statistic_fcn = (
            at_least_one
            if "count_col" not in statistic_config
            else partial(at_least_one, count_col=statistic_config['count_col'])
        )

    elif statistic_config['function'] == "count":
        statistic_fcn = (
            count
            if "count_col" not in statistic_config
            else partial(count, count_col=statistic_config['count_col'])
        )

    return statistic_fcn, statistic_config['function']


def construct_aggregation_fcn(aggregation_config):
    """
    Construct the aggregation function from the specified configuration.
    """
    aggregation_fcn = None
    if aggregation_config['function'] == "sum":
        aggregation_fcn = np.sum

    elif aggregation_config['function'] == "mean":
        aggregation_fcn = np.mean

    elif aggregation_config['function'] == "median":
        aggregation_fcn = np.median

    elif aggregation_config['function'] == "window":
        aggregation_fcn = partial(
            windowed_fcn,
            window_agg=aggregation_config['window_agg'],
            window_len=aggregation_config['window_len'],
        )

    return aggregation_fcn


def construct_oracle(oracle_config, oracle_df, query, statistic_fcn):
    """
    Construct the oracle class from the specified configuration.
    """

    return Oracle(oracle_df, query, statistic_fcn)


def construct_proxy(proxy_config, proxy_df, oracle_df, statistic_fcn, statistic_fcn_name):
    """
    Construct the proxy from the specified configuration.
    """
    proxy = None
    if proxy_config['model'] == "random":
        proxy = RandomProxy()

    elif proxy_config['model'] == "random_interval":
        proxy = RandomIntervalProxy(
            proxy_config['proxy_min'],
            proxy_config['proxy_max'],
            proxy_config['proxy_reals'].lower() == "true"
        )

    elif proxy_config['model'] == "precomputed":
        proxy = (
            PrecomputedProxy(
                proxy_df,
                count_col=proxy_config['count_col'],
                statistic_fcn=statistic_fcn_name,
            )
            if "count_col" in proxy_config
            else PrecomputedProxy(
                proxy_df,
                count_cols=proxy_config['columns'],
                statistic_fcn=statistic_fcn_name,
                weighted=False
            )
        )

    elif proxy_config['model'] == "oracle":
        proxy = OracleProxy(oracle_df, statistic_fcn)

    return proxy


def construct_sampling_strategy(sampling_config, query, agg_config):
    """
    Construct the oracle sampler class from the specified configuration.
    """
    sampling_strategy = None
    if sampling_config['strategy'] == "uniform":
        sampling_strategy = UniformSampling(query, sampling_config, agg_config)

    elif sampling_config['strategy'] == "static":
        sampling_strategy = StaticSampling(
            sampling_config['num_strata'],
            sampling_config['num_segments'],
            query,
        )
    
    elif sampling_config['strategy'] == "dynamic":
        sampling_strategy = DynamicSampling(
            sampling_config['num_strata'],
            sampling_config['num_segments'],
            query,
            defensive=(sampling_config['defensive'] == "true"),
            defensive_frac=(
                sampling_config['defensive_frac']
                if 'defensive_frac' in sampling_config
                else 0.1
            ),
            pilot_sample_frac=(
                sampling_config['pilot_sample_frac']
                if 'pilot_sample_frac' in sampling_config
                else 0.1
            ),
            pilot_query_frac=(
                sampling_config['pilot_query_frac']
                if 'pilot_query_frac' in sampling_config
                else 0.1
            ),
            strata_ewm_alpha=(
                sampling_config['strata_ewm_alpha']
                if 'strata_ewm_alpha' in sampling_config
                else 0.8
            ),
            alloc_ewm_alpha=(
                sampling_config['alloc_ewm_alpha']
                if 'alloc_ewm_alpha' in sampling_config
                else 0.8
            ),
            min_strata_gap=(
                sampling_config['min_strata_gap']
                if 'min_strata_gap' in sampling_config
                else None
            ),
            subsample=(
                sampling_config['subsample']
                if 'subsample' in sampling_config
                else 1
            ),
            fix_strata=(
                (sampling_config['fix_strata'] == "true")
                if 'fix_strata' in sampling_config
                else False
            ),
            fix_alloc=(
                (sampling_config['fix_alloc'] == "true")
                if 'fix_alloc' in sampling_config
                else False
            ),
        )

    return sampling_strategy


def run_experiment(config_and_data: Tuple[int, dict, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """
    Simulate processing a query over a dataset given oracle and proxy values.
    """
    # unpack 
    trial_idx, config, oracle_df, proxy_df = config_and_data

    # seed randomness using trial_idx
    np.random.seed(trial_idx)
    random.seed(trial_idx)

    # get oracle_idx from trial_idx
    oracle_idx = trial_idx // config['trials_per_oracle_limit']

    # construct core elements of AQP
    query = construct_query(config['query'], oracle_idx)
    statistic_fcn, statistic_fcn_name = construct_statistic_fcn(config['statistic'])
    aggregation_fcn = construct_aggregation_fcn(config['aggregation'])
    oracle = construct_oracle(config['oracle'], oracle_df, query, statistic_fcn)
    proxy = construct_proxy(config['proxy'], proxy_df, oracle_df, statistic_fcn, statistic_fcn_name)
    sampling_strategy = construct_sampling_strategy(config['sampling'], query, config['aggregation'])

    # iterate over frames and apply sampling/selection logic
    proxy_preds, targets = [], []
    step = 1 if "subsample" not in config['sampling'] else config['sampling']['subsample']
    for frame in range(query.start_frame, query.end_frame + 1, step):
        proxy_val = proxy.predict(frame)
        oracle_pred, oracle_matches_predicate = oracle.predict(frame)
        if oracle_matches_predicate:
            targets.append(oracle_pred)

        sampling_strategy.sample(proxy_val, oracle_pred, oracle_matches_predicate, frame)

    # compute prediction
    prediction = sampling_strategy.compute_prediction(trial_idx)  # (trial_idx)

    # compute metrics given the prediction(s) and target(s); aggregation_fcn is None for selection queries
    results = compute_metrics(prediction, targets, config, aggregation_fcn)

    return results


def simulator(config_filepath, results_dir, num_trials, num_processes, alpha, num_segments):
    """
    Entrypoint for running the simulator.
    """
    # read configuration file; see configs/README.md for a definition of the config schema
    config = None
    with open(config_filepath, 'r') as f:
        config = json.load(f)

    # override alpha and num segments if necessary
    if alpha is not None:
        config['sampling']['strata_ewm_alpha'] = alpha
        config['sampling']['alloc_ewm_alpha'] = alpha

    if num_segments is not None:
        config['sampling']['num_segments'] = num_segments

    # read and filter oracle dataframe as this is an expensive operation
    oracle_df = (
        pd.read_csv(config['oracle']['oracle_csv'], names=config['oracle']['columns'])
        if "columns" in config['oracle'] and config['oracle']['columns']
        else pd.read_csv(config['oracle']['oracle_csv'])
    )
    if config['oracle']['frame_col'] == "false":
        oracle_df['frame'] = np.arange(oracle_df.shape[0])

    start_frame = config['query']['start_frame']
    end_frame = start_frame + config['query']['time_limit']
    query_filter = f"({start_frame} <= frame) & (frame <= {end_frame})"
    oracle_df = oracle_df.query(query_filter)

    # read and filter proxy dataframe as this is an expensive operation
    proxy_df = None
    if config['proxy']['model'] == "precomputed":
        proxy_df = (
            pd.read_csv(config['proxy']['proxy_csv'], names=config['proxy']['columns'])
            if "columns" in config['proxy'] and config['proxy']['columns']
            else pd.read_csv(config['proxy']['proxy_csv'])
        )
        if "upsample" in config['proxy'] and config['proxy']['upsample'] == "ffill":
            proxy_df = proxy_df.loc[proxy_df.index.repeat(2)]
        if config['proxy']['frame_col'] == "false":
            proxy_df['frame'] = np.arange(proxy_df.shape[0])

        query_filter = f"({start_frame} <= frame) & (frame <= {end_frame})"
        proxy_df = proxy_df.query(query_filter)
        if 'proxy' not in proxy_df.columns:
            proxy_df['proxy'] = proxy_df['car_count_1'] + proxy_df['car_count_2'] + proxy_df['car_count_3']

    # if subsampling is specified, subsample dataframes
    if "subsample" in config['sampling']:
        step_size = config['sampling']['subsample']
        oracle_df = oracle_df.iloc[::step_size]
        proxy_df = proxy_df.iloc[::step_size]

    # run experiment(s)
    results = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(
                run_experiment,
                [(trial_idx, config, oracle_df, proxy_df) for trial_idx in range(num_trials)]
            ),
            total=num_trials
        ))

    # construct full dataframe of results
    results_df = pd.DataFrame(results)

    # save results locally
    ts = datetime.now().timestamp()
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(f"{os.path.join(results_dir, f'results_{ts}.csv')}")
    write_json(config, ts, local=True, results_dir=results_dir)


if __name__ == "__main__":
    # parse input argument which should specify a path to a configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath", help="path to configuration file", type=str)
    parser.add_argument("--results-dir", help="local directory for storing experiment results", type=str, default="results/")
    parser.add_argument("--num-trials", help="number of trials to run", type=int, default=8)
    parser.add_argument("--num-processes", help="number of processes to use", type=int, default=8)
    parser.add_argument("--alpha", help="smoothing parameter for EWMA", type=float, default=None)
    parser.add_argument("--segments", help="number of segments to use", type=int, default=None)
    args = parser.parse_args()

    # run simulator
    simulator(args.config_filepath, args.results_dir, args.num_trials, args.num_processes, args.alpha, args.segments)
