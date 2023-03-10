"""
Classes that implement different sampling strategies.
"""
from query.query import AggregationQuery

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import random


class BaseSampling(ABC):
    """
    An abstract base class for our candidate sampling logic.
    """

    def __init__(self):
        """
        No base initialization.
        """
        pass

    @abstractmethod
    def sample(self, proxy_pred, oracle_pred):
        """
        The abstract method that executes sampling strategy. The function
        takes in the proxy and oracle prediction(s) (if the oracle was sampled)
        and returns a final prediction for this sample
        """
        pass


class UniformSampling(BaseSampling):
    """
    An implementation that naively samples in uniform fashion throughout time
    """

    def __init__(self, query, sampling_config, agg_config):
        # compute uniformly spaced frames
        step = 1 if "subsample" not in sampling_config else sampling_config['subsample']
        frames = np.arange(query.start_frame, query.end_frame + 1, step)
        self.frames_to_sample = set(np.random.choice(
            frames,
            size=min(query.oracle_limit, len(frames)),
            replace=False,
        ))
        self.agg_config = agg_config

        # lists to contain samples and whether or not they match the predicate
        self.samples = []
        self.samples_match_predicate = []

    def compute_prediction(self, trial_idx):
        """
        Return the predicted answer(s) to the given query.
        """
        # get samples matching predicate
        samples = self.get_samples()

        prediction = None
        if self.agg_config['function'] == "mean":
            prediction = np.nanmean(samples) if len(samples) > 0 else 0

        elif self.agg_config['function'] == "sum":            
            prediction = np.sum(samples) if len(samples) > 0 else 0

        return prediction

    def get_samples(self):
        """
        Return the samples generated by the sampling strategy; filter for
        the samples that actually match the predicate.
        """
        samples = []
        for idx, sample in enumerate(self.samples):
            matches_predicate = self.samples_match_predicate[idx]

            # keep sample if it matches the predicate
            if matches_predicate:
                samples.append(sample)

        return samples
    
    def sample(self, proxy_pred, oracle_pred, oracle_matches_predicate, frame):
        """
        Sample the frame if it is in frames_to_sample
        """
        if frame in self.frames_to_sample:
            self.samples.append(oracle_pred)
            self.samples_match_predicate.append(oracle_matches_predicate)


class StaticSampling(BaseSampling):
    """
    Static stratified sampling baseline.
    """

    def __init__(
        self,
        num_strata: int,
        num_segments: int,
        query: AggregationQuery,
    ):
        """
        Initialize state needed to do basic sampling.
        """
        self.num_strata = num_strata
        self.budget = query.oracle_limit
        self.query = query
        self.num_segments = num_segments

        # set budget per segment
        self.segment_budgets = [
            round((1/(self.num_segments)) * self.budget)
            for segment_idx in range(self.num_segments)
        ]

        # add back any rounded off samples
        idx = 0
        while sum(self.segment_budgets) < self.budget:
            self.segment_budgets[idx] += 1
            idx = (idx + 1) % self.num_segments

        # determine the end frame for each segment
        self.segment_end_frames = []
        for idx in range(self.num_segments):
            end_frame = (
                round(((idx + 1)/self.num_segments) * query.end_frame)
                if idx < self.num_segments - 1
                else query.end_frame + 1
            )
            self.segment_end_frames.append(end_frame)
 
        # initialize counts
        self.counts = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]

        # initialize strata budgets
        self.samples_per_strata = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]

        # evenly distribute samples for all segments
        for segment_idx in range(0, self.num_segments):
            num_strata_samples = int(np.floor(self.segment_budgets[segment_idx]/self.num_strata))
            for strata_idx in range(self.num_strata):
                self.samples_per_strata[segment_idx][strata_idx] = num_strata_samples
        
            # add back any rounded off samples
            idx = 0
            while sum(self.samples_per_strata[segment_idx]) < self.segment_budgets[segment_idx]:
                self.samples_per_strata[segment_idx][idx] += 1
                idx = (idx + 1) % self.num_strata

        # list of samples per strata
        self.samples = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        self.samples_match_predicate = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        self.sample_frames = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]

        # the strata for the current segment
        self.strata = [0.3333, 0.6667, 1.0001]

    def compute_prediction(self, trial_idx):
        """
        Return the predicted answer(s) to the given query.
        """
        samples_by_segment = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        hits_by_segment = [
            [0 for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        sampled_by_segment = [
            [0 for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]

        for segment in range(self.num_segments):
            for strata_idx in range(self.num_strata):
                samples_ = self.samples[segment][strata_idx]
                samples_match_predicate_ = self.samples_match_predicate[segment][strata_idx]
                sampled_by_segment[segment][strata_idx] = len(samples_)

                for sample, matches_predicate in zip(samples_, samples_match_predicate_):
                    if matches_predicate:
                        samples_by_segment[segment][strata_idx].append(sample)
                        hits_by_segment[segment][strata_idx] += 1

        my_means = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]
        my_counts = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]
        for segment in range(self.num_segments):
            for strata_idx in range(self.num_strata):
                mean = np.mean(samples_by_segment[segment][strata_idx]) if len(samples_by_segment[segment][strata_idx]) > 0 else 0
                pos_rate = (
                    hits_by_segment[segment][strata_idx] / sampled_by_segment[segment][strata_idx]
                    if sampled_by_segment[segment][strata_idx] > 0
                    else 0
                )
                count = self.counts[segment][strata_idx] * pos_rate
                my_means[segment][strata_idx] = mean
                my_counts[segment][strata_idx] = count

        total_counts = np.sum(np.array(my_counts))
        prediction = 0
        for segment in range(self.num_segments):
            for strata_idx in range(self.num_strata):
                if total_counts > 0:
                    prediction += my_means[segment][strata_idx] * (my_counts[segment][strata_idx]/total_counts)

        return prediction

    def get_strata(self, proxy_pred: float) -> int:
        """
        Return the index of the strata for the given proxy value.
        """
        for idx, upper_bound in enumerate(self.strata):
            if proxy_pred < upper_bound:
                return idx

        raise Exception(f"didn't find strata idx for {proxy_pred} and {self.strata}")

    def sampling(self, strata_idx, oracle_pred, oracle_matches_predicate, segment, frame):
        """
        Perform sampling within the specified strata for the given frame in the query.
        """
        # get the strata's sample budget
        strata_budget = int(self.samples_per_strata[segment][strata_idx])

        # if reservoir isn't filled yet then add this frame to samples
        if len(self.samples[segment][strata_idx]) < strata_budget:
            self.samples[segment][strata_idx].append(oracle_pred)
            self.samples_match_predicate[segment][strata_idx].append(oracle_matches_predicate)
            self.sample_frames[segment][strata_idx].append(frame)

        # otherwise apply reservoir sampling logic
        else:
            sample_prob = strata_budget / self.counts[segment][strata_idx]
            if random.random() < sample_prob:
                reservoir_idx = random.randint(0, strata_budget - 1)
                self.samples[segment][strata_idx][reservoir_idx] = oracle_pred
                self.samples_match_predicate[segment][strata_idx][reservoir_idx] = oracle_matches_predicate
                self.sample_frames[segment][strata_idx][reservoir_idx] = frame

    def sample(self, proxy_val, oracle_pred, oracle_matches_predicate, frame):
        """
        Update internal logic of the sampling class and return a prediction.
        """
        # get segment depending on frame
        segment = None
        for segment_idx in range(self.num_segments):
            if frame < self.segment_end_frames[segment_idx]:
                segment = segment_idx
                break

        # get strata for sample based on proxy
        strata_idx = self.get_strata(proxy_val)

        # update counts depending on segment
        self.counts[segment][strata_idx] += 1

        # make call to reservoir sampling
        self.sampling(strata_idx, oracle_pred, oracle_matches_predicate, segment, frame)


class InQuestSampling(BaseSampling):
    """
    Sampling according to the InQuest algorithm.
    """

    def __init__(
        self,
        num_strata: int,
        num_segments: int,
        query: AggregationQuery,
        defensive: bool=False,
        defensive_frac: float=0.1,
        pilot_sample_frac: float=0.1,
        pilot_query_frac: float=0.1,
        strata_ewm_alpha: float=0.8,
        alloc_ewm_alpha: float=0.8,
        min_strata_gap: float=None,
        subsample: int=None,
        strata_epsilon: float=1e-6,
        fix_strata: bool=False,
        fix_alloc: bool=False,
    ):
        """
        Initialize state needed to do basic sampling.
        """
        self.num_strata = num_strata
        self.budget = query.oracle_limit
        self.query = query
        self.num_segments = num_segments
        self.defensive = defensive
        self.defensive_frac = defensive_frac
        self.pilot_sample_frac = pilot_sample_frac
        self.pilot_query_frac = pilot_query_frac
        self.strata_ewm_alpha = strata_ewm_alpha
        self.alloc_ewm_alpha = alloc_ewm_alpha
        self.min_strata_gap = min_strata_gap
        self.subsample = subsample
        self.strata_epsilon = strata_epsilon
        self.fix_strata = fix_strata
        self.fix_alloc = fix_alloc

        # determine pilot budget and update budget
        self.pilot_budget = int(np.floor(self.budget * self.pilot_sample_frac))
        self.budget -= self.pilot_budget

        # compute pilot end_frame
        self.pilot_end_frame = query.start_frame + int(np.floor((query.end_frame - query.start_frame) * self.pilot_query_frac))

        # set budget per segment
        self.segment_budgets = [
            self.pilot_budget
            if segment_idx == 0
            else round((1/(self.num_segments - 1)) * self.budget)
            for segment_idx in range(self.num_segments)
        ]

        # add back any rounded off samples
        idx = 0
        while sum(self.segment_budgets) < self.budget:
            self.segment_budgets[idx] += 1
            idx = (idx + 1) % self.num_segments

        # determine the end frame for each segment
        self.segment_end_frames = []
        for idx in range(self.num_segments):
            # if we're pilot sampling
            if idx == 0:
                self.segment_end_frames.append(self.pilot_end_frame)
            else:
                end_frame = (
                    self.pilot_end_frame + round((idx/(self.num_segments - 1)) * (query.end_frame - self.pilot_end_frame))
                    if idx < self.num_segments - 1
                    else query.end_frame + 1
                )
                self.segment_end_frames.append(end_frame)
 

        self.dynamic_computed = [False for _ in range(self.num_segments)]
        self.strata_computed = [False for _ in range(self.num_segments)]

        # initialize counts
        self.counts = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]

        # initialize dynamic strata budgets
        self.samples_per_strata = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]

        # for lesion study, if we fix allocations evenly distribute samples for all segments (pilot will still use uniform)
        if self.fix_alloc:
            for segment_idx in range(1, self.num_segments):
                num_strata_samples = int(np.floor(self.segment_budgets[segment_idx]/self.num_strata))
                for strata_idx in range(self.num_strata):
                    self.samples_per_strata[segment_idx][strata_idx] = num_strata_samples
            
                # add back any rounded off samples
                idx = 0
                while sum(self.samples_per_strata[segment_idx]) < self.segment_budgets[segment_idx]:
                    self.samples_per_strata[segment_idx][idx] += 1
                    idx = (idx + 1) % self.num_strata

        # select frames to sample during uniform sampling
        frames = np.arange(0, self.pilot_end_frame, self.subsample)
        self.pilot_uniform_frames_to_sample = set(np.random.choice(
            frames,
            size=min(self.segment_budgets[0], len(frames)),
            replace=False,
        ))

        # list to store pilot samples
        self.pilot_samples = []

        # list of samples per strata
        self.samples = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        self.samples_match_predicate = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        self.sample_frames = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]

        # keep track of proxy values that evenly divide each segment into strata
        self.segment_to_dividing_proxy_vals = []

        # keep track of raw allocations based on samples
        self.raw_dynamic_allocations = []

        # keep track of mapping from frame to proxy value for all proxy values to compute optimal strata
        self.all_proxy_values = []

        # the strata for the current segment
        self.strata = None

        # for lesion study we may choose to fix the strata
        if self.fix_strata:
            self.strata = [0.3333, 0.6667, 1.0001]

        # keep track of strata computations
        self.computed_strata = []

    def compute_prediction(self, trial_idx):
        """
        Return the predicted answer(s) to the given query.
        """
        samples_by_segment = [
            [[] for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        hits_by_segment = [
            [0 for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]
        sampled_by_segment = [
            [0 for _ in range(self.num_strata)]
            for _ in range(self.num_segments)
        ]

        for segment in range(self.num_segments):
            for strata_idx in range(self.num_strata):
                samples_ = self.samples[segment][strata_idx]
                samples_match_predicate_ = self.samples_match_predicate[segment][strata_idx]
                sampled_by_segment[segment][strata_idx] = len(samples_)

                for sample, matches_predicate in zip(samples_, samples_match_predicate_):
                    if matches_predicate:
                        samples_by_segment[segment][strata_idx].append(sample)
                        hits_by_segment[segment][strata_idx] += 1

        my_means = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]
        my_counts = [[0 for _ in range(self.num_strata)] for _ in range(self.num_segments)]
        for segment in range(self.num_segments):
            for strata_idx in range(self.num_strata):
                mean = np.mean(samples_by_segment[segment][strata_idx]) if len(samples_by_segment[segment][strata_idx]) > 0 else 0
                pos_rate = (
                    hits_by_segment[segment][strata_idx] / sampled_by_segment[segment][strata_idx]
                    if sampled_by_segment[segment][strata_idx] > 0
                    else 0
                )
                count = self.counts[segment][strata_idx] * pos_rate
                my_means[segment][strata_idx] = mean
                my_counts[segment][strata_idx] = count

        total_counts = np.sum(np.array(my_counts))
        prediction = 0
        for segment in range(self.num_segments):
            for strata_idx in range(self.num_strata):
                if total_counts > 0:
                    prediction += my_means[segment][strata_idx] * (my_counts[segment][strata_idx]/total_counts)

        return prediction

    def compute_strata(self, segment):
        # no computation needed for lesion study -- other than populating counts after pilot
        if self.fix_strata:
            self.computed_strata.append(self.strata)

            if segment == 1:
                proxy_df = pd.DataFrame(self.all_proxy_values)
                for oracle_pred, oracle_matches_predicate, frame in self.pilot_samples:
                    proxy_val = proxy_df[proxy_df.frame == frame]['proxy'].iloc[0]
                    strata = self.get_strata(proxy_val)

                    # update relevant state
                    self.samples[0][strata].append(oracle_pred)
                    self.samples_match_predicate[0][strata].append(oracle_matches_predicate)
                    self.sample_frames[0][strata].append(frame)

                frames = np.arange(0, self.pilot_end_frame, self.subsample)
                for frame in frames:
                    proxy_val = proxy_df[proxy_df.frame == frame]['proxy'].iloc[0]
                    strata = self.get_strata(proxy_val)
                    self.counts[0][strata] += 1

            return

        # create dataframe of proxy values so far; each row has (frame, proxy)
        proxy_df = pd.DataFrame(self.all_proxy_values)

        # for previous segment, find evenly dividing proxy values
        # identify subset of proxy values within segment
        prev_segment = segment - 1
        segment_start_frame = self.segment_end_frames[prev_segment - 1] if prev_segment > 0 else 0
        segment_end_frame = self.segment_end_frames[prev_segment]
        segment_proxy_df = proxy_df[
            (segment_start_frame <= proxy_df.frame)
            & (proxy_df.frame < segment_end_frame)
        ]

        # compute strata using .quantile() on proxy values
        strata = [0 for _ in range(self.num_strata)]
        for strata_idx in range(self.num_strata):
            quantile = (strata_idx + 1)/self.num_strata
            strata[strata_idx] = (
                segment_proxy_df.proxy.quantile(quantile)
                if strata_idx < self.num_strata - 1
                else 1.0001
            )

        # add evenly dividing strata to segment_to_dividing_proxy_vals
        self.segment_to_dividing_proxy_vals.append(strata)

        # compute upcoming segment's strata using weighted moving average
        dividing_proxy_df = pd.DataFrame(self.segment_to_dividing_proxy_vals)
        self.strata = dividing_proxy_df.ewm(alpha=self.strata_ewm_alpha).mean().iloc[-1].tolist()

        # add small epsilon to each strata upper bound; we do this because our splits
        # have a non-trivial likelihood of falling on values that the proxy will
        # frequently output; if we set that exact value to be the upper bound then
        # samples with that proxy value will not be included in the strata due to
        # the fact that we use lower_bound <= proxy < upper_bound to determine placement
        for strata_idx in range(self.num_strata):
            self.strata[strata_idx] += self.strata_epsilon

        # manually ensure that there's at least some min gap between strata values
        for strata_idx in range(self.num_strata - 1):
            # note: we use >= because it's possible that three consecutive
            # strata could have the same value, in which case if we bump
            # the second strata to be self.min_strata_gap more than the first,
            # this will be greater than the value for the third strata
            if self.strata[strata_idx] >= self.strata[strata_idx + 1]:
                self.strata[strata_idx + 1] = self.strata[strata_idx] + self.min_strata_gap

        # manually set last strata to have upper limit of 1.0001
        self.strata[self.num_strata - 1] = 1.0001

        # add finalized strata to list of computed strata
        self.computed_strata.append(self.strata)

        # if we just finished the pilot, compute self.counts using strata we just computed for segment == 1
        if segment == 1:
            proxy_df = pd.DataFrame(self.all_proxy_values)
            for oracle_pred, oracle_matches_predicate, frame in self.pilot_samples:
                proxy_val = proxy_df[proxy_df.frame == frame]['proxy'].iloc[0]
                strata = self.get_strata(proxy_val)

                # update relevant state
                self.samples[0][strata].append(oracle_pred)
                self.samples_match_predicate[0][strata].append(oracle_matches_predicate)
                self.sample_frames[0][strata].append(frame)

            frames = np.arange(0, self.pilot_end_frame, self.subsample)
            for frame in frames:
                proxy_val = proxy_df[proxy_df.frame == frame]['proxy'].iloc[0]
                strata = self.get_strata(proxy_val)
                self.counts[0][strata] += 1

    def get_strata(self, proxy_pred: float) -> int:
        """
        Return the index of the strata for the given proxy value.
        """
        for idx, upper_bound in enumerate(self.strata):
            if proxy_pred < upper_bound:
                return idx

        raise Exception(f"didn't find strata idx for {proxy_pred} and {self.strata}")

    def compute_dynamic_sample_allocation(self, segment):
        """
        Compute allocation for dynamic samples across strata.
        """
        samples_by_segment = [
            [[] for _ in range(self.num_strata)]
            for _ in range(segment)
        ]
        hits_by_segment = [
            [0 for _ in range(self.num_strata)]
            for _ in range(segment)
        ]
        sampled_by_segment = [
            [0 for _ in range(self.num_strata)]
            for _ in range(segment)
        ]
        for segment_idx in range(segment):
            for strata_idx in range(self.num_strata):
                samples_ = self.samples[segment_idx][strata_idx]
                samples_match_predicate_ = self.samples_match_predicate[segment_idx][strata_idx]
                sampled_by_segment[segment_idx][strata_idx] = len(samples_)

                for sample, matches_predicate in zip(samples_, samples_match_predicate_):
                    if matches_predicate:
                        samples_by_segment[segment_idx][strata_idx].append(int(sample))
                        hits_by_segment[segment_idx][strata_idx] += 1

        # start computing raw allocation based on samples from previous segment
        prev_segment = segment - 1

        # compute strata pos. rate
        pos_rate_per_strata = [
            hits_by_segment[prev_segment][strata_idx] / sampled_by_segment[prev_segment][strata_idx] if sampled_by_segment[prev_segment][strata_idx] > 0 else 0
            for strata_idx in range(self.num_strata)
        ]

        means, stds = [0 for _ in range(self.num_strata)], [0 for _ in range(self.num_strata)]
        for strata_idx in range(self.num_strata):
            means[strata_idx] = (
                np.mean(samples_by_segment[prev_segment][strata_idx])
                if len(samples_by_segment[prev_segment][strata_idx]) > 0
                else 0
            )

            stds[strata_idx] = (
                np.std(samples_by_segment[prev_segment][strata_idx], ddof=1)
                if len(samples_by_segment[prev_segment][strata_idx]) > 1
                else 0
            )
    
        # compute strata weights and total weight
        strata_weights = [
            np.sqrt(pos_rate_per_strata[strata_idx] * (self.counts[prev_segment][strata_idx]/np.sum(self.counts[prev_segment]))) * stds[strata_idx]
            for strata_idx in range(self.num_strata)
        ]

        total_weight = np.sum(strata_weights)

        # compute allocs
        raw_allocs = (
            [strata_weight / total_weight for strata_weight in strata_weights]
            if total_weight > 0
            else [1 / self.num_strata for _ in strata_weights]
        )
        self.raw_dynamic_allocations.append(raw_allocs)

        # compute upcoming segment's dynamic alloc using weighted moving average
        raw_alloc_df = pd.DataFrame(self.raw_dynamic_allocations)
        allocs = raw_alloc_df.ewm(alpha=self.alloc_ewm_alpha).mean().iloc[-1].tolist()

        # compute defensive set of samples
        strata_defensive_samples = 0
        if self.defensive:
            strata_defensive_samples = np.floor((self.segment_budgets[segment] * self.defensive_frac) / self.num_strata)

        # compute dynamic segment budget accounting for defensive samples
        segment_budget = self.segment_budgets[segment]
        if self.defensive:
            segment_budget = np.floor(self.segment_budgets[segment] * (1 - self.defensive_frac))

        # compute dynamic samples per strata
        self.samples_per_strata[segment] = [
            np.floor(strata_defensive_samples + segment_budget * allocs[strata_idx])
            for strata_idx in range(self.num_strata)
        ]

        # add back floored out samples; use original segment budget for this
        if sum(self.samples_per_strata[segment]) < self.segment_budgets[segment]:
            num_samples = int(self.segment_budgets[segment] - sum(self.samples_per_strata[segment]))
            while num_samples > self.num_strata:
                for strata_idx in range(self.num_strata):
                    self.samples_per_strata[segment][strata_idx] += 1
                num_samples = int(self.segment_budgets[segment] - sum(self.samples_per_strata[segment]))

            indices = np.random.choice(np.arange(self.num_strata), size=num_samples, replace=False)

            for strata_idx in indices:
                self.samples_per_strata[segment][strata_idx] += 1

    def sampling(self, strata_idx, oracle_pred, oracle_matches_predicate, segment, frame):
        """
        Perform sampling within the specified strata for the given frame in the query.
        """
        # if we're in the pilot, perform uniform sampling
        if segment == 0:
            if frame in self.pilot_uniform_frames_to_sample:
                self.pilot_samples.append((oracle_pred, oracle_matches_predicate, frame))

            return

        # compute dynamic sampling allocation if we haven't yet
        if segment > 0 and not self.dynamic_computed[segment] and not self.fix_alloc:  # second condition ensures we do this once
            self.compute_dynamic_sample_allocation(segment)
            self.dynamic_computed[segment] = True

        # get the strata's dynamic sample budget
        strata_budget = int(self.samples_per_strata[segment][strata_idx])

        # if reservoir isn't filled yet then add this frame to samples
        if len(self.samples[segment][strata_idx]) < strata_budget:
            self.samples[segment][strata_idx].append(oracle_pred)
            self.samples_match_predicate[segment][strata_idx].append(oracle_matches_predicate)
            self.sample_frames[segment][strata_idx].append(frame)

        # otherwise apply reservoir sampling logic
        else:
            sample_prob = strata_budget / self.counts[segment][strata_idx]
            if random.random() < sample_prob:
                reservoir_idx = random.randint(0, strata_budget - 1)
                self.samples[segment][strata_idx][reservoir_idx] = oracle_pred
                self.samples_match_predicate[segment][strata_idx][reservoir_idx] = oracle_matches_predicate
                self.sample_frames[segment][strata_idx][reservoir_idx] = frame

    def sample(self, proxy_val, oracle_pred, oracle_matches_predicate, frame):
        """
        Update internal logic of the sampling class and return a prediction.
        """
        # get segment depending on frame
        segment = None
        for segment_idx in range(self.num_segments):
            if frame < self.segment_end_frames[segment_idx]:
                segment = segment_idx
                break

        # compute strata for segment if they haven't been computed already
        if segment > 0 and not self.strata_computed[segment]:
            self.compute_strata(segment)
            self.strata_computed[segment] = True

        # get strata_idx
        strata_idx = None
        if segment > 0:
            # get strata for sample based on proxy
            strata_idx = self.get_strata(proxy_val)

            # update counts depending on segment
            self.counts[segment][strata_idx] += 1

        # make call to reservoir sampling
        self.sampling(strata_idx, oracle_pred, oracle_matches_predicate, segment, frame)

        # add proxy_val to list of all proxy values
        self.all_proxy_values.append({"frame": frame, "proxy": proxy_val, "count": oracle_pred, "matches_pred": oracle_matches_predicate})
