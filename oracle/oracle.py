from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseOracle(ABC):
    """
    An abstract base class for our Oracle logic.
    """
    def __init__(self, oracle_df, query, statistic_fcn):
        """
        The class takes in the csv file with oracle predictions as well as the
        query we're executing and the statistic we're measuring. The query provides
        us with the oracle limit and the predicate of interest, while the statistic
        specifies what exactly we're measuring for the frames matching our predicate.
        """
        self.oracle_df = oracle_df
        self.oracle_limit = query.oracle_limit
        self.predicate = query.predicate
        self.statistic_fcn = statistic_fcn

    def predict(self, frame):
        """
        Select all entries in the oracle dataset for the given frame and apply
        the statistic function to compute the oracle prediction(s).
        """
        # filter for frame of interest then filter on predicate
        frame_df = self.oracle_df[self.oracle_df.frame == frame]
        matches_predicate = (
            not frame_df.query(self.predicate).empty
            if self.predicate != ""
            else not frame_df.empty
        )

        return self.statistic_fcn(frame_df), matches_predicate

    @abstractmethod
    def sample_frame(self, frame, *args, **kwargs):
        """
        The abstract method for deciding whether or not to sample from
        the oracle for a given frame.
        """
        pass


class Oracle(BaseOracle):
    """
    The oracle takes in a CSV with the oracle predictions and uses them
    to make its predictions. It also takes in the statistic function
    which is applied to the dataframe with the oracle predictions.
    """
    def __init__(self, oracle_df, query, statistic_fcn):
        """
        Read in the CSV with per-frame proxy predictions.
        """
        super().__init__(oracle_df, query, statistic_fcn)

    def sample_frame(self, frame, *args, **kwargs):
        """
        No longer useful
        """
        pass


class UniformOracle(BaseOracle):
    """
    An implementation of the oracle sampling logic that samples from the oracle
    uniformly throughout the video.
    """
    def __init__(self, oracle_predicate_df, query, statistic_fcn):
        """
        This constructor implements the base class constructor and also takes in
        a function for computing the statistic of interest and the query. The
        query is used for determining the range of frames over which we uniformly
        sample.
        """
        super().__init__(oracle_predicate_df, query, statistic_fcn)

        # compute uniformly spaced frames
        self.frames_to_sample = set(np.linspace(
            query.start_frame,
            query.end_frame,
            num=self.oracle_limit,
            dtype=int,
        ))

    def sample_frame(self, frame, *args, **kwargs):
        """
        Implementation of the abstract method for deciding whether to sample the oracle
        for the given frame. We return True if the input frame is one of the oracle_limit
        uniformly spaced frames in [start_frame, start_frame + time_limit]
        """
        return int(frame in self.frames_to_sample)


class RandomOracle(BaseOracle):
    """
    An implementation of the oracle sampling logic that samples from the oracle
    at random throughout the video. It will sample exactly oracle_limit times
    (provided that the number of frames in the video is larger than the oracle_limit).
    """
    def __init__(self, oracle_predicate_df, query, statistic_fcn):
        """
        This constructor implements the base class constructor and also takes in
        a function for computing the statistic of interest and the query. The
        query is used for determining the range of frames over which we randomly
        sample.
        """
        super().__init__(oracle_predicate_df, query, statistic_fcn)

        # TODO: fix if self.oracle_limit == len(frames)
        # compute uniformly spaced frames including final frame
        frames = np.arange(query.start_frame, query.end_frame + 1)
        self.frames_to_sample = set(
            np.random.choice(frames, size=min(self.oracle_limit, len(frames)), replace=False)
        )

    def sample_frame(self, frame):
        """
        Implementation of the abstract method for deciding whether to sample the oracle
        for the given frame. We return True if the input frame is one of the oracle_limit
        randomly chosen frames in [start_frame, start_frame + time_limit]
        """
        return int(frame in self.frames_to_sample)


class ProbTransitionOracle(BaseOracle):
    """
    An implementation of the oracle sampling logic that samples from the oracle
    based on changes in proxy values. The oracle sampling can be triggered based
    on thresholding on the proxy values or thresholding on the difference(s) of
    the proxy values. Sampling can be done greedily or uniformly. In the greedy
    case we sample the oracle until we reach the oracle_limit. In the uniform
    case we spread the samples across buckets based on time. Samples are used
    greedily within each bucket, and if any are left by the end of the bucket
    they are spread uniformly across the remaining buckets
    """
    # TODO: specify num continuous oracle samples for when threshold is exceeded
    # TODO: use more than one sample and/or immediate differences (or use smoothing)
    # TODO: ? learn from sampled oracle values ? (this will probably be its own sampler)
    def __init__(
        self, oracle_predicate_df, query, statistic_fcn,
        threshold, difference=False, greedy=True, samples_per_bucket=1,
    ):
        """
        This constructor implements the base class constructor and also takes in
        a function for computing the statistic of interest and the query. The
        query is used for determining the range of frames over which we randomly
        sample.
        """
        super().__init__(oracle_predicate_df, query, statistic_fcn)

        # keep track of history of proxy values
        self.proxy_history = []

        # keep track of oracle samples we have left
        self.oracle_samples_left = self.oracle_limit

        # store values used for frame sampling logic
        self.threshold = threshold
        self.difference = difference
        self.greedy = greedy

        # if we're using unform sampling set up buckets for oracle samples
        if not greedy:
            # set index of first oracle bucket
            self.bucket_idx = 0

            # create buckets
            self.buckets = np.linspace(
                query.start_frame,
                query.end_frame,
                num=self.oracle_limit//samples_per_bucket,
                dtype=int,
                endpoint=False,
            )

            # spread oracle samples evenly across buckets; overflow will go
            # to the earlier buckets since excess can be spread across subsequent
            # buckets
            self.samples_per_bucket = np.zeros(len(self.buckets), dtype=int)
            for idx in range(self.oracle_limit):
                bucket_idx = idx % len(self.buckets)
                self.samples_per_bucket[bucket_idx] += 1
    
    def update_sample_buckets(self, frame):
        """
        This helper function carries over oracle samples to the remaining buckets
        when the frame passes into the next bucket. It also returns the current
        bucket and the number of samples left in that bucket.
        """
        # if this is the final bucket, simply return the bucket index and samples remaining
        if self.bucket_idx == len(self.buckets) - 1:
            return self.bucket_idx, self.samples_per_bucket[self.bucket_idx]

        # update current bucket and carry over samples if necessary
        next_bucket_start_frame = self.buckets[self.bucket_idx + 1]
        if frame >= next_bucket_start_frame:
            # carry over any samples that remain in the current bucket
            samples_left = self.samples_per_bucket[self.bucket_idx]
            for idx in range(samples_left):
                buckets_left = len(self.buckets) - (self.bucket_idx + 1)
                next_bucket_idx = (self.bucket_idx + 1) + (idx % buckets_left)
                self.samples_per_bucket[next_bucket_idx] += 1

            # set samples in current bucket to 0 and update bucket_idx
            self.samples_per_bucket[self.bucket_idx] = 0
            self.bucket_idx += 1

        return self.bucket_idx, self.samples_per_bucket[self.bucket_idx]

    def sample_frame(self, frame, proxy_pred_to_prob, *args, **kwargs):
        """
        Implementation of the abstract method for deciding whether to sample the oracle
        for the given frame. We return True if the proxy value (or its difference) exceeds
        the threshold passed into the class.
        """
        # let proxy_val be the combined probability of the non-zero prediction probabilities
        proxy_val = proxy_pred_to_prob[1] + proxy_pred_to_prob[2]

        # if we need to compute a difference then return False if this is the first sample
        if not self.proxy_history and self.difference:
            self.proxy_history.append(proxy_val)
            return False
        
        # handle the greedy sampling case
        if self.greedy:
            # if we're not taking differences, sample the frame if the proxy value exceeds the threshold
            if not self.difference and self.oracle_samples_left > 0:
                sample = proxy_val > self.threshold
                if sample:
                    self.oracle_samples_left -= 1

                return sample

            # if we're taking differences, sample the frame if the difference in proxy values exceeds the threshold
            elif self.difference:
                last_proxy_val = self.proxy_history[-1]
                self.proxy_history.append(proxy_val)
                if self.oracle_samples_left > 0:
                    sample = proxy_val - last_proxy_val > self.threshold
                    if sample:
                        self.oracle_samples_left -= 1

                    return sample
        
        # handle the uniform sampling case
        else:
            bucket_idx, samples_left_in_bucket = self.update_sample_buckets(frame)

            # if we're not taking differences, sample the frame if the proxy value exceeds the threshold
            if not self.difference and samples_left_in_bucket > 0:
                sample = proxy_val > self.threshold
                if sample:
                    self.samples_per_bucket[bucket_idx] -= 1

                return sample

            # if we're taking differences, sample the frame if the difference in proxy values exceeds the threshold
            elif self.difference:
                last_proxy_val = self.proxy_history[-1]
                self.proxy_history.append(proxy_val)
                if samples_left_in_bucket > 0:
                    sample = proxy_val - last_proxy_val > self.threshold
                    if sample:
                        self.samples_per_bucket[bucket_idx] -= 1

                    return sample

        # if we have no samples left return False
        return False
