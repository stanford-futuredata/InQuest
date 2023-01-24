from abc import ABC, abstractmethod

import random


class BaseProxy(ABC):
    """
    An abstract base class for mocking the behavior of our proxy model.
    """
    def __init__(self):
        """
        No base initialization.
        """
        pass

    @abstractmethod
    def predict(self, frame):
        """
        The abstract method for computing the proxy prediction for a given frame.
        """
        pass


class RandomProxy(BaseProxy):
    """
    This proxy returns a random real in the interval [0, 1] for
    every prediction.
    """

    def predict(self, frame):
        """
        Return a random real in the interval [0, 1].
        """
        return random.random()


class RandomIntervalProxy(BaseProxy):
    """
    This proxy returns a random number in the interval specified by
    the arguments to the init function. Also allows for specifying
    whether the proxy should predict reals or integers in that interval.
    """
    def __init__(self, min, max, reals=True):
        """
        Set min and max bounds for our interval and specify whether it
        should produce reals or integers.
        """
        self.min = min
        self.max = max
        self.reals = reals

    def predict(self, frame):
        """
        Return a random real or integer in the interval [min, max].
        """
        number = (
            self.min + random.random() * (self.max - self.min)
            if self.reals
            else random.randint(self.min, self.max)
        )

        return number


class PrecomputedProxy(BaseProxy):
    """
    This proxy takes in a CSV with per-frame proxy predictions that
    have been computed ahead of time.
    """
    def __init__(self, proxy_df, count_col=None, count_cols=[], statistic_fcn="count", weighted=False):
        """
        Read in the CSV with per-frame proxy predictions.
        """
        self.proxy_pred_df = proxy_df
        self.count_col = count_col
        self.count_cols = count_cols
        self.statistic_fcn = statistic_fcn
        self.weighted = weighted

    def predict(self, frame):
        """
        Return the proxy prediction for this frame.
        """
        # filter for the frame in question
        frame_df = self.proxy_pred_df[self.proxy_pred_df.frame == frame]
        pred = frame_df['proxy'].iloc[0]

        # if statistic_fcn is `at_least_one` then apply that to the pred
        if self.statistic_fcn == "at_least_one":
            pred = int(pred > 0)

        return pred


class OracleProxy(BaseProxy):
    """
    This proxy takes in a CSV with the oracle predictions and uses them
    to make its predictions. It also takes in the statistic function
    which is applied to the dataframe with the oracle predictions.
    """
    def __init__(self, oracle_predicate_df, statistic_fcn):
        """
        Read in the CSV with per-frame proxy predictions.
        """
        self.oracle_df = oracle_predicate_df
        self.statistic_fcn = statistic_fcn

    def predict(self, frame):
        """
        Return all proxy predictions for the given frame.
        """
        frame_df = self.oracle_df[self.oracle_df.frame == frame]

        return self.statistic_fcn(frame_df)
