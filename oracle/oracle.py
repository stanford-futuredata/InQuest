from abc import ABC, abstractmethod


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
