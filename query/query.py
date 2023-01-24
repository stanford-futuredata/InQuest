from abc import ABC


class BaseQuery(ABC):
    """
    A base class for selection and aggregation queries containing the features
    that that must be present for each query type.
    """

    def __init__(self, query_config):
        """
        Initialize query features that are shared by selection and aggregation.
        """
        self.query_type = query_config['type']
        self.oracle_limit = query_config['oracle_limit']
        self.start_frame = query_config['start_frame']
        self.time_limit = query_config['time_limit']
        self.predicate = query_config['predicate']

    @property
    def end_frame(self):
        return self.start_frame + self.time_limit


class AggregationQuery(BaseQuery):
    """
    A class definition for aggregation queries.
    """

    def __init__(self, query_config, oracle_idx):
        """
        Initialize query features for aggregation queries.
        """
        # initialize base features
        super().__init__(query_config)

        # override oracle_limit
        self.oracle_limit = query_config['oracle_limit'][oracle_idx]
