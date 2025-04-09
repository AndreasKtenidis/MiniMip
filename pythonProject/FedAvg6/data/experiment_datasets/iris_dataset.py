import numpy as np

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

from data.abstract_table import AbstractTable


class IrisDataset(AbstractTable):

    def get_attributes(self, *attributes: str):
        pass

    # Static attribute to store the global fds object
    fds = None

    def __init__(self, partition_id, num_partitions):
        # Initialize the data only once
        if IrisDataset.fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            IrisDataset.fds = FederatedDataset(
                dataset="scikit-learn/iris",
                partitioners={"train": partitioner}
            )
        # Load a specific partition and format it as pandas DataFrame
        self.dataset = IrisDataset.fds.load_partition(partition_id, "train").with_format("pandas")[:]

    def get_attribute(self, attribute)->np.ndarray:
        return self.dataset[attribute].values

    def get_attribute_names(self):
        return list(self.dataset.keys())