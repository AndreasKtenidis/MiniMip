from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

from data.abstract_table import AbstractTable


class IrisDataset2(AbstractTable):

    def get_attributes(self, *attributes: str):
        pass

    # Static attribute to store the global fds object
    fds = None

    def __init__(self, partition_id, num_partitions):
        # Initialize the data only once
        if IrisDataset2.fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            IrisDataset2.fds = FederatedDataset(
                dataset="scikit-learn/iris",
                partitioners={"train": partitioner}
            )
        # Load a specific partition and format it as pandas DataFrame
        self.dataset = IrisDataset2.fds.load_partition(partition_id, "train").with_format("pandas")[:]

    def get_attribute(self, attribute):
        return self.dataset[attribute]

    def get_attribute_names(self):
        return list(self.dataset.keys())