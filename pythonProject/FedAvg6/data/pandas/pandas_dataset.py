

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

from data.dataset import Dataset


class PandasDataset(Dataset):
    # Static attribute to store the global fds object
    fds = None

    def __init__(self, partition_id, num_partitions):
        # Initialize the data only once
        if PandasDataset.fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            PandasDataset.fds = FederatedDataset(
                dataset="scikit-learn/iris",
                partitioners={"train": partitioner}
            )

        # Load a specific partition and format it as pandas DataFrame
        self.dataset = PandasDataset.fds.load_partition(partition_id, "train").with_format("pandas")[:]

    def get_attribute(self, attribute):
        return self.dataset[attribute].values

    def get_type(self,attribute):
        return self.dataset[attribute].dtype

    def get_attributes(self):
        return list(self.dataset.keys())



