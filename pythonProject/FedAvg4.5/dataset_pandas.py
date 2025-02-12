from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset


class PandasDataset:
    # Static attribute to store the global fds object
    fds = None

    def __init__(self, partition_id, num_partitions):
        # Initialize the dataset only once
        if PandasDataset.fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            PandasDataset.fds = FederatedDataset(
                dataset="scikit-learn/iris",
                partitioners={"train": partitioner}
            )
        # Load a specific partition and format it as pandas DataFrame
        self.dataset = PandasDataset.fds.load_partition(partition_id, "train").with_format("pandas")[:]

    def get_data(self):
        return self.dataset

    def get_attribute(self, value):
        return self.dataset[value].values

# dataset:PandasDataset =  PandasDataset(0,1)
# x:ndarray =dataset.get_attribute("SepalLengthCm")
# print(x)