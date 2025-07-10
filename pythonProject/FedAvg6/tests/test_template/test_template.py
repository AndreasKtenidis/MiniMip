from library.templates.partitioned_table import PartitionedPandasTable
from system.client.grpc_agg_client import GRPCClient
from abc import ABC, abstractmethod

class FederationTestTemplate(ABC):

    def __init__(self, partition_id, number_of_partitions):
        self.client:GRPCClient = GRPCClient(partition_id, number_of_partitions)
        pt_dataset:PartitionedPandasTable= self.get_partitioned_pandas_table()
        self.local_dataset = pt_dataset.get_local_dataset(partition_id, number_of_partitions)
        self.global_dataset = self.get_partitioned_pandas_table().get_dataset()

    def __call__(self):
        local_output = self.local_computation()
        global_output = self.global_computation()
        self.compare(local_output,global_output)

    @abstractmethod
    def local_computation(self):
        pass

    @abstractmethod
    def global_computation(self):
        pass

    @abstractmethod
    def get_partitioned_pandas_table(self)->PartitionedPandasTable:
        pass

    @abstractmethod
    def compare(self, local_output,global_output):
        pass

