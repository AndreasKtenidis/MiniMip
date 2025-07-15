from library.templates.partitioned_table import PartitionedPandasTable
from system.client.grpc_agg_client import GRPCClient
from abc import ABC, abstractmethod

class FederationTestTemplate(ABC):

    def __init__(self, partition_id, number_of_partitions,*,operation_id=0):
        self.client:GRPCClient = GRPCClient(partition_id, number_of_partitions,operation_id=operation_id)
        pt_dataset:PartitionedPandasTable= self.get_partitioned_pandas_table()
        # Creating the federated and the centralized versions of the same dataset
        local_dataset = pt_dataset.get_local_dataset(partition_id, number_of_partitions)
        global_dataset = self.get_partitioned_pandas_table().get_dataset()
        # Executing computations in federated and centralized mode
        local_output = self.federated_computation(local_dataset)
        global_output = self.centralized_computation(global_dataset)
        # Comparing results in federated and centralized mode
        self.compare(local_output, global_output)


    @abstractmethod
    def federated_computation(self, local_dataset):
        pass

    @abstractmethod
    def centralized_computation(self, centralized_dataset):
        pass

    @abstractmethod
    def get_partitioned_pandas_table(self)->PartitionedPandasTable:
        pass

    @abstractmethod
    def compare(self, federated_output, global_output):
        pass