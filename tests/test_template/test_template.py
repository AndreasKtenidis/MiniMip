from library.templates.partitioned_table import PartitionedPandasTable
from mini_mip_system.client.grpc_agg_client import GRPCClient
from abc import ABC, abstractmethod

class FederationTestTemplate(ABC):

    def __init__(self, client_id, client_count, *, dataset:PartitionedPandasTable, operation_id=0, aggregation_server="localhost:50051"):
        self.client:GRPCClient = GRPCClient(client_id, client_count, operation_id=operation_id,aggregation_server=aggregation_server)
        # Creating the federated and the centralized versions of the same dataset
        local_dataset = dataset.get_local_dataset(client_id, client_count)
        global_dataset = dataset.get_dataset()
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

    # @abstractmethod
    # def get_partitioned_pandas_table(self)->PartitionedPandasTable:
    #     pass

    @abstractmethod
    def compare(self, federated_output, global_output):
        pass