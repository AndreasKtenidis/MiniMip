from library.core.partitioned_table import PartitionedPandasTable
from mini_mip_system.client.grpc_agg_client import GRPCClient
from abc import ABC, abstractmethod

from tests_and_experiments.core.grizzly_table_factory import GrizzlyFactory


class GrizzlyFederationTestTemplate(ABC):

    def __init__(self, client_id, client_count, *, dataset:PartitionedPandasTable, operation_id=0, aggregation_server="localhost:50051"):
        self.client:GRPCClient = GRPCClient(client_id, client_count, operation_id=operation_id,aggregation_server=aggregation_server)
        # Creating the federated and the centralized versions of the same dataset, for the federated use a Grizzly persion of the dataset
        factory = GrizzlyFactory()
        conn, local_dataset = factory.create_db(dataset, client_id = client_id, num_clients = client_count)
        # Creating the global dataset
        global_dataset = dataset.get_global_dataset()
        # Executing computations in federated and centralized mode
        local_output = self.federated_computation(conn,local_dataset)
        global_output = self.centralized_computation(global_dataset)
        # Comparing results in federated and centralized mode
        self.compare(local_output, global_output)

    @abstractmethod
    def federated_computation(self,conn, local_dataset):
        pass

    @abstractmethod
    def centralized_computation(self, centralized_dataset):
        pass


    @abstractmethod
    def compare(self, federated_output, global_output):
        pass




