import random
import grpc
import concurrent.futures
from constants import AGG,client_count
import grpc_example.aggregator_pb2 as pb2
import grpc_example.aggregator_pb2_grpc as pb2_grpc
from client.aggregation_client import NumpyAggregationClient
from dataset.dataset_pandas import PandasDataset
import inspect
import numpy as np

from function.abstract_function import AggFunc
from function.bivariate_statistics import PearsonCorrelation
from function.univariate_statistics import SumOfSquares


class GRPCClient(NumpyAggregationClient):

    def __init__(self,client_id,client_count2):
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = pb2_grpc.AggregatorStub(self.channel)
        self.agg_round = 0
        self.operation_id = 0
        self.client_id = client_id
        self.client_count = client_count2

    def __global_sum__(self, local_sum):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.SUM, self.agg_round, [local_sum])

    def __global_count__(self, local_count):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.COUNT, self.agg_round, [local_count])

    def __global_avg__(self, local_sum, local_count):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.AVG, self.agg_round, [local_sum, local_count])

    def __global_min__(self, local_min):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.MIN, self.agg_round, [local_min])

    def __global_max__(self, local_max):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.MAX, self.agg_round, [local_max])

    def send_aggregation_request(self, operation_id, agg_func, agg_round, values):
        request = pb2.Agg(
            operation_id=operation_id,
            agg_func=agg_func.value,
            agg_round=agg_round,
            values=values
        )

        response = self.stub.GetServerResponse(request)
        return response

    @staticmethod
    def get_clientapp_dataset(partition_id: int, num_partitions: int):
        return PandasDataset(partition_id=partition_id, num_partitions=num_partitions)


    def map_and_execute(self,agg_class:type[AggFunc], mapping):
        aggregation_function = agg_class.__new__(agg_class)
        aggregation_function.__init__(self)
        dataset = GRPCClient.get_clientapp_dataset(self.client_id,self.client_count )
        local_input={}
        for key, value in mapping.items():
            local_input[key] = dataset.get_attribute(value)

        # Get function parameters
        # compute = aggregation_function.compute
        params = inspect.signature(aggregation_function.compute).parameters
        param_names = params.keys()
        # Extract relevant arguments from the dictionary
        mapped_args = {param: local_input[param] for param in param_names if param in mapping}
        # Execute the function with the mapped arguments
        return aggregation_function.compute(**mapped_args)

def run_client(client_id, client_c):
    client = GRPCClient(client_id, client_c)
    answer = client.map_and_execute(agg_class=PearsonCorrelation,mapping={'x':'SepalWidthCm','y':'SepalLengthCm'})
    print(answer)

if __name__ == "__main__":
    # run_client(0,0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
        futures = [executor.submit(run_client, client_id, client_count) for client_id in range(client_count)]
        # Wait for all clients to finish
        concurrent.futures.wait(futures)
