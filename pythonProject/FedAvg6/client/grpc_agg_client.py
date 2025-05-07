from typing import Dict, Union, List, Any

import grpc
import concurrent.futures
from constants import AGG,client_count
import grpc_.aggregator_pb2 as pb2
import grpc_.aggregator_pb2_grpc as pb2_grpc
from client.aggregation_client import AggregationClient
from data.experiment_datasets.blobs_dataset import BlobDataset
from data.experiment_datasets.blobs_dataset2 import BlobDataset2
from data.experiment_datasets.calibration_dataset import CalibrationDataset
from data.experiment_datasets.iris_dataset import IrisDataset

import random
import inspect



from function.abstract_function import AggFunc
from library.bivariate_statistics import PearsonCorrelation, Covariance, LeastSquaresRegression, SumOfProducts
from library.univariate_statistics import Variance
from library.calibration_belt import CalibrationBelt
from library.k_means import KMeans


class GRPCClient(AggregationClient):

    def __init__(self,client_id,client_count2,seed):
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = pb2_grpc.AggregatorStub(self.channel)
        self.agg_round = 0
        self.operation_id = 0
        self.client_id = client_id
        self.client_count = client_count2
        self.random = random.Random(seed)



    def __global_sum__(self, local_sum):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.SUM, self.agg_round, local_sum)

    def __global_min__(self, local_min):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.MIN, self.agg_round, local_min)

    def __global_max__(self, local_max):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.MAX, self.agg_round, local_max)

    def send_aggregation_request(self, operation_id, agg_func, agg_round, array):
        request = pb2.Agg(
            operation_id=operation_id,
            agg_func=agg_func.value,
            agg_round=agg_round,
            values=array
        )
        response = self.stub.GetServerResponse(request)
        return response.answer

    @staticmethod
    def get_clientapp_dataset(dataset,partition_id: int, num_partitions: int):
        return dataset(partition_id=partition_id, num_partitions=num_partitions)


    def map_and_execute(self,dataset,agg_class:type[AggFunc], mapping: Dict[str, Union[str, List[str]]],constants: Dict[str,Any]):
        aggregation_function = agg_class.__new__(agg_class)
        aggregation_function.__init__(self)
        dataset = GRPCClient.get_clientapp_dataset(dataset,self.client_id,self.client_count )
        local_input={}
        for key, value in mapping.items():
            if isinstance(value, list):
                local_input[key] = dataset.get_attributes(*value)
            else:
                local_input[key] = dataset.get_attribute(value)

        # Get function parameters
        # compute = aggregation_function.compute
        params = inspect.signature(aggregation_function.compute).parameters
        param_names = params.keys()
        # Extract relevant arguments from the dictionary
        mapped_args = {param: local_input[param] for param in param_names if param in mapping}
        mapped_args.update(constants)
        # Execute the function with the mapped arguments
        # Execute the function with the mapped arguments
        return aggregation_function.compute(**mapped_args)

def run_client(client_id, client_c):
    client = GRPCClient(client_id, client_c,154)
    # answer = client.map_and_execute(dataset = IrisDataset,agg_class=LeastSquaresRegression,mapping={'x':'SepalWidthCm','y':'SepalLengthCm'},constants={})
    # answer = client.map_and_execute(dataset=IrisDataset2, agg_class=PearsonCorrelation,mapping={'x': 'SepalWidthCm', 'y': 'SepalLengthCm'}, constants={})
    # answer = client.map_and_execute(dataset=IrisDataset, agg_class=Variance,mapping={'x': 'SepalWidthCm', 'y': 'SepalLengthCm'}, constants={})
    # answer = client.map_and_execute(dataset=BlobDataset2, agg_class=KMeans, mapping={'x': ['x', 'y']}, constants={'k': 3})
    answer = client.map_and_execute(dataset=CalibrationDataset, agg_class=CalibrationBelt, mapping={'o':'target','e': 'SVM'},constants={})
    # print(answer)

if __name__ == "__main__":
    # run_client(0,0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
        futures = [executor.submit(run_client, client_id, client_count) for client_id in range(client_count)]
        # Wait for all clients to finish
        concurrent.futures.wait(futures)
