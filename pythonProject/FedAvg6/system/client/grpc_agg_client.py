from typing import Dict, Union, List, Any

import grpc
import concurrent.futures
import numpy as np

from data.experiment_datasets.calibration_dataset import CalibrationDataset

from constants import AGG,client_count
import system._grpc.aggregator_pb2 as pb2
import system._grpc.aggregator_pb2_grpc as pb2_grpc
from system.client.aggregation_client import AggregationClient
# from data.experiment_datasets.blobs_dataset import BlobDataset
# from data.experiment_datasets.calibration_dataset import CalibrationDataset
# from data.experiment_datasets.pandas_datasets.diabetes_dataset import DiabetesDataset
# from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
from library.stats._statistical_function import AggFunc
from library.stats.calibration_belt import CalibrationBelt

import random
import inspect





# from library.fd_models.logistic_regression2 import FederatedLogisticRegression


class GRPCClient(AggregationClient):

    def __init__(self,client_id,client_count2,seed):
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = pb2_grpc.AggregatorStub(self.channel)
        self.agg_round = 0
        self.operation_id = 0
        self.client_id = client_id
        self.client_count = client_count2
        self.random = random.Random(seed)


    def __global_union__(self, categories,c_type):
        self.agg_round += 1
        if c_type == np.int64:
            return self.send_int_aggregation_request(self.operation_id, AGG.UNION, self.agg_round, categories)
        if c_type == np.float64:
            return self.send_aggregation_request(self.operation_id, AGG.UNION, self.agg_round, categories)
        if c_type == str:
            return self.send_category_aggregation_request(self.operation_id, AGG.UNION, self.agg_round, categories)
        else:
            raise Exception("Not Supported Type")

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

    def send_int_aggregation_request(self, operation_id, agg_func, agg_round, array):
        request = pb2.IntegerAgg(
            operation_id=operation_id,
            agg_func=agg_func.value,
            agg_round=agg_round,
            values=array
        )
        response = self.stub.GetIntServerResponse(request)
        return response.answer

    def send_category_aggregation_request(self, operation_id, agg_func, agg_round, array):
        request = pb2.CategoryAgg(
            operation_id=operation_id,
            agg_func=agg_func.value,
            agg_round=agg_round,
            values=array
        )
        response = self.stub.GetCategoryServerResponse(request)
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
    # answer = client.map_and_execute(dataset=IrisDataset, agg_class=StandardizedMeanDifferences, mapping={'x': 'SepalWidthCm', 'y': 'SepalLengthCm'}, constants={})
    # answer = client.map_and_execute(dataset=BlobDataset, agg_class=KMeans, mapping={'x': ['x', 'y']}, constants={'k': 3})
    # answer = client.map_and_execute(dataset=InsuranceDataset, agg_class=FederatedLinearRegression,
    #                                 mapping={'input': ['age', 'bmi', 'children', 'sex', 'smoker', 'region_northeast',
    #    'region_northwest', 'region_southeast', 'region_southwest', 'charges'], 'output' :'charges'},constants={})
    # answer = client.map_and_execute(dataset=InsuranceDataset, agg_class=FedOLS,
    #                                 mapping={'x': ['age', 'bmi', 'children', 'sex', 'smoker', 'region_northeast',
    #    'region_northwest', 'region_southeast', 'region_southwest', 'charges'], 'y' :'charges'},constants={})
    # answer = client.map_and_execute(dataset=TitanicPandasDataset, agg_class=FederatedLogisticRegression,
    #                                 mapping={'input': ['Age', 'Fare', 'Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
    #                                                    'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1',
    #                                                    'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6'], 'output': ['Survived']}, constants={})
    # answer = client.map_and_execute(dataset=DiabetesDataset, agg_class=FederatedLogisticRegressionLBFGS,
    #                                 mapping={'x': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    #                                          'y': ['Outcome']}, constants={})
    answer = client.map_and_execute(dataset=CalibrationDataset, agg_class=CalibrationBelt, mapping={'o':'target','e': 'SVM'},constants={})
    # answer = client.map_and_execute(dataset=TitanicPandasDataset, agg_class=ChiSquared, mapping={'factor_to_outcome': ['Pclass','Survived']}, constants={}) #

    # FedOneHotEncoder
    print(answer)

if __name__ == "__main__":
    # run_client(0,0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
        futures = [executor.submit(run_client, client_id, client_count) for client_id in range(client_count)]
        # Wait for all clients to finish
        concurrent.futures.wait(futures)
