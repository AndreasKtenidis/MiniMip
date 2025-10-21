import grpc
import numpy as np

from mini_mip_system.constants import AGG,client_count
import mini_mip_system._grpc.aggregator_pb2 as pb2
import mini_mip_system._grpc.aggregator_pb2_grpc as pb2_grpc
from library.utils.aggregation_client import AggregationClientInterface


import random



class GRPCClient(AggregationClientInterface):

    def __init__(self, client_id, client_count, *, seed=1234, operation_id=0, aggregation_server="localhost:50051"):
        self.channel = grpc.insecure_channel(aggregation_server)
        self.stub = pb2_grpc.AggregatorStub(self.channel)
        self.agg_round = 0
        self.operation_id = operation_id
        self.client_id = client_id
        self.client_count = client_count
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
