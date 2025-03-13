import random
import grpc
import concurrent.futures
from constants import AGG
import grpc_example.aggregator_pb2 as pb2
import grpc_example.aggregator_pb2_grpc as pb2_grpc
from client.aggregation_client import NumpyAggregationClient


class GRPCClient(NumpyAggregationClient):
    def __init__(self):
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = pb2_grpc.AggregatorStub(self.channel)
        self.agg_round = 0
        self.operation_id = 0

    def __global_sum__(self, local_sum):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.SUM, self.agg_round, [local_sum])

    def __global_count__(self, local_count):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.COUNT, self.agg_round, [local_count])

    def __global_avg__(self, local_sum, local_count):
        self.agg_round += 1
        return self.send_aggregation_request(self.operation_id, AGG.AVG, self.agg_round, [local_sum, local_count])

    def send_aggregation_request(self, operation_id, agg_func, agg_round, values):
        request = pb2.Agg(
            operation_id=operation_id,
            agg_func=agg_func.value,
            agg_round=agg_round,
            values=values
        )
        print(f"Sending request: {request}")

        response = self.stub.GetServerResponse(request)
        print(f"Received response: {response.answer}")
        return response


def run_client():
    client = GRPCClient()
    random_number = random.randint(1, 10)
    print(f"Client {random_number} started")
    client.__global_sum__(random_number)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_client) for _ in range(3)]

        # Wait for all clients to finish
        concurrent.futures.wait(futures)
