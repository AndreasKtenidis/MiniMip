import grpc
from concurrent import futures
import grpc_example.aggregator_pb2 as pb2
import grpc_example.aggregator_pb2_grpc as pb2_grpc
import  server.aggregation_server as server
from constants import AGG
import asyncio  # Import asyncio to use asyncio.sleep
import traceback

class GRPCServer(pb2_grpc.AggregatorServicer, server.NumpyAggregationServer):
    def __init__(self):
        self.operations = {}
        self.answers = {}
        self.lock = asyncio.Lock()  # Async lock for thread safety
        self.available_clients = 3

    async def GetServerResponse(self, request, context):
        print("GetServerResponse", request)
        triple = (request.operation_id, request.agg_func, request.agg_round)
        async with self.lock:
            if triple not in self.operations:
                self.operations[triple] = list(request.values)
            else:
                print(list(request.values))
                self.operations[triple].extend(list(request.values))
        # Simulate async processing
        while len(self.operations[triple]) != self.available_clients:
            await asyncio.sleep(0.05)

        # Now compute the sum outside the lock
        try:
            if triple not in self.answers:
                async with self.lock:
                    if request.agg_func == AGG.SUM.value:
                        self.answers[triple] = self.sum(self.operations[triple])
                    elif request.agg_func == AGG.COUNT.value:
                        self.answers[triple] = self.count(self.operations[triple])
                    elif request.agg_func == AGG.AVG.value:  # Corrected here
                        self.answers[triple] = self.avg(self.operations[triple])
        except Exception as e:
            traceback.print_exc()
        response = pb2.AggResponse(answer= self.answers[triple])
        print(f"Sending response: {response}")
        return response


async def serve():
    server = grpc.aio.server()
    pb2_grpc.add_AggregatorServicer_to_server(GRPCServer(), server)
    server.add_insecure_port("[::]:50051")
    print("gRPC Server running on port 50051...")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())