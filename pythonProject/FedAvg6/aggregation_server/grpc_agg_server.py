import grpc
from concurrent import futures
import grpc_example.aggregator_pb2 as pb2
import grpc_example.aggregator_pb2_grpc as pb2_grpc
import  aggregation_server
from constants import AGG
import threading
import asyncio  # Import asyncio to use asyncio.sleep






class GRPCServer(pb2_grpc.AggregatorServicer,aggregation_server.NumpyAggregationServer):

    def __init__(self):
        self.operations = {}
        self.answers = {}
        self.lock = threading.Lock()
        self.available_clients = 3

    def GetServerResponse(self, request, context):
        print("GetServerResponse", request)

        # Creating a tuple to store operation details
        triple = (request.operation_id, request.agg_func, request.agg_round)

        # Thread-safe section: ensuring no other thread accesses this data simultaneously
        with self.lock:
            # Add or append the values to the operations dictionary
            if triple not in self.operations:
                self.operations[triple] = [request.values]
            else:
                self.operations[triple].append(request.values)

            # When the required number of clients have sent data, perform aggregation
            if len(self.operations[triple]) == self.available_clients:
                if request.agg_func == AGG.SUM.value:
                    self.answers[triple] = self.sum(self.operations[triple])
                elif request.agg_func == AGG.COUNT.value:
                    self.answers[triple] = self.count(self.operations[triple])
                elif request.agg_func == AGG.AVG.value:  # Corrected here
                    self.answers[triple] = self.avg(self.operations[triple])

        # Wait until the result is available in the answers dictionary
        # while triple not in self.answers:
        #     await asyncio.sleep(0.2)  # Non-blocking sleep, allowing event loop to continue

        # Get the result and create the response
        # result = self.answers[triple]
        result=3.0
        response = pb2.AggResponse(answer=result)

        print(f"Sending response: {response}")

        return response



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_AggregatorServicer_to_server(GRPCServer(), server)
    server.add_insecure_port("[::]:50051")
    print("gRPC Server running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
