from sklearn.datasets import make_classification


from library.stats.k_means_rand import KMeansRand
from system.client.grpc_agg_client import GRPCClient
from server import available_clients
from tests.help_datasets.blob import BlobDataset
import time
from  pandas import DataFrame


def start_client(*,aggregation_server="localhost:50051",client_id):
    dataset:DataFrame = BlobDataset(n_samples=1000, centers=3).get_local_dataset(client_id,available_clients)
    client: GRPCClient = GRPCClient(client_id, available_clients, operation_id=0, aggregation_server=aggregation_server)
    # Creating the federated and the centralized versions of the same dataset
    model:KMeansRand = KMeansRand(client)


    # Timing starts
    start_time = time.time()
    model.compute(dataset.values,3)
    # Timing ends
    end_time = time.time()
    # Calculate elapsed time in milliseconds
    elapsed_time_ms = (end_time - start_time) * 1000
    print(elapsed_time_ms)

if __name__ == "__main__":
    start_client(client_id=0)