import torch
from flwr.simulation import run_simulation
import FlowerServer,FlowerClient



NUM_PARTITIONS = 10
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU


# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

client = FlowerClient.app
server = FlowerServer.app

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)

print(server)