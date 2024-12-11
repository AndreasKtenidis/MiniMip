import torch
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation

from fed_Custom import FedCustom
from flowerClient import client_fn

NUM_PARTITIONS = 10
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

def server_fn(context: Context) -> ServerAppComponents:
    # Configure the server for just 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(
        config=config,
        # strategy=FedCustom(),  # <-- pass the new strategy here
    )

# Create the ServerApp
server = ServerApp(server_fn=server_fn)
# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)