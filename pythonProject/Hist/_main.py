import torch
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import  ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation

from pythonProject.Hist import Client_app, server_app

# Flower ClientApp
client = Client_app.client
server = server_app.server


import torch

NUM_PARTITIONS = 10
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

from flwr.server import ServerApp, ServerConfig, ServerAppComponents


def server_fn(context: Context) -> ServerAppComponents:
    # Configure the server for just 3 rounds of training
    print(context)
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(
        config=config,
        # strategy=FedCustom(),  # <-- pass the new strategy here
    )

from flwr.simulation import run_simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)