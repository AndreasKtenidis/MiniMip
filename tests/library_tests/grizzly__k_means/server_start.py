import asyncio
from mini_mip_system.server.grpc_agg_server import serve

if __name__ == "__main__":
    asyncio.run(serve(available_clients=2))