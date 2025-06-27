import subprocess
import time
import os
import signal

def launch_server():
    return subprocess.Popen(
        ["python", "-m", "pythonProject.FedAvg6.system.server.grpc_agg_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Use a new session to allow killing the process group
    )

def run_clients(client1, client2):
    start = time.time()
    p1 = subprocess.Popen(["python", "-m", client1])
    p2 = subprocess.Popen(["python", "-m", client2])
    p1.wait()
    p2.wait()
    end = time.time()
    return end - start

import socket

def wait_for_port(port, host='127.0.0.1', timeout=10.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"Server on {host}:{port} not ready after {timeout} seconds")

def terminate_server(server_proc):
    os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
    server_proc.wait()

def main():
    print("=== Pandas Benchmark ===")
    server_proc = launch_server()
    wait_for_port(50051)  # Ensure server is ready before starting clients
    pandas_time = run_clients(
        "pythonProject.FedAvg6.tests.covariance_pandas.client1",
        "pythonProject.FedAvg6.tests.covariance_pandas.client2"
    )
    terminate_server(server_proc)
    print(f"Pandas clients finished in {pandas_time:.2f} seconds\n")

    print("=== Grizzly Benchmark ===")
    server_proc = launch_server()
    wait_for_port(50051)  # Ensure server is ready before starting clients
    grizzly_time = run_clients(
        "pythonProject.FedAvg6.tests.covariance_grizzly.client1",
        "pythonProject.FedAvg6.tests.covariance_grizzly.client2"
    )
    terminate_server(server_proc)
    print(f"Grizzly clients finished in {grizzly_time:.2f} seconds\n")

if __name__ == "__main__":
    main()