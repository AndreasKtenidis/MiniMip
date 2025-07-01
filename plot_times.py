import subprocess
import time
import os
import signal
import matplotlib.pyplot as plt
import numpy as np

DATA_GEN_SCRIPT = "pythonProject/FedAvg6/tests/covariance_grizzly/db_setup.py"
PANDAS_CLIENTS = [
    "pythonProject.FedAvg6.tests.covariance_pandas.client1",
    "pythonProject.FedAvg6.tests.covariance_pandas.client2"
]
GRIZZLY_CLIENTS = [
    "pythonProject.FedAvg6.tests.covariance_grizzly.client1",
    "pythonProject.FedAvg6.tests.covariance_grizzly.client2"
]
SERVER_SCRIPT = "pythonProject.FedAvg6.system.server.grpc_agg_server"
CSV_PATH = "pythonProject/FedAvg6/data/grizzly_pandas_test/covariance_100mb.csv"
DUCKDB_PATH = "pythonProject/FedAvg6/data/grizzly_pandas_test/covariance_client1.duckdb"

def launch_server():
    return subprocess.Popen(
        ["python", "-m", SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

def wait_for_port(port, host='127.0.0.1', timeout=10.0):
    import socket
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

def parse_client_times(logfile):
    # Expects lines like: "Time taken for covariance computation on client 0: 1.23 seconds"
    times = {'Covariance': [], 'Pearson Correlation': [], 'Least Squares Regression': []}
    global_time = None
    with open(logfile) as f:
        for line in f:
            if "Time taken for covariance computation" in line:
                times['Covariance'].append(float(line.split(":")[-1].split()[0]))
            elif "Time taken for Pearson correlation computation" in line:
                times['Pearson Correlation'].append(float(line.split(":")[-1].split()[0]))
            elif "Time taken for least squares regression computation" in line:
                times['Least Squares Regression'].append(float(line.split(":")[-1].split()[0]))
            elif "Time taken with data loading, for all three algorithms" in line:
                global_time = float(line.split(":")[-1].split()[0])
    # Return max time per algorithm (since both clients run in parallel)
    result = {k: max(v) if v else 0 for k, v in times.items()}
    return result, global_time

def run_and_time(method, clients):
    # Remove previous logs
    for i, c in enumerate(clients):
        log = f"client_{method}_{i}.log"
        if os.path.exists(log):
            os.remove(log)
    # Launch server
    server_proc = launch_server()
    wait_for_port(50051)
    # Run clients and capture output
    procs = []
    for i, c in enumerate(clients):
        log = open(f"client_{method}_{i}.log", "w")
        p = subprocess.Popen(["python", "-m", c], stdout=log, stderr=log)
        procs.append((p, log))
    for p, log in procs:
        p.wait()
        log.close()
    terminate_server(server_proc)
    # Parse logs for algorithm times and global time
    alg_times = []
    global_times = []
    for i in range(2):
        alg_time, global_time = parse_client_times(f"client_{method}_{i}.log")
        alg_times.append(alg_time)
        global_times.append(global_time)
    # For each algorithm, take the max time across clients (since they run in parallel)
    result = {}
    for alg in ['Covariance', 'Pearson Correlation', 'Least Squares Regression']:
        result[alg] = max(alg_times[0][alg], alg_times[1][alg])
    # For global time, take the max (since both clients run in parallel)
    total_time = max(global_times)
    return result, total_time

def ensure_data():
    # Generate data if not present
    if not (os.path.exists(CSV_PATH) and os.path.exists(DUCKDB_PATH)):
        print("Generating data...")
        subprocess.run(["python", DATA_GEN_SCRIPT, "--num-clients", "2", "--size-mb", "500"], check=True)

def main():
    ensure_data()
    print("Running Pandas clients (with loading)...")
    pandas_alg_times, pandas_total = run_and_time("pandas", PANDAS_CLIENTS)
    print("Running Grizzly clients (with loading)...")
    grizzly_alg_times, grizzly_total = run_and_time("grizzly", GRIZZLY_CLIENTS)

    # Plot 1: Histogram of per-algorithm times
    labels = ['Covariance', 'Pearson Correlation', 'Least Squares Regression']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    pandas_times = [pandas_alg_times[l] for l in labels]
    grizzly_times = [grizzly_alg_times[l] for l in labels]
    rects1 = ax.bar(x - width/2, pandas_times, width, label='Pandas', color='tab:blue')
    rects2 = ax.bar(x + width/2, grizzly_times, width, label='Grizzly', color='tab:red')

    ax.set_ylabel('Time (s)')
    ax.set_title('Algorithm run times (including data loading)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")  # Move legend to top left
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    # Add extra space above the tallest bar
    max_height = max(pandas_times + grizzly_times)
    ax.set_ylim(0, max_height * 1.15)

    # Attach full-precision time labels
    for rect, t in zip(rects1, pandas_times):
        ax.annotate(f'{t:.6f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for rect, t in zip(rects2, grizzly_times):
        ax.annotate(f'{t:.6f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("federated_alg_times.png")
    plt.show()

    # Plot 2: Histogram of total time (with loading)
    fig, ax = plt.subplots()
    total_times = [pandas_total, grizzly_total]
    bars = ax.bar(['Pandas (total)', 'Grizzly (total)'], total_times, width=0.5, color=['tab:blue', 'tab:red'])
    ax.set_ylabel('Time (s)')
    ax.set_title('Total time to run all algorithms (including data loading)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    # Add extra space above the tallest bar
    max_total = max(total_times)
    ax.set_ylim(0, max_total * 1.15)

    for rect, t in zip(bars, total_times):
        ax.annotate(f'{t:.6f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("federated_total_times.png")
    plt.show()

if __name__ == "__main__":
    main()