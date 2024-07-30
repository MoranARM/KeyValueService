import sys

if len(sys.argv) != 6:
    print("Usage: python3 gen_benchmark_cmds.py <protocol.py mode> <replication_factor> <num_servers> <server_base_port> <num_clients>")
    sys.exit(1)

string = sys.argv[1]
replication_factor = int(sys.argv[2])
num_servers = int(sys.argv[3])
server_base_port = int(sys.argv[4])
num_clients = int(sys.argv[5])

base_address = "127.0.0.1"
# For each client, we need to know the addresses of all servers
server_addresses = []

# Generate commands for each server
for i in range(1, num_servers + 1):
    server_port = f"{server_base_port + i - 1}"
    print(f"python3 protocol.py dist_server {base_address} {server_port}")
    server_addresses.append(f"{base_address}:{server_port}")

# Sleep for 10 seconds to allow servers to start
print(f"sleep 10")

# Generate commands for each client
for j in range(1, num_clients + 1):
    print(f"python3 protocol.py {string} {replication_factor} {' '.join(server_addresses)}")

# Generate commands for shutdown
for i in range(1, num_servers + 1):
    server_port = f"{server_base_port + i - 1}"
    print(f"python3 shutdown_client.py {base_address} {server_port}")

# Example usage based on the fibonacci sequence:
# python3 gen_benchmark_cmds.py dist_client_rw 3 3 12345 1
# python3 gen_benchmark_cmds.py dist_client_rw 3 5 12345 1
# python3 gen_benchmark_cmds.py dist_client_rw 3 8 12345 1
# python3 gen_benchmark_cmds.py dist_client_rw 5 5 12345 1
# python3 gen_benchmark_cmds.py dist_client_rw 5 8 12345 1
# python3 gen_benchmark_cmds.py dist_client_rw 8 8 12345 1

# python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 2
# python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 3
# python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 5
# python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 8