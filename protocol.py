import matplotlib.pyplot as plt
from btree.btree_memory import Node, BTree
from enum import IntEnum
import random
import struct
import socket
import time
import sys

protocol_requirements = """
The protocol for communication with the server follows a specific byte layout. Here is a brief description of the byte layout:

1. **Operation Code (1 byte):** Indicates the type of operation to be performed (0 for PUT, 1 for GET, or 2 CONTAINS).
2. **Key Data Type (1 Byte):** Specifies the data type of the key, 0 for float, 1 for int, and 2 for string
3. **Key Length (1 byte):** Specifies the length of the key in bytes.
4. **Key (variable length):** The actual key data.
5. **Value Data Type (1 Byte):** Specifies the data type of the key, 0 for float, 1 for int, and 2 for string, no byte if value length is zero
6. **Value Length (1 byte):** Specifies the length of the value in bytes.
7. **Value (variable length):** The actual value data.

Please note that the byte layout described above is a high-level overview. The actual implementation is slightly more efficient by only including the value when needed. I.e. if the value length is set to indicate zero bytes, such as is the case for a GET and CONTAINS operation, no additional bytes will be read. By that same logic calling PUT with zero for the value length is not supported as that would attempt to insert a null value.
"""

return_protocol_requirements = """
Return Byte Layout:
1. **Operation Code (1 byte):** Indicates the type of operation that was performed (0 for PUT, 1 for GET, or 2 CONTAINS).
2. **Value Data Type (1 Byte):** Specifies the data type of the value, 0 for float, 1 for int, 2 for string, 3 for boolean, and -1 for None
3. **Value Length (1 byte):** 
    * Specifies the length of the value in bytes if it was a successful PUT operations with a prior value or successful GET operation. 
    * If it was a contains operation the boolean will be stored in the value byte for whether the value was contained and this byte will hold a length of 1.
    * If the put operation did not have an old value or the get operation could not find the key then a null byte will be here to indicate a null.
4. **Value (variable length):** The actual value data in the case of a successful PUT operation with a prior value or successful GET operation, or the boolean value for whether the key was contained.
"""

SUPPORTED_TYPES = [float, int, str, bool, None]  # float, int, str are fully supported


class ProtocolOperationCode(IntEnum):
    PUT = 0
    GET = 1
    CONTAINS = 2


class ProtocolDataType(IntEnum):
    FLOAT = SUPPORTED_TYPES.index(float)
    INT = SUPPORTED_TYPES.index(int)
    STRING = SUPPORTED_TYPES.index(str)
    BOOLEAN = SUPPORTED_TYPES.index(bool)  # Only used in the CONTAINS response
    NONE = SUPPORTED_TYPES.index(None)


BTREE = BTree(15)  # Initialize the Database


def _to_bytes_type(data, data_type: ProtocolDataType) -> bytes:
    if data_type == ProtocolDataType.FLOAT:
        return struct.pack(">f", SUPPORTED_TYPES[data_type](data))  # Big endian float
    elif data_type == ProtocolDataType.INT:
        return struct.pack(">i", SUPPORTED_TYPES[data_type](data))  # Big endian int
    elif data_type == ProtocolDataType.STRING:
        return data.encode("utf-8")
    else:
        raise ValueError("Unsupported key data type")


def _from_bytes_type(data: bytes, data_type: ProtocolDataType):
    if data_type == ProtocolDataType.FLOAT:
        return struct.unpack(">f", data)[0]
    elif data_type == ProtocolDataType.INT:
        return struct.unpack(">i", data)[0]
    elif data_type == ProtocolDataType.STRING:
        return data.decode("utf-8")
    else:
        raise ValueError("Unsupported key data type")


def to_bytes(
    operation_code: int, key, key_data_type: int, value=None, value_data_type: int = -1
):
    """
    Converts operation details into bytes following the updated protocol layout, including data types for key and value.
    @param operation_code (int): holds the enum representation of the operation code: (0 for PUT, 1 for GET, or 2 CONTAINS).
    @param key: The key to be used in the operation.
    @param key_data_type (int): The data type of the key represented as a single byte.
    @param value (optional): The value to be inserted. Relevant only for PUT operations. Defaults to None.
    @param value_data_type (int, optional): The data type of the value represented as a single byte. Relevant only for PUT operations. Defaults to -1.
    @return bytes: The byte representation of the operation details.
    """
    # Convert operation code to bytes
    op_code_byte = operation_code.to_bytes(1, byteorder="big")

    # Convert key to bytes, get its length, and convert data type to bytes
    key_data_type_byte = key_data_type.to_bytes(1, byteorder="big")
    key_bytes = _to_bytes_type(key, ProtocolDataType(key_data_type))
    key_length = len(key_bytes)
    key_length_bytes = key_length.to_bytes(1, byteorder="big")

    # Initialize value bytes and data type to default to an empty string
    value_bytes = b""
    value_length_bytes = (0).to_bytes(1, byteorder="big")
    value_data_type_byte = (ProtocolDataType.STRING).to_bytes(1, byteorder="big")

    # If value is provided, convert it to bytes, get its length, and convert data type to bytes
    if value != None and value_data_type != -1:
        value_data_type_byte = value_data_type.to_bytes(1, byteorder="big")
        value_bytes = _to_bytes_type(value, ProtocolDataType(value_data_type))
        value_length = len(value_bytes)
        value_length_bytes = value_length.to_bytes(1, byteorder="big")

    # Combine all parts into the final byte sequence
    return (
        op_code_byte
        + key_data_type_byte
        + key_length_bytes
        + key_bytes
        + value_data_type_byte
        + value_length_bytes
        + value_bytes
    )


def from_bytes(data: bytes) -> tuple:
    """
    Parses bytes into operation details following the updated protocol layout, including data types for key and value.
    @param data (bytes): The byte representation of operation details.

    @return A tuple containing the operation code (int), key, key data type (int), value (str or None), and value data type (int).
    """
    # Extract operation code
    operation_code = ProtocolDataType(int.from_bytes(data[0:1], byteorder="big"))

    # Extract key length, key, and key data type
    key_data_type = ProtocolDataType(int.from_bytes(data[1:2], byteorder="big"))
    key_length = int.from_bytes(data[2:3], byteorder="big")
    if key_length < 1:
        if key_data_type != ProtocolDataType.STRING:
            raise ValueError(
                f"Key has a length of {key_length} which is not a valid number of bytes and cannot be parsed"
            )
        key = ""  # An empty string has no length but may be desired
    else:
        key_bytes = data[3 : 3 + key_length]
        key = _from_bytes_type(key_bytes, key_data_type)

    # Extract data type and value length
    value_data_type_byte = data[3 + key_length : 3 + key_length + 1]
    value_length_bytes = data[3 + key_length + 1 : 3 + key_length + 2]
    value_length = int.from_bytes(value_length_bytes, byteorder="big")

    # Initialize value and its data type
    value = None
    value_data_type = -1

    # If value length is not zero, extract value and its data type
    if value_length > 0:
        value_data_type = ProtocolDataType(
            int.from_bytes(value_data_type_byte, byteorder="big")
        )
        value_bytes = data[3 + key_length + 2 : 3 + key_length + 2 + value_length]
        value = _from_bytes_type(value_bytes, value_data_type)

    return (operation_code, key, key_data_type, value, value_data_type)


def get_protocol_data_type(value):
    """
    Determines the data type of a value and returns the corresponding ProtocolDataType enum.
    @param value: The value to determine the data type for.
    @return ProtocolDataType: The ProtocolDataType enum for the value.
    """
    data_type = type(value)
    if data_type == float:
        return ProtocolDataType.FLOAT
    elif data_type == int:
        return ProtocolDataType.INT
    elif data_type == str:
        return ProtocolDataType.STRING
    elif data_type == bool:
        return ProtocolDataType.BOOLEAN
    elif data_type == type(None):
        return ProtocolDataType.NONE  #  None is only used in responses
    else:
        raise ValueError("Unsupported data type")


def insert_and_search_example():
    b_tree = BTree(3)
    print(f"Order {b_tree.t}")

    for i in range(20):
        b_tree.insert(i, i * 2)
    for i in range(20):
        b_tree.insert(i, i * 3)

    b_tree.print_tree(b_tree.pages[b_tree.root_index])
    print()

    keys_to_search_for = [1, 2, 3, 5, 8, 13, 21]
    for key in keys_to_search_for:
        if b_tree.search(key) is not None:
            print(f"{key} is in the tree")
        else:
            print(f"{key} is NOT in the tree")

    print(f"Next Index: {b_tree.next_index}")
    for k, v in b_tree.pages.items():
        print(
            f"Index: {k}, Children: {v.children_indices}, Keys: {v.keys}, Values: {v.values}"
        )

    print(f"B.root.index: {b_tree.root_index}")


server_requirements = """
Connections should be maintained until the client disconnects (instead of establishing a new connection for each operation)

There will be two versions, first the minimal to test functionality then the normal implementation

Minimal version includes the following:
The server supports a single connection at a time.
No I/O multiplexing or parallel processing is being done
Benchmarking is done to determine the amount of time required to complement 10, 100, 1,000, 10,000, 100,000, 1,000,000 read operations with random keys. 

Normal version includes the following:
Implemented using multiplexing I/O such as select() or asycio in python, to support multiple concurrent connections
The server will be blocking I/O such that only one request can be handled at a time
WHen waiting on a client to inititate a request, the server can serve requests from other clients
There is no need to worry about concurrent modifications since no parallel processing is involved
Benchmarking is done to determine the impact of concurrent conncetions
Benchmarking also evalutes the time required for a client to complete 10,000 read operation with random keys, then performs this same experiemnt with 5, 10, 20, 50, and 100 additional clients in the background issuing read requests for random keys
matplotlib is used to plot the run time of the main client against the number of additional clients.
"""


# Function to initialize the server
def initialize_server(address: str = "127.0.0.1", port: int = 12345):
    """
    Initializes the server to listen on the specified port.
    @param address: where the server will be located, defaults to localhost
    @param port: The port number on which the server will listen for incoming connections.
    @return the socket connection
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((address, port))
    server_socket.listen(1)
    print(f"Server listening on port {port}")
    return server_socket


# Function to handle client connections for the minimal version
def handle_client_minimal(connection):
    """
    Handles a single client connection for the minimal version of the server.
    @param connection: The client connection object.
    """
    try:
        while True:
            request = read_request(connection)
            if not request:
                break  # Client closed connection
            operation_code, response = process_request(request)
            byte_response = generate_return_response(
                operation_code, response, get_protocol_data_type(response)
            )
            send_response(connection, byte_response)
    except KeyboardInterrupt:  # Useful for exiting the server when testing locally
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        close_connection(connection)


# Function to perform benchmarking for the minimal version
def benchmark_minimal():
    """
    Performs benchmarking for the minimal version of the server to determine the time required to complete a series of read operations.
    """
    port = 12345
    server_socket = initialize_server(port)
    print("Starting benchmark for minimal version...")
    start_time = time.time()

    # Simulate read operations with random keys
    for _ in range(10000):  # Adjust the range for different benchmarks
        key = random.randint(1, 10000)
        # Simulate a GET operation
        request = to_bytes(ProtocolOperationCode.GET, key, ProtocolDataType.INT)
        # Normally, you would send this request to the server and measure the response time
        # For this example, we'll just simulate a delay
        time.sleep(0.0001)  # Simulate network and processing delay

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time} seconds")

    # Close the server socket
    server_socket.close()


# Function to initialize the server with multiplexing for the normal version
def initialize_server_multiplexing(port: int):
    """
    Initializes the server with I/O multiplexing to listen on the specified port and handle multiple concurrent connections.
    @param port: The port number on which the server will listen for incoming connections.
    """


# Function to handle client connections for the normal version
def handle_client_normal(connection):
    """
    Handles client connections for the normal version of the server using multiplexing I/O.
    @param connection: The client connection object.
    """


# Function to perform benchmarking for the normal version with concurrent connections
def benchmark_normal_concurrent_connections():
    """
    Performs benchmarking for the normal version of the server to determine the impact of concurrent connections on the time required for a client to complete read operations.
    """


# Function to plot benchmark results using matplotlib
def plot_benchmark_results(results):
    """
    Plots the benchmark results using matplotlib to visualize the run time of the main client against the number of additional clients.
    @param results: The benchmark results to be plotted.
    """


# Function to accept connections
def accept_connections(server_socket):
    """
    Accepts incoming client connections and handles them according to the server version (minimal or normal).
    @param server_socket: The server socket object listening for incoming connections.
    """


# Function to read request from client
def read_request(client_socket) -> bytes:
    """
    Reads a request from the client socket and parses it according to the protocol.
    @param client_socket: The client socket from which the request is read.
    """
    try:
        # Assuming the first byte contains the operation code, followed by the key data type (1 byte) and length (1 byte)
        header = client_socket.recv(3)
        if not header:
            return None  # No data received, client may have closed connection
        # Further processing to parse the full request based on the protocol
        key_length = int.from_bytes(header[2:3], byteorder="big")
        # read in as many bytes as the key_length specifies
        key_bytes = client_socket.recv(key_length) if key_length > 0 else b""
        val_header = client_socket.recv(2)
        val_length = int.from_bytes(val_header[1:2], byteorder="big")
        val_bytes = client_socket.recv(val_length) if val_length > 0 else b""
        return header + key_bytes + val_header + val_bytes
    except Exception as e:
        print(f"Error reading request: {e}")
        return None


# Function to process request
def process_request(request, verbose=False):
    """
    Processes the client request and performs the corresponding operation (PUT, GET, CONTAINS) on the BTree database.
    @param request: The parsed client request.
    """
    if verbose:
        print(f"Processing request: {request}")
    operation_code, key, key_data_type, value, value_data_type = from_bytes(request)

    if operation_code == ProtocolOperationCode.PUT:
        return operation_code, BTREE.insert(key, value)
    elif operation_code == ProtocolOperationCode.GET:
        search_result = BTREE.search(key)
        # Return the operation code and the value associated with the key
        return operation_code, (
            search_result[0].values[search_result[1]]
            if search_result is not None
            else None
        )
    elif operation_code == ProtocolOperationCode.CONTAINS:
        return operation_code, BTREE.search(key) != None
    else:
        raise NotImplementedError(
            "Only the PUT (0) GET (1) and CONTAINS (2) operations are supported by the protocol"
        )


# Function to send response to client
def send_response(client_socket, response):
    """
    Sends a response to the client socket based on the outcome of the request processing.
    @param client_socket: The client socket to which the response is sent.
    @param response: The response bytes to be sent to the client.
    """
    try:
        client_socket.sendall(response)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Error sending response: {e}")


# Function to close connection
def close_connection(client_socket):
    """
    Closes the client connection after the request has been processed.
    @param client_socket: The client socket to be closed.
    """
    try:
        client_socket.close()
    except Exception as e:
        print(f"Error closing connection: {e}")


# Populate the database
def fill_DB(lower, upper, mult=2):
    """
    Populates the database with example values to be read during benchmarking
    @param lower: the lower bound of the keys to insert
    @param upper: the upper bound of the keys to insert
    @param mult:
    """
    for i in range(lower, upper):
        BTREE.insert(i, i * mult)


def generate_return_response(operation_code: int, value, value_data_type: int):
    """
    Generates a return response based on the return byte layout defined in the return_protocol_requirements.
    @param operation_code: The operation code indicating the type of operation that was performed (0 for PUT, 1 for GET, or 2 for CONTAINS).
    @param value: The value to be included in the response.
    @param value_data_type: The data type of the value (0 for float, 1 for int, 2 for string, 3 for boolean, and -1 for None).
    @return: The byte representation of the return response.
    """
    # Convert operation code to bytes
    op_code_byte = operation_code.to_bytes(1, byteorder="big")

    # Convert value data type to bytes
    value_data_type_byte = value_data_type.to_bytes(1, byteorder="big")

    # If value is None, set value length to 0 and return response
    if value is None:
        value_length_byte = (0).to_bytes(1, byteorder="big")
        return op_code_byte + value_data_type_byte + value_length_byte

    # Convert value to bytes
    if value_data_type == ProtocolDataType.FLOAT:
        value_bytes = struct.pack(">f", value)  # Big endian float
    elif value_data_type == ProtocolDataType.INT:
        value_bytes = struct.pack(">i", value)  # Big endian int
    elif value_data_type == ProtocolDataType.STRING:
        value_bytes = value.encode("utf-8")
    elif value_data_type == ProtocolDataType.BOOLEAN:
        value_bytes = int(value).to_bytes(1, byteorder="big")
    else:
        raise ValueError("Unsupported value data type")

    # Get value length and convert to bytes
    value_length = len(value_bytes)
    value_length_byte = value_length.to_bytes(1, byteorder="big")

    # Combine all parts into the final byte sequence
    return op_code_byte + value_data_type_byte + value_length_byte + value_bytes


# Function to read request from client
def read_response(client_socket) -> bytes:
    """
    Reads a request from the client socket and parses it according to the protocol.
    @param client_socket: The client socket from which the response is read.
    """
    try:
        # Assuming the first byte contains the operation code, followed by the value data type (1 byte) and value length (1 byte)
        header = client_socket.recv(3)
        if not header:
            return None  # No data received, client may have closed connection
        # Further processing to parse the full response based on the protocol
        val_type = int.from_bytes(header[1:2], byteorder="big")
        if val_type == ProtocolDataType.NONE:
            return header
        op_code = int.from_bytes(header[0:1], byteorder="big")
        if op_code == ProtocolOperationCode.CONTAINS:
            bool_val_byte = client_socket.recv(1)
            return header + bool_val_byte
        val_len = int.from_bytes(header[2:3], byteorder="big")
        val_bytes = client_socket.recv(val_len)
        return header + val_bytes
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Error reading response: {e}")
        return None


def process_response(response: bytes, verbose=False) -> tuple:
    """
    Processes the client response and return the parsed value from the BTree database.
    @param response: The client response bytes.
    @return the parsed response values
    """
    if verbose:
        print(f"Processing response: {response}")
    operation_code = int.from_bytes(response[0:1], byteorder="big")
    val_type = int.from_bytes(response[1:2], byteorder="big")

    # Parse out null reponse for new put or empty get operations
    if val_type == ProtocolDataType.NONE:
        return operation_code, val_type, None

    # Parse out bool if it was a contains operation
    if operation_code == ProtocolOperationCode.CONTAINS:
        return operation_code, val_type, bool(response[3:4])

    # Parse out returning old put or successful get operations
    val_len = int.from_bytes(response[2:3], byteorder="big")
    if val_len > 0:
        value_data_type = ProtocolDataType(val_type)
        value = _from_bytes_type(response[3 : 3 + val_len], value_data_type)
    return operation_code, val_type, value


def server_main_minimal():
    # Populate data for reads
    fill_DB(0, 1_000_000)
    server_socket = initialize_server(address="127.0.0.1", port=12345)
    print("Server initialized. Waiting for connections...")

    while True:
        connection, address = server_socket.accept()
        print(f"Connection from {address}")
        handle_client_minimal(connection)


def client_main_minimal(plot_results=False, num_trials=1):
    address = "127.0.0.1"
    port = 12345

    if plot_results:
        op_read_times = []

    # Values from 10 to 1,000,000 increasing by a factor of 10 each time
    num_read_operations = [10**i for i in range(1, 7)]

    for read_ops in num_read_operations:
        trial_op_read_times = []
        for trial in range(num_trials):
            # Create a new client
            client_socket = socket.socket()
            client_socket.connect((address, port))
            start_time = time.time()

            for _ in range(read_ops):
                key = str(random.randint(0, 1000000))
                # Create read request
                req_bytes = to_bytes(ProtocolOperationCode.GET, key, ProtocolDataType.INT)
                client_socket.send(req_bytes)
                response_bytes = read_response(client_socket)
                operation_code, val_type, value = process_response(response_bytes)
                # value now holds the response value from the DB

            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time for trial {trial+1}/{num_trials} with {read_ops} read operations: {time_taken} seconds")
            client_socket.close()

            if plot_results:
                trial_op_read_times.append(time_taken)
        if plot_results:
            op_read_times.append(sum(trial_op_read_times) / num_trials)

    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(num_read_operations, op_read_times, marker="o", linestyle="-")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Read Operations")
        plt.ylabel("Run Times [s]")
        plt.title("Single Connection Run Times vs Read Operations")
        plt.grid(True, which="both", ls="--")

        # Save the figure as a PNG file
        plt.savefig("benchmark_results_minimal.png")


# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an argument for whether to run as a client or server.")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "server":
        server_main_minimal()  # Run this on the server terminal
    elif mode == "client":
        client_main_minimal(plot_results=True, num_trials=3)  # Run this on the client terminal
    else:
        print("Invalid mode. Please choose 'server' or 'client'.")
        sys.exit(1)
