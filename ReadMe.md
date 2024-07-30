# Distributed TCP/IP Server for a Key-Value Library

This ReadMe provides an overview of a Distributed TCP/IP server that handles PUT, GET, and CONTAINS operations for key-value pairs. The server supports data types such as strings, integers, and floats.

## Protocol Functionality

### PUT Operation
The PUT operation allows clients to store key-value pairs in the server's database. If a key already exists in the database, the PUT operation will return the old value associated with the key. If the key is not present, the PUT operation will return None.

### GET Operation
The GET operation retrieves the value associated with a given key from the server's database. If the key is absent, the GET operation will return None.

### CONTAINS Operation
The CONTAINS operation checks whether a given key exists in the server's database. It returns a boolean value indicating whether the key was found.

### SHUTDOWN Operation
The SHUTDOWN operation was made for the special administrative client. It performs a graceful shutdown of the server.

## Protocol Byte Layout

The protocol for communication with the server follows a specific byte layout. Here is a brief description of the byte layout:

1. **Operation Code (1 byte):** Indicates the type of operation to be performed (0 for PUT, 1 for GET, 2 for CONTAINS, or 3 for SHUTDOWN).
2. **Key Data Type (1 Byte):** Specifies the data type of the key, 0 for float, 1 for int, and 2 for string
3. **Key Length (1 byte):** Specifies the length of the key in bytes.
4. **Key (variable length):** The actual key data.
5. **Value Data Type (1 Byte):** Specifies the data type of the key, 0 for float, 1 for int, and 2 for string, no byte if value length is zero
6. **Value Length (1 byte):** Specifies the length of the value in bytes.
7. **Value (variable length):** The actual value data.


Please note that the byte layout described above is a high-level overview. The actual implementation is slightly more efficient by only including the value when needed. I.e. if the value length is set to indicate zero bytes, such as is the case for a GET and CONTAINS operation, no additional bytes will be read. By that same logic calling PUT with zero for the value length is not supported as that would attempt to insert a null value. If the desired behavior is to reset its value, either entering an empty string `""` or an integer zero `0` or a float zero `0.0` should suffice given that the database does not support deletions at this time. 

Additionally note that the protocol assumes a maximum of 64 byte keys and values due to the implementation of the B-Tree that will be used as the underlying architecture for the key-value store

### Return Byte Layout
1. **Operation Code (1 byte):** Indicates the type of operation that was performed (0 for PUT, 1 for GET, 2 for CONTAINS, or 3 for SHUTDOWN).
2. **Value Data Type (1 Byte):** Specifies the data type of the value, 0 for float, 1 for int, 2 for string, 3 for boolean, and -1 for None
3. **Value Length (1 byte):** 
    * Specifies the length of the value in bytes if it was a successful PUT operations with a prior value or successful GET operation. 
    * If it was a contains operation the boolean will be stored in the value byte for whether the value was contained and this byte will hold a length of 1.
    * If the put operation did not have an old value or the get operation could not find the key then a null byte will be here to indicate a null.
4. **Value (variable length):** The actual value data in the case of a successful PUT operation with a prior value or successful GET operation, or the boolean value for whether the key was contained.

## Server Functionality
### Running the server
```bash
python3 protocol.py server
```
### Running the client
```bash
python3 protocol.py client
```
### Running the Distributed Server
Requires passing in the address and port desired for the server to operate on
A separate command must be run for each desired server address and port pair
```bash
python3 protocol.py dist_server 127.0.0.1 12345
python3 protocol.py dist_server 127.0.0.1 12346
python3 protocol.py dist_server 127.0.0.1 12347
```

### Running the Distributed Client
requires the address and port of the servers available, even though only a fraction of these will be used based on the replication factor, although all will be used when the replication factor matches the number of servers like in this case where it is 3.
```bash
python3 protocol.py dist_client_rw 3 127.0.0.1:12345 127.0.0.1:12346 127.0.0.1:12347
```

### Shutting Down a Server
The special administrative client found in `shutdown_client.py` can be used to shutdown a server on a specific port py passing in the address and port
```bash
python3 shutdown_client.py 127.0.0.1 12345
```

### Generating Commands for Benchmarking
#### Example usage based on the fibonacci sequence:
```bash
python3 gen_benchmark_cmds.py dist_client_rw 3 3 12345 1
python3 gen_benchmark_cmds.py dist_client_rw 3 5 12345 1
python3 gen_benchmark_cmds.py dist_client_rw 3 8 12345 1
python3 gen_benchmark_cmds.py dist_client_rw 5 5 12345 1
python3 gen_benchmark_cmds.py dist_client_rw 5 8 12345 1
python3 gen_benchmark_cmds.py dist_client_rw 8 8 12345 1

python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 2
python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 3
python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 5
python3 gen_benchmark_cmds.py dist_client_r 3 8 12345 8
```
Running the above commands will generate individual commands for running the distributed clients, and the distributed servers, as well as shutting down the servers made. 
