from protocol import ProtocolOperationCode, to_bytes, read_response, process_response
import socket
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 shutdown_client.py <server_ip> <server_port>")
        sys.exit(1)

    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])

    # Create a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the server
        client_socket.connect((server_ip, server_port))

        # Send a SHUTDOWN operation to the server
        shutdown_message = to_bytes(ProtocolOperationCode.SHUTDOWN)
        client_socket.sendall(shutdown_message)

        # Receive confirmation message
        response = read_response(client_socket)
        protocol_op_code, val_type, confirmation_message = process_response(response)
        if confirmation_message != "Shutdown confirmed":
            print("Unexpected confirmation message:", confirmation_message)
        else:
            print("Server shutdown confirmed")

    except socket.error as e:
        print("Error occurred while connecting to the server:", e)
    finally:
        client_socket.close()
