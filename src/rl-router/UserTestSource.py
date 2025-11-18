import socket
import time

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

def run_source():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")
        for i in range(5):
            message = f"Hello from source {i}"
            s.sendall(message.encode())
            print(f"Sent: {message}")
            time.sleep(1)
        print("Finished sending messages.")

if __name__ == '__main__':
    run_source()
