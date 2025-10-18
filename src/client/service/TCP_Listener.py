import socket
import threading
import time
import json
import sys
import struct
import logging
from typing import Callable, Optional, Dict, Any, Union
from queue import Queue

# --- Configure logging ---
# This is better than print() for applications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)

class TCPListener(threading.Thread):
    """
    A robust TCP listener service that runs in a background thread.
    Uses a 4-byte Big-Endian length prefix for message framing.
    """

    def __init__(self, host: str, port: int, handler: Optional[Callable[[Dict], None]] = None, incoming_queue: Optional[Queue] = None):
        """
        Initializes the Listener.
        Args:
            host: The host to bind to (e.g., '0.0.0.0').
            port: The port to listen on.
            handler: (Optional) A callback function to process received messages.
            incoming_queue: (Optional) A queue to put received messages into.
        """
        super().__init__()
        if not handler and not incoming_queue:
            raise ValueError("Must provide at least a handler or an incoming_queue.")
            
        self.host = host
        self.port = port
        self.handler = handler
        self.incoming_queue = incoming_queue
        self.running = True
        self.listener_socket: Optional[socket.socket] = None
        self.name = f"TCPListener:{port}"
        self.daemon = True

    def stop(self):
        """Signals the listener thread to stop."""
        self.running = False
        logging.info("Stop request received.")
        # Closing the socket will cause accept() to raise an error, unblocking it.
        if self.listener_socket:
            self.listener_socket.close()

    def run(self):
        """The main loop of the listener thread."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.listener_socket:
                self.listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.listener_socket.bind((self.host, self.port))
                self.listener_socket.listen(5)
                logging.info(f"Started and listening on {self.host}:{self.port}")

                while self.running:
                    try:
                        # Using a timeout allows the loop to check self.running periodically
                        self.listener_socket.settimeout(1.0)
                        conn, addr = self.listener_socket.accept()
                        
                        # Spawn a new thread to handle the client connection
                        client_thread = threading.Thread(
                            target=self._handle_connection, 
                            args=(conn, addr), 
                            daemon=True,
                            name=f"ClientHandler:{addr[1]}"
                        )
                        client_thread.start()
                        
                    except socket.timeout:
                        continue # Normal, just checking the self.running flag
                    except OSError:
                        # This error is expected when self.listener_socket.close() is called in stop()
                        if self.running:
                           logging.exception("An unexpected OS error occurred.")
                        break # Exit loop if socket is closed
                        
        except Exception:
            if self.running:
                logging.exception("FATAL: Could not bind or listen on the specified port.")
        finally:
            logging.info("Stopped.")

    def _recvall(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Helper function to reliably receive n bytes."""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _handle_connection(self, conn: socket.socket, addr: tuple):
        """Handles a single client connection using a length-prefix protocol."""
        logging.info(f"Accepted connection from {addr}")
        try:
            # Use context manager to ensure the socket is always closed
            with conn:
                # 1. Receive the 4-byte header for message length
                header_bytes = self._recvall(conn, 4)
                if not header_bytes:
                    logging.warning(f"Connection closed by {addr} before header was sent.")
                    return

                # 2. Unpack the length (Big-Endian, Unsigned Integer)
                message_length = struct.unpack('>I', header_bytes)[0]
                
                # 3. Receive the JSON payload
                json_bytes = self._recvall(conn, message_length)
                if not json_bytes:
                    logging.warning(f"Connection closed by {addr} before full payload was sent.")
                    return
                
                # 4. Decode and process the message
                packet_dict = json.loads(json_bytes.decode('utf-8'))
                message = {"source_addr": addr, "packet_data": packet_dict}
                
                # Process the message
                if self.handler:
                    self.handler(message)
                if self.incoming_queue:
                    self.incoming_queue.put(message)

        except json.JSONDecodeError:
            logging.error(f"Invalid JSON received from {addr}.")
        except Exception:
            logging.exception(f"An error occurred while handling connection from {addr}.")

if __name__ == '__main__':
    # --- Standalone test block ---
    
    def simple_handler(received_data: Dict):
        addr = received_data['source_addr']
        packet = received_data['packet_data']
        payload_preview = packet.get('payloadDataBase64', '')[:30]
        
        logging.info(f"--- PACKET RECEIVED from {addr} ---")
        logging.info(f"  ID: {packet.get('packetId', 'N/A')}, Type: {packet.get('type', 'N/A')}")
        logging.info(f"  Payload Preview: {payload_preview}...")
        logging.info("------------------------------------")

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    try:
        LISTEN_PORT = int(sys.argv[1])
    except ValueError:
        print("Error: Port must be a number.")
        sys.exit(1)

    listener = TCPListener(host='0.0.0.0', port=LISTEN_PORT, handler=simple_handler)
    listener.start()
    
    print(f"\nListener is running on port {LISTEN_PORT}. Press Ctrl+C to stop.")
    try:
        # Keep the main thread alive to see logs
        listener.join()
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        listener.stop()
        # Wait for the thread to fully terminate
        listener.join()
        print("Program finished.")