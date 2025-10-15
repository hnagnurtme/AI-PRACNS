import threading
import time
import json
from network.tcp_server import TCPServer
from network.tcp_client import TCPClient
from network.latency_monitor import LatencyMonitor


def simple_handler(packet):
    print("Handler processed packet:", packet.get("packetId"))


def run_server():
    monitor = LatencyMonitor(log_dir="/tmp/sagsin_logs")
    server = TCPServer(host="127.0.0.1", port=9010, handler=simple_handler, latency_monitor=monitor)
    server.start()


if __name__ == "__main__":
    # start server in a background thread
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(0.5)

    client = TCPClient()
    pkt = {
        "packetId": "TEST-001",
        "source": "USER_A",
        "dest": "USER_B",
        "timestamp": "2025-10-14T00:00:00",
        "isUseRL": True
    }

    res = client.send_packet(pkt, "127.0.0.1", 9010)
    print("Client result:", res)
    time.sleep(0.5)
    print("Test complete")
