# ============================================
# 📂 network/tcp_client.py
# --------------------------------------------
# Chức năng: Gửi packet qua TCP
# ============================================

import socket
import json
from datetime import datetime

class TCPClient:
    def __init__(self, buffer_size=8192):
        self.buffer_size = buffer_size

    def send_packet(self, packet_dict: dict, target_ip: str, target_port: int, timeout=5):
        """Gửi packet TCP đến server (trạm mặt đất hoặc node)."""
        try:
            start_time = datetime.utcnow()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((target_ip, target_port))
                s.sendall(json.dumps(packet_dict).encode("utf-8"))

                response_data = s.recv(self.buffer_size)
                end_time = datetime.utcnow()

                latency_ms = (end_time - start_time).total_seconds() * 1000
                response = json.loads(response_data.decode("utf-8"))

                return {
                    "ack": response,
                    "latency_ms": latency_ms,
                    "status": "SUCCESS"
                }

        except Exception as e:
            return {
                "ack": None,
                "latency_ms": None,
                "status": f"ERROR: {e}"
            }
