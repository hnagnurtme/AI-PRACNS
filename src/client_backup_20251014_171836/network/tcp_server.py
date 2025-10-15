# ============================================
# 📂 network/tcp_server.py
# --------------------------------------------
# Chức năng: Nhận packet TCP và phản hồi ACK
# ============================================

import socket
import json
import threading
from datetime import datetime
from typing import Optional

from .latency_monitor import LatencyMonitor


class TCPServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, handler=None, buffer_size: int = 8192,
                 latency_monitor: Optional[LatencyMonitor] = None):
        """TCP server that receives JSON-serialized packets and returns ACKs.

        Behavior:
        - Parses incoming JSON packet (expected keys: packetId, timestamp, isUseRL)
        - Calls optional handler(packet)
        - Computes end-to-end latency using packet['timestamp'] (ISO8601 expected)
        - Logs latency to provided LatencyMonitor (if present) using algorithm 'RL' or 'Dijkstra'
        - Sends back a JSON ACK containing ackFor and status
        """
        self.host = host
        self.port = port
        self.handler = handler
        self.buffer_size = buffer_size
        self.latency_monitor = latency_monitor
        self._serv_sock: Optional[socket.socket] = None

    def start(self):
        """Start listening and spawn a thread per connection."""
        print(f"[SERVER] Listening on {self.host}:{self.port} ...")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen()
                self._serv_sock = s

                while True:
                    conn, addr = s.accept()
                    thread = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                    thread.start()
        except KeyboardInterrupt:
            print("[SERVER] Shutting down (KeyboardInterrupt)")
        except Exception as e:
            print(f"[SERVER ERROR] {e}")

    def _handle_client(self, conn: socket.socket, addr):
        """Handle a single client connection."""
        try:
            raw = b""
            # read once (suitable for small packets). If the client sends bigger payloads, this should
            # be updated to loop until EOF or a framing protocol is used.
            raw = conn.recv(self.buffer_size)
            if not raw:
                return

            try:
                packet = json.loads(raw.decode("utf-8"))
            except Exception:
                print(f"[SERVER] Failed to decode JSON from {addr}")
                return

            packet_id = packet.get("packetId") or packet.get("id")
            print(f"[SERVER] Received from {addr}: {packet_id}")

            # Call user-defined handler (if present) to process packet
            if self.handler:
                try:
                    self.handler(packet)
                except Exception as e:
                    print(f"[HANDLER ERROR] {e}")

            # Try compute latency if packet contains timestamp
            sent_ts = packet.get("timestamp")
            latency_ms = None
            algorithm = "unknown"
            try:
                if sent_ts:
                    # Expect ISO format string
                    sent_dt = datetime.fromisoformat(sent_ts)
                    now = datetime.utcnow()
                    latency_ms = (now - sent_dt).total_seconds() * 1000

                # Determine algorithm
                is_use_rl = packet.get("isUseRL")
                if isinstance(is_use_rl, bool):
                    algorithm = "RL" if is_use_rl else "Dijkstra"
                else:
                    # Accept 'rl'/'dijkstra' strings too
                    alg_raw = str(packet.get("algorithm") or packet.get("algo") or "").lower()
                    if "rl" in alg_raw:
                        algorithm = "RL"
                    elif "dijkstra" in alg_raw:
                        algorithm = "Dijkstra"

            except Exception:
                latency_ms = None

            # Log latency if monitor provided and we could compute it
            if self.latency_monitor and latency_ms is not None and packet_id:
                try:
                    self.latency_monitor.log_latency(packet_id, latency_ms, algorithm)
                except Exception as e:
                    print(f"[LATENCY MONITOR ERROR] {e}")

            ack_packet = {
                "ackFor": packet_id,
                "status": "RECEIVED",
                "timestamp": datetime.utcnow().isoformat(),
                "latencyMs": latency_ms,
                "algorithm": algorithm
            }

            conn.sendall(json.dumps(ack_packet).encode("utf-8"))

        except Exception as e:
            print(f"[SERVER ERROR] {e}")

        finally:
            try:
                conn.close()
            except Exception:
                pass
