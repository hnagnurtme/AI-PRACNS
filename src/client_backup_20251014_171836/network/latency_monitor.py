# ============================================
# 📂 network/latency_monitor.py
# --------------------------------------------
# Chức năng: Theo dõi và lưu độ trễ để vẽ biểu đồ
# ============================================

import json
from datetime import datetime
from pathlib import Path

class LatencyMonitor:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def log_latency(self, packet_id: str, latency_ms: float, algorithm: str):
        """Ghi log độ trễ từng gói theo thuật toán (RL hoặc Dijkstra)."""
        log_file = self.log_dir / f"latency_{algorithm.lower()}.json"
        entry = {
            "packetId": packet_id,
            "latencyMs": latency_ms,
            "timestamp": datetime.utcnow().isoformat()
        }

        logs = []
        if log_file.exists():
            with open(log_file, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []

        logs.append(entry)

        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"[LOG] {algorithm}: {packet_id} = {latency_ms:.2f} ms")
