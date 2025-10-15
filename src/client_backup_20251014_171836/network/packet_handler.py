# ============================================
# 📂 network/packet_handler.py
# --------------------------------------------
# Chức năng: Xử lý logic khi nhận packet
# ============================================

from datetime import datetime
import random

class PacketHandler:
    def __init__(self, logger=None):
        self.logger = logger

    def process_packet(self, packet: dict):
        """Giả lập xử lý packet và log kết quả."""
        latency = random.uniform(3.0, 10.0)
        packet["processedTimestamp"] = datetime.utcnow().isoformat()
        packet["simulatedProcessingDelayMs"] = latency

        if self.logger:
            self.logger(packet)

        return packet
