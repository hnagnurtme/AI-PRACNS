# ============================================
# 📂 models/packet.py
# --------------------------------------------
# Định nghĩa cấu trúc Packet (gói dữ liệu truyền trong mạng)
# ============================================

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


@dataclass
class Packet:
    packetId: str
    sourceUserId: str
    destinationUserId: str
    type: str  # DATA | ACK | CONTROL

    TTL: int
    isUseRL: bool

    serviceType: str  # VIDEO_STREAM | AUDIO | TEXT | CONTROL
    serviceQoS: Dict[str, Any]

    pathHistory: List[str] = field(default_factory=list)
    hopRecords: List[Dict[str, Any]] = field(default_factory=list)

    accumulatedDelayMs: float = 0.0
    priorityLevel: int = 1
    maxAcceptableLatencyMs: float = 150.0
    maxAcceptableLossRate: float = 0.01

    stationSource: Optional[str] = None
    stationDest: Optional[str] = None

    timestampCreated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # -------------------------------------------------------
    # 🧰 Utility methods
    # -------------------------------------------------------

    def add_hop(self, from_node: str, to_node: str, latency_ms: float):
        """Ghi nhận một hop mới (để vẽ đồ thị và log độ trễ)."""
        ts = int(datetime.utcnow().timestamp() * 1000)
        self.hopRecords.append({
            "fromNodeId": from_node,
            "toNodeId": to_node,
            "latencyMs": latency_ms,
            "timestampMs": ts
        })
        self.pathHistory.append(to_node)
        self.accumulatedDelayMs += latency_ms

    def to_dict(self) -> dict:
        """Chuyển packet thành dict JSON-ready."""
        return {
            "packetId": self.packetId,
            "sourceUserId": self.sourceUserId,
            "destinationUserId": self.destinationUserId,
            "type": self.type,
            "TTL": self.TTL,
            "isUseRL": self.isUseRL,
            "serviceType": self.serviceType,
            "serviceQoS": self.serviceQoS,
            "pathHistory": self.pathHistory,
            "hopRecords": self.hopRecords,
            "accumulatedDelayMs": self.accumulatedDelayMs,
            "priorityLevel": self.priorityLevel,
            "maxAcceptableLatencyMs": self.maxAcceptableLatencyMs,
            "maxAcceptableLossRate": self.maxAcceptableLossRate,
            "stationSource": self.stationSource,
            "stationDest": self.stationDest,
            "timestampCreated": self.timestampCreated
        }

    @staticmethod
    def from_dict(data: dict) -> "Packet":
        """Khởi tạo packet từ JSON hoặc Mongo."""
        return Packet(
            packetId=data.get("packetId", f"PKT-{uuid.uuid4().hex[:6].upper()}"),
            sourceUserId=data["sourceUserId"],
            destinationUserId=data["destinationUserId"],
            type=data.get("type", "DATA"),
            TTL=data.get("TTL", 6),
            isUseRL=data.get("isUseRL", True),
            serviceType=data.get("serviceType", "VIDEO_STREAM"),
            serviceQoS=data.get("serviceQoS", {}),
            pathHistory=data.get("pathHistory", []),
            hopRecords=data.get("hopRecords", []),
            accumulatedDelayMs=data.get("accumulatedDelayMs", 0.0),
            priorityLevel=data.get("priorityLevel", 1),
            maxAcceptableLatencyMs=data.get("maxAcceptableLatencyMs", 150.0),
            maxAcceptableLossRate=data.get("maxAcceptableLossRate", 0.01),
            stationSource=data.get("stationSource"),
            stationDest=data.get("stationDest"),
            timestampCreated=data.get("timestampCreated", datetime.utcnow().isoformat())
        )


# ==================================================
# ⚡ Example usage
# ==================================================
if __name__ == "__main__":
    packet = Packet(
        packetId="PKT-001",
        sourceUserId="USER_A",
        destinationUserId="USER_B",
        type="DATA",
        TTL=6,
        isUseRL=True,
        serviceType="VIDEO_STREAM",
        serviceQoS={
            "serviceType": "VIDEO_STREAM",
            "defaultPriority": 1,
            "maxLatencyMs": 150.0,
            "maxJitterMs": 30.0,
            "minBandwidthMbps": 5.0,
            "maxLossRate": 0.01
        },
        stationSource="GS-01",
        stationDest="GS-03"
    )

    packet.add_hop("USER_A", "LEO-001", 4.8)
    packet.add_hop("LEO-001", "MEO-002", 7.9)
    packet.add_hop("MEO-002", "GS-01", 3.2)
    print(packet.to_dict())
