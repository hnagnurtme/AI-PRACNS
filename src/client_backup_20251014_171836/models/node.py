# ============================================
# 📂 models/node.py
# --------------------------------------------
# Định nghĩa cấu trúc Node (trạm hoặc vệ tinh)
# ============================================

from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any
from datetime import datetime


@dataclass
class Node:
    nodeId: str
    nodeName: str
    type: str  # GROUND_STATION | LEO_SATELLITE | MEO_SATELLITE | GEO_SATELLITE

    position: Dict[str, float]                 # {latitude, longitude, altitude}
    velocity: Optional[Dict[str, float]] = None
    orbit: Optional[Dict[str, float]] = None

    communication: Optional[Dict[str, Any]] = None
    status: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        """Chuyển object thành dict (dễ insert Mongo hoặc serialize JSON)."""
        return {
            "nodeId": self.nodeId,
            "nodeName": self.nodeName,
            "type": self.type,
            "position": self.position,
            "velocity": self.velocity or {},
            "orbit": self.orbit or {},
            "communication": self.communication or {},
            "status": self.status or {},
            "metadata": self.metadata or {}
        }

    @staticmethod
    def from_dict(data: dict) -> "Node":
        """Khởi tạo Node từ dict (lấy từ Mongo hoặc JSON)."""
        return Node(
            nodeId=data.get("nodeId", ""),
            nodeName=data.get("nodeName", ""),
            type=data.get("type", ""),
            position=data.get("position", {}),
            velocity=data.get("velocity", {}),
            orbit=data.get("orbit", {}),
            communication=data.get("communication", {}),
            status=data.get("status", {}),
            metadata=data.get("metadata", {})
        )

    def __repr__(self):
        return f"<Node {self.nodeId} ({self.type})>"


# ==================================================
# ⚡ Example usage
# ==================================================
if __name__ == "__main__":
    node = Node(
        nodeId="GS-01",
        nodeName="Ground Station Hanoi",
        type="GROUND_STATION",
        position={"latitude": 21.0285, "longitude": 105.8542, "altitude": 0.1},
        communication={
            "frequencyGHz": 14.25,
            "bandwidthMHz": 500,
            "ipAddress": "127.0.0.1",
            "port": 8080
        },
        status={"active": True, "lastUpdated": datetime.utcnow().isoformat()},
        metadata={"operator": "VNPT Space"}
    )

    print(node)
    print(node.to_dict())
