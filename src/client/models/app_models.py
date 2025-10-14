# app_models.py
from typing import List, Optional, Any
from dataclasses import dataclass, field
import base64
import time

# --- 1.1. Packet Components ---
@dataclass
class ServiceQoS:
    maxLatencyMs: float = 150.0
    maxLossRate: float = 0.01
    defaultPriority: int = 1

@dataclass
class Packet:
    packetId: str
    sourceUserId: str
    destinationUserId: str
    type: str = "DATA"
    serviceQoS: ServiceQoS = field(default_factory=ServiceQoS)
    
    # Payload File
    payloadDataBase64: str = ""
    payloadFileName: Optional[str] = None 
    payloadSizeByte: int = 0
    
    # Các trường mạng/routing
    TTL: int = 10
    priorityLevel: int = 1
    isUseRL: bool = False 
    
    timeSentFromSourceMs: int = int(time.time() * 1000)
    pathHistory: List[str] = field(default_factory=list)
    acknowledgedPacketId: Optional[str] = None

    def get_decoded_payload_preview(self) -> str:
        """Giải mã một phần payload để xem trước."""
        try:
            # Chỉ giải mã 50 ký tự đầu tiên
            preview_bytes = base64.b64decode(self.payloadDataBase64[:68]) # 68 ký tự base64 ~ 50 bytes
            return preview_bytes.decode('utf-8', errors='ignore').strip() + "..."
        except Exception:
            return "[Binary/Non-Text Data]"

# --- 1.2. Node Components ---
@dataclass
class Communication:
    ipAddress: str = "127.0.0.1"
    port: int = 50001

@dataclass
class Node:
    nodeId: str
    nodeName: str
    communication: Communication = field(default_factory=Communication)