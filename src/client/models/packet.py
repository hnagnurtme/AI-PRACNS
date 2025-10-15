import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import base64

# --- 1. Class lồng nhau: HopRecord (Chi tiết từng chặng) ---
@dataclass
class HopRecord:
    fromNodeId: str
    toNodeId: str
    latencyMs: float
    timestampMs: int

# --- 2. Class lồng nhau: ServiceQoS (Chất lượng Dịch vụ Yêu cầu) ---
@dataclass
class ServiceQoS:
    serviceType: str
    defaultPriority: int
    maxLatencyMs: float
    maxJitterMs: float
    minBandwidthMbps: float
    maxLossRate: float

# --- 3. Class Chính: Packet ---
@dataclass
class Packet:
    """
    Biểu diễn cấu trúc gói tin mạng P2P/Mesh dựa trên định dạng JSON.
    """
    # --- ID và Địa chỉ ---
    packetId: str
    sourceUserId: str
    destinationUserId: str
    stationSource: str = field(metadata={'json': 'stationSource'})
    stationDest: str = field(metadata={'json': 'stationDest'})

    # --- Trạng thái Thời gian và Phân loại ---
    type: str  # DATA, ACK
    acknowledgedPacketId: Optional[str]
    timeSentFromSourceMs: int

    # --- Dữ liệu Ứng dụng ---
    payloadDataBase64: str
    payloadSizeByte: int
    serviceType: str
    
    # Đối tượng QoS lồng nhau
    serviceQoS: ServiceQoS = field(metadata={'json': 'serviceQoS'})

    # --- Định tuyến và Theo dõi ---
    TTL: int
    currentHoldingNodeId: str
    nextHopNodeId: str
    pathHistory: List[str]
    
    # Mảng các bản ghi chặng (tùy chọn)
    hopRecords: List[HopRecord] = field(default_factory=list)

    accumulatedDelayMs: float = 0.0
    priorityLevel: int = 1
    isUseRL: bool = field(default=False, metadata={'json': 'isUseRL'})

    # --- QoS Yêu cầu Tối đa Chấp nhận được ---
    maxAcceptableLatencyMs: float = 150.0
    maxAcceptableLossRate: float = 0.01

    # --- Trạng thái Drop ---
    dropped: bool = False
    dropReason: Optional[str] = None


    # --- Phương thức hỗ trợ ---

    def to_json(self) -> str:
        """Chuyển đối tượng Packet thành chuỗi JSON."""
        # Dùng thư viện json để serialization
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_data: str) -> 'Packet':
        """Tạo đối tượng Packet từ chuỗi JSON."""
        data: Dict[str, Any] = json.loads(json_data)
        
        # Tái tạo các đối tượng lồng nhau
        data['serviceQoS'] = ServiceQoS(**data['serviceQoS'])
        data['hopRecords'] = [HopRecord(**hr) for hr in data.get('hopRecords', [])]
        
        return cls(**data)

    def get_decoded_payload(self) -> str:
        """Giải mã payload từ Base64 sang chuỗi (giả định là UTF-8)."""
        try:
            return base64.b64decode(self.payloadDataBase64).decode('utf-8')
        except Exception:
            return "[Lỗi giải mã Base64]"

# --- Ví dụ về cách sử dụng ---
if __name__ == '__main__':
    # JSON mẫu (sử dụng JSON trong câu hỏi của bạn)
    sample_json = """
    {
        "packetId": "PKT-001",
        "sourceUserId": "USER_A",
        "destinationUserId": "USER_B",
        "stationSource": "GS-01",
        "stationDest": "GS-A5",
        "type": "DATA",
        "acknowledgedPacketId": null,
        "timeSentFromSourceMs": 1739512300000,
        "payloadDataBase64": "UZ2FtcGxIIGRhdG EgYmFzZTY0", 
        "payloadSizeByte": 512,
        "serviceType": "VIDEO_STREAM",
        "serviceQoS": {
            "serviceType": "VIDEO_STREAM",
            "defaultPriority": 1,
            "maxLatencyMs": 150.0,
            "maxJitterMs": 30.0,
            "minBandwidthMbps": 5.0,
            "maxLossRate": 0.01
        },
        "TTL": 6,
        "currentHoldingNodeId": "ME0-002",
        "nextHopNodeId": "GS-01",
        "pathHistory": [
            "USER_A",
            "LEO-001",
            "ME0-002",
            "GS-01",
            "USER_B"
        ],
        "hopRecords": [
            { "fromNodeId": "USER_A", "toNodeId": "LEO-001", "latencyMs": 4.8, "timestampMs": 1739512304800 },
            { "fromNodeId": "LEO-001", "toNodeId": "ME0-002", "latencyMs": 7.9, "timestampMs": 1739512312700 },
            { "fromNodeId": "ME0-002", "toNodeId": "GS-01", "latencyMs": 3.2, "timestampMs": 1739512315900 },
            { "fromNodeId": "GS-01", "toNodeId": "USER_B", "latencyMs": 4.1, "timestampMs": 1739512320000 }
        ],
        "accumulatedDelayMs": 20.0,
        "priorityLevel": 1,
        "isUseRL": true,
        "maxAcceptableLatencyMs": 150.0,
        "maxAcceptableLossRate": 0.01,
        "dropped": false,
        "dropReason": null
    }
    """
    
    # 1. Khởi tạo từ JSON (Deserialization)
    packet_obj = Packet.from_json(sample_json)
    
    print("--- Khởi tạo thành công từ JSON ---")
    print(f"Packet ID: {packet_obj.packetId}")
    print(f"Nguồn: {packet_obj.sourceUserId} (Station: {packet_obj.stationSource})")
    print(f"Đích: {packet_obj.destinationUserId} (Station: {packet_obj.stationDest})")
    print(f"TTL còn lại: {packet_obj.TTL}")
    print(f"Total Hops: {len(packet_obj.hopRecords)}")
    print(f"Jitter QoS: {packet_obj.serviceQoS.maxJitterMs} ms")
    print(f"Payload giải mã: {packet_obj.get_decoded_payload()}") # UZ2FtcGxIIGRhdG EgYmFzZTY0 -> Ví dụ base64

    # 2. Chuyển ngược lại thành JSON (Serialization)
    print("\n--- Chuyển ngược lại thành JSON ---")
    json_output = packet_obj.to_json()
    print(json_output[:400] + "...")