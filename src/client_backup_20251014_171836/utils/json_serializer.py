# client_app/utils/json_serializer.py
import json
from models.packet import Packet
from dataclasses import asdict
from typing import Dict, List, Any

def packet_to_json(packet: Packet) -> str:
    """Chuyển đổi Packet dataclass thành chuỗi JSON, loại bỏ None."""
    data = asdict(packet)
    data = {k: v for k, v in data.items() if v is not None}
    return json.dumps(data)

def json_to_packet(json_str: str) -> Packet:
    """Chuyển đổi chuỗi JSON thành Packet dataclass."""
    data = json.loads(json_str)
    return Packet(**data) 

def create_data_packet_instance(packet_id: str, source: str, dest: str, service_type: str, payload_size: int, theory_path: List[str], theory_delay: float) -> Packet:
    """Hàm tạo instance Packet DATA hoàn chỉnh."""
    return Packet(
        packetId=packet_id,
        sourceUserId=source,
        destinationUserId=dest,
        payloadSizeByte=payload_size,
        serviceType=service_type,
        theoryPath=theory_path,
        theoryDelayMs=theory_delay
    )