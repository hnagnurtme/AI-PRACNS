import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

# --- 1. Sử dụng Enum để định nghĩa các loại Node ---
class NodeType(str, Enum):
    GROUND_STATION = "GROUND_STATION"
    LEO_SATELLITE = "LEO_SATELLITE"
    MEO_SATELLITE = "MEO_SATELLITE"
    GEO_SATELLITE = "GEO_SATELLITE"

# --- Định nghĩa lại các class lồng nhau (tương tự như cũ) ---
@dataclass
class Position:
    latitude: float
    longitude: float
    altitude: float  # km

@dataclass
class Velocity:
    velocityX: float
    velocityY: float
    velocityZ: float

@dataclass
class Orbit:
    semiMajorAxisKm: float
    eccentricity: float
    inclinationDeg: float
    raanDeg: float
    argumentOfPerigeeDeg: float
    trueAnomalyDeg: float

@dataclass
class Communication:
    frequencyGHz: float
    bandwidthMHz: float
    transmitPowerDbW: float
    antennaGainDb: float
    beamWidthDeg: float
    maxRangeKm: float
    minElevationDeg: float
    ipAddress: str
    port: int
    protocol : Optional[str] = None


@dataclass
class Status:
    active: bool
    lastUpdated: datetime
    batteryChargePercent: Optional[float] = None

@dataclass
class Metadata:
    operator: str
    launchDate: datetime 
    notes: str

# --- Class chính: NODE (Đã cải tiến) ---
@dataclass
class Node:
    nodeId: str
    nodeName: str
    type: NodeType  # <-- THAY ĐỔI: Sử dụng Enum NodeType
    position: Position
    velocity: Velocity
    communication: Communication
    status: Status
    metadata: Metadata
    orbit: Optional[Orbit] = None
    _id: Any = field(default=None, repr=False) # repr=False để không in ra _id

    # --- Phương thức hỗ trợ đã cải tiến ---

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đối tượng Node thành dictionary, xử lý datetime và Enum."""
        # Sử dụng asdict cho đơn giản và hiệu quả
        data = asdict(self)
        # Chuyển đổi Enum và datetime thành string để tương thích JSON
        data['type'] = self.type.value
        data['status']['lastUpdated'] = self.status.lastUpdated.isoformat()
        data['metadata']['launchDate'] = self.metadata.launchDate.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Tạo đối tượng Node từ dictionary, xử lý datetime và Enum."""
        # Chuyển đổi các giá trị string ngược lại thành đối tượng
        data['type'] = NodeType(data['type'])
        data['status']['lastUpdated'] = datetime.fromisoformat(data['status']['lastUpdated'])
        data['metadata']['launchDate'] = datetime.fromisoformat(data['metadata']['launchDate'])

        # Tự động ánh xạ các dictionary lồng nhau vào các dataclass tương ứng
        nested_fields = {
            'position': Position,
            'velocity': Velocity,
            'communication': Communication,
            'status': Status,
            'metadata': Metadata,
            'orbit': Orbit
        }
        for key, class_type in nested_fields.items():
            if key in data and data[key] is not None:
                data[key] = class_type(**data[key])
        
        return cls(**data)

# --- Ví dụ sử dụng phiên bản mới ---
if __name__ == '__main__':
    sample_data = {
        "nodeId": "LEO-001",
        "nodeName": "Sat-LEO-1",
        "type": "LEO_SATELLITE",
        "_id": "651d95c102c77d46f5611111",
        "position": {"latitude": 10.7626, "longitude": 106.6602, "altitude": 550.0},
        "velocity": {"velocityX": 0.0, "velocityY": 7.56, "velocityZ": 0.0},
        "orbit": {"semiMajorAxisKm": 6871.0, "eccentricity": 0.001, "inclinationDeg": 53.0, "raanDeg": 45.0, "argumentOfPerigeeDeg": 0.0, "trueAnomalyDeg": 12.5},
        "communication": {"frequencyGHz": 14.25, "bandwidthMHz": 500.0, "transmitPowerDbW": 20.0, "antennaGainDb": 15.0, "beamWidthDeg": 3.0, "maxRangeKm": 2000.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.12", "port": 8080},
        # Sử dụng chuẩn ISO 8601 cho chuỗi thời gian
        "status": {"active": True, "batteryChargePercent": 82.5, "lastUpdated": "2025-10-13T16:00:00+00:00"},
        "metadata": {"operator": "VNPT Space", "launchDate": "2024-03-01T00:00:00+00:00", "notes": "Experimental low-latency link node"}
    }

    try:
        # 1. Khởi tạo từ Dictionary
        node_obj = Node.from_dict(sample_data.copy())
        
        print("--- Node Object Khởi tạo Thành công (Phiên bản Cải tiến) ---")
        print(f"ID Node: {node_obj.nodeId}, Loại: {node_obj.type.name}")
        # Có thể thực hiện tính toán thời gian
        time_since_launch = datetime.now(node_obj.metadata.launchDate.tzinfo) - node_obj.metadata.launchDate
        print(f"Thời gian hoạt động: {time_since_launch.days} ngày")
        print(f"Cập nhật lần cuối: {node_obj.status.lastUpdated.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # 2. Chuyển ngược lại thành Dictionary/JSON
        node_dict = node_obj.to_dict()
        print("\n--- Chuyển ngược lại thành Dictionary/JSON ---")
        print(json.dumps(node_dict, indent=2))
        
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {e}")