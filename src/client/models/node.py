import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# --- 1. Class lồng nhau: TỌA ĐỘ KHÔNG GIAN ---
@dataclass
class Position:
    latitude: float
    longitude: float
    altitude: float  # km

# --- 2. Class lồng nhau: ĐỘNG HỌC ---
@dataclass
class Velocity:
    velocityX: float
    velocityY: float
    velocityZ: float

# --- 3. Class lồng nhau: QUỸ ĐẠO (Tùy chọn) ---
@dataclass
class Orbit:
    semiMajorAxisKm: float
    eccentricity: float
    inclinationDeg: float
    raanDeg: float
    argumentOfPerigeeDeg: float
    trueAnomalyDeg: float

# --- 4. Class lồng nhau: THÔNG SỐ TRUYỀN THÔNG ---
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

# --- 5. Class lồng nhau: TRẠNG THÁI HOẠT ĐỘNG ---
@dataclass
class Status:
    active: bool
    batteryChargePercent: Optional[float]
    lastUpdated: str  # Sử dụng string cho dễ dàng tương thích với JSON/MongoDB

# --- 6. Class lồng nhau: THÔNG TIN BỔ TRỢ ---
@dataclass
class Metadata:
    operator: str
    launchDate: str  # Sử dụng string cho dễ dàng tương thích với JSON/MongoDB
    notes: str

# --- 7. Class Chính: NODE ---
@dataclass
class Node:
    """
    Biểu diễn một node mạng (vệ tinh hoặc trạm mặt đất) với các thông số vật lý 
    và truyền thông chi tiết.
    """
    nodeId: str
    nodeName: str
    type: str # Enum: GROUND_STATION | LEO_SATELLITE | MEO_SATELLITE | GEO_SATELLITE
    
    # Các đối tượng lồng nhau
    position: Position
    velocity: Velocity
    orbit: Optional[Orbit] # Có thể là null nếu là trạm mặt đất
    communication: Communication
    status: Status
    metadata: Metadata

    # ID nội bộ MongoDB, không cần ánh xạ nếu không sử dụng trực tiếp
    _id: Any = field(default=None, metadata={'json': '_id'}) 

    # --- Phương thức hỗ trợ ---

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đối tượng Node thành dictionary, thích hợp cho JSON serialization."""
        def recursive_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: recursive_to_dict(v) for k, v in obj.__dict__.items() if v is not None}
            elif isinstance(obj, list):
                return [recursive_to_dict(i) for i in obj]
            return obj
        
        # Dùng json.loads/dumps để xử lý dataclasses dễ dàng hơn
        return json.loads(json.dumps(self, default=recursive_to_dict))


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Tạo đối tượng Node từ dictionary (ví dụ: từ MongoDB hoặc JSON)."""
        
        # Hàm hỗ trợ tạo đối tượng lồng nhau từ dictionary
        def create_nested(cls_type, key):
            # Kiểm tra nếu key tồn tại và không phải là None
            if key in data and data[key] is not None:
                # Nếu là Orbit và giá trị là None (cho trạm mặt đất), trả về None
                if cls_type is Orbit and data[key] is None:
                    return None
                return cls_type(**data.pop(key))
            # Xử lý trường hợp Orbit là null
            return None if cls_type is Orbit else cls_type() # Nếu không phải Orbit, cần xử lý mặc định

        # Tái tạo các đối tượng lồng nhau (theo thứ tự từ nhỏ đến lớn)
        position_obj = Position(**data.pop('position'))
        velocity_obj = Velocity(**data.pop('velocity'))
        communication_obj = Communication(**data.pop('communication'))
        status_obj = Status(**data.pop('status'))
        metadata_obj = Metadata(**data.pop('metadata'))
        
        # Xử lý Orbit (có thể là None)
        orbit_data = data.pop('orbit', None)
        orbit_obj = Orbit(**orbit_data) if orbit_data is not None else None

        # Trả về đối tượng Node hoàn chỉnh
        return cls(
            position=position_obj,
            velocity=velocity_obj,
            orbit=orbit_obj,
            communication=communication_obj,
            status=status_obj,
            metadata=metadata_obj,
            **data # Các trường còn lại (nodeId, type, etc.)
        )

# --- Ví dụ về cách sử dụng ---
if __name__ == '__main__':
    # Giả lập dữ liệu đầu vào (từ JSON hoặc MongoDB)
    sample_data = {
      "nodeId": "LEO-001",
      "nodeName": "Sat-LEO-1",
      "type": "LEO_SATELLITE",
      "_id": "651d95c102c77d46f5611111", # Giả lập ObjectId
      "position": {
        "latitude": 10.7626,
        "longitude": 106.6602,
        "altitude": 550.0
      },
      "velocity": {
        "velocityX": 0.0,
        "velocityY": 7.56,
        "velocityZ": 0.0
      },
      "orbit": {
        "semiMajorAxisKm": 6871.0,
        "eccentricity": 0.001,
        "inclinationDeg": 53.0,
        "raanDeg": 45.0,
        "argumentOfPerigeeDeg": 0.0,
        "trueAnomalyDeg": 12.5
      },
      "communication": {
        "frequencyGHz": 14.25,
        "bandwidthMHz": 500.0,
        "transmitPowerDbW": 20.0,
        "antennaGainDb": 15.0,
        "beamWidthDeg": 3.0,
        "maxRangeKm": 2000.0,
        "minElevationDeg": 10.0,
        "ipAddress": "10.0.0.12",
        "port": 8080
      },
      "status": {
        "active": True,
        "batteryChargePercent": 82.5,
        "lastUpdated": "2025-10-13T16:00:00Z"
      },
      "metadata": {
        "operator": "VNPT Space",
        "launchDate": "2024-03-01T00:00:00Z",
        "notes": "Experimental low-latency link node"
      }
    }

    # 1. Khởi tạo từ Dictionary (Ví dụ: từ MongoDB/JSON)
    try:
        node_obj = Node.from_dict(sample_data.copy())
        
        print("--- Node Object Khởi tạo Thành công ---")
        print(f"ID Node: {node_obj.nodeId} ({node_obj.nodeName})")
        print(f"Loại: {node_obj.type}")
        print(f"Vĩ độ: {node_obj.position.latitude}°")
        print(f"Vận tốc Y: {node_obj.velocity.velocityY} km/s")
        print(f"Bán trục lớn (Orbit): {node_obj.orbit.semiMajorAxisKm if node_obj.orbit else 'N/A'}")
        print(f"Băng thông: {node_obj.communication.bandwidthMHz} MHz")
        print(f"Đang hoạt động: {node_obj.status.active}")

        # 2. Chuyển ngược lại thành Dictionary/JSON
        node_dict = node_obj.to_dict()
        print("\n--- Chuyển ngược lại thành Dictionary/JSON ---")
        print(json.dumps(node_dict, indent=2)[:300] + "...")
        
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {e}")