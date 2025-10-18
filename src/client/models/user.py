import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# --- 1. Định nghĩa các Dataclass lồng nhau ---
# Việc này giúp cấu trúc dữ liệu trở nên rõ ràng và tường minh.

@dataclass
class UserPosition:
    """Đại diện cho tọa độ địa lý của người dùng."""
    latitude: float
    longitude: float
    altitude: float = 0.0

@dataclass
class UserCommunication:
    """Đại diện cho thông tin kết nối mạng của người dùng."""
    ipAddress: str
    port: int

@dataclass
class UserStatus:
    """Đại diện cho trạng thái hiện tại của người dùng."""
    active: bool
    lastSeen: datetime

# --- 2. Định nghĩa Class User chính ---
# Class này tổng hợp các thành phần trên thành một đối tượng User hoàn chỉnh.

@dataclass
class User:
    """Đại diện cho một người dùng cuối trong hệ thống mô phỏng."""
    userId: str
    userName: str
    position: UserPosition
    communication: UserCommunication
    status: UserStatus
    _id: Any = field(default=None, repr=False) # Dùng cho _id của MongoDB, không hiển thị ra ngoài

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi (serialize) đối tượng User thành một dictionary.
        Phương thức này sẽ chuyển đổi datetime thành chuỗi ISO 8601 để lưu trữ an toàn.
        """
        data = asdict(self)
        # Chuyển đối tượng datetime thành chuỗi chuẩn ISO để tương thích với JSON/MongoDB
        data['status']['lastSeen'] = self.status.lastSeen.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        Tạo (deserialize) một đối tượng User từ dictionary.
        Phương thức này tự động chuyển đổi các dictionary con và chuỗi thời gian.
        """
        # Chuyển đổi các dictionary lồng nhau thành các đối tượng dataclass tương ứng
        data['position'] = UserPosition(**data['position'])
        data['communication'] = UserCommunication(**data['communication'])

        # Xử lý việc chuyển đổi chuỗi thời gian ISO 8601 ngược lại thành đối tượng datetime
        if isinstance(data['status']['lastSeen'], str):
            # Đảm bảo đối tượng datetime tạo ra có thông tin múi giờ (timezone-aware)
            if data['status']['lastSeen'].upper().endswith('Z'):
                    data['status']['lastSeen'] = datetime.fromisoformat(data['status']['lastSeen'].replace('Z', '+00:00'))
            else:
                    data['status']['lastSeen'] = datetime.fromisoformat(data['status']['lastSeen'])

        data['status'] = UserStatus(**data['status'])

        return cls(**data)

# --- 3. Ví dụ sử dụng ---
# Đoạn code này chỉ chạy khi bạn thực thi trực tiếp file này.

if __name__ == '__main__':
    # Dữ liệu mẫu, tương tự như dữ liệu bạn lấy từ MongoDB
    sample_user_data = {
        "userId": "USER-01",
        "userName": "MobileUser-DaNang",
        "_id": "651d95c102c77d46f5699999", # ID mẫu từ MongoDB
        "position": { "latitude": 16.0545, "longitude": 108.2022, "altitude": 0.01 },
        "communication": {
            "ipAddress": "192.168.1.101",
            "port": 9001
        },
        "status": {
            "active": True,
            "lastSeen": "2025-10-17T09:30:00+00:00" # Chuẩn ISO 8601
        }
    }

    print("--- 1. Tạo đối tượng User từ dictionary ---")
    try:
        user_object = User.from_dict(sample_user_data.copy())
        print(f"Tạo đối tượng thành công: {user_object.userName}")
        print(f"   - ID người dùng: {user_object.userId}")
        print(f"   - Tọa độ: Vĩ độ={user_object.position.latitude}, Kinh độ={user_object.position.longitude}")
        print(f"   - Trạng thái: Hoạt động -> {user_object.status.active}")

        # Giờ đây bạn có thể thao tác trực tiếp với đối tượng datetime
        time_since_last_seen = datetime.now(timezone.utc) - user_object.status.lastSeen
        print(f"   - Thời gian từ lần cuối thấy: {time_since_last_seen}")

    except Exception as e:
        print(f"❌ Lỗi khi tạo đối tượng: {e}")


    print("\n--- 2. Chuyển đổi đối tượng User ngược lại thành dictionary ---")
    try:
        user_dictionary = user_object.to_dict()
        # Sử dụng json.dumps để in ra đẹp hơn
        print(json.dumps(user_dictionary, indent=2))

        # Kiểm tra để chắc chắn rằng 'lastSeen' đã được chuyển về dạng chuỗi
        print(f"\n   - Kiểu dữ liệu của 'lastSeen' trong dictionary: {type(user_dictionary['status']['lastSeen'])}")

    except Exception as e:
        print(f"❌ Lỗi khi chuyển thành dictionary: {e}")