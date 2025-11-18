import json
from typing import Dict, Any, Optional

class User:
    def __init__(self, cityName: str, ipAddress: str, port: int,
                 userId: str, userName: str, latitude: float = 0.0, longitude: float = 0.0):
        self.cityName = cityName
        self.ipAddress = ipAddress
        self.port = port
        self.userId = userId
        self.userName = userName
        self.latitude = latitude
        self.longitude = longitude
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đối tượng User thành dictionary"""
        return {
            "cityName": self.cityName,
            "ipAddress": self.ipAddress,
            "port": self.port,
            "userId": self.userId,
            "userName": self.userName,
            "latitude": self.latitude,
            "longitude": self.longitude
        }
    
    def __str__(self) -> str:
        return f"User({self.userName}, {self.cityName}, {self.ipAddress}:{self.port})"
    
    def __repr__(self) -> str:
        return self.__str__()

class UserManager:
    """Quản lý danh sách users"""
    
    def __init__(self):
        self.users = []
    
    def add_user(self, user: User):
        """Thêm user mới"""
        self.users.append(user)
    
    def create_user(self, cityName: str, ipAddress: str, port: int, 
                   userId: str, userName: str) -> User:
        """Tạo và thêm user mới"""
        user = User(cityName, ipAddress, port, userId, userName)
        self.add_user(user)
        return user
    
    def get_user_by_id(self, userId: str) -> Optional[User]:
        """Lấy user theo ID"""
        for user in self.users:
            if user.userId == userId:
                return user
        return None
    
    def remove_user(self, userId: str) -> bool:
        """Xóa user theo ID"""
        user = self.get_user_by_id(userId)
        if user:
            self.users.remove(user)
            return True
        return False
    
    def save_to_json(self, filename: str = "users.json"):
        """Lưu danh sách users vào file JSON"""
        data = [user.to_dict() for user in self.users]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(self.users)} users to {filename}")
    
    def load_from_json(self, filename: str = "users.json"):
        """Tải danh sách users từ file JSON"""
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            self.users = []
            for item in data:
                user = User(
                    cityName=item["cityName"],
                    ipAddress=item["ipAddress"],
                    port=item["port"],
                    userId=item["userId"],
                    userName=item["userName"],
                    latitude=item.get("latitude", 0.0),
                    longitude=item.get("longitude", 0.0)
                )
                self.users.append(user)
            print(f"✅ Loaded {len(self.users)} users from {filename}")
        except FileNotFoundError:
            print(f"⚠️ File {filename} not found, starting with empty user list")
        except Exception as e:
            print(f"❌ Error loading users: {e}")

# Tạo user mẫu theo yêu cầu
def create_singapore_user() -> User:
    return User(
        cityName="Singapore",
        ipAddress="127.0.0.1",
        port=10000,
        userId="user-Singapore",
        userName="User_Singapore"
    )

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo user manager
    user_manager = UserManager()
    
    # Tạo user Singapore
    sg_user = create_singapore_user()
    user_manager.add_user(sg_user)
    
    # Tạo thêm một vài user khác
    user_manager.create_user(
        cityName="Hanoi",
        ipAddress="127.0.0.1",
        port=10001,
        userId="user-Hanoi",
        userName="User_Hanoi"
    )
    
    user_manager.create_user(
        cityName="Tokyo",
        ipAddress="127.0.0.1",
        port=10002,
        userId="user-Tokyo",
        userName="User_Tokyo"
    )
    
    # Hiển thị thông tin users
    print("📋 Danh sách users:")
    for user in user_manager.users:
        print(f"  - {user}")
    
    # Lưu vào file JSON
    user_manager.save_to_json("users.json")