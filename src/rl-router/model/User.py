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

import socket
import json
import sys
import os
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import time
import random
import uuid
import math
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Add project root to path to allow imports from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.Packet import Packet, QoS, AnalysisData, HopRecord, Position, BufferState, RoutingDecisionInfo, RoutingAlgorithm
from python.utils.db_connector import MongoConnector


class UserManager:
    """
    Quản lý users với đầy đủ chức năng CRUD và vị trí
    """
    
    # Default coordinates for major cities
    CITY_COORDINATES = {
        'Hanoi': (21.0285, 105.8542),
        'Ho Chi Minh City': (10.7769, 106.7009),
        'Da Nang': (16.0544, 108.2022),
        'Singapore': (1.3521, 103.8198),
        'Tokyo': (35.6895, 139.6917),
        'Seoul': (37.5665, 126.9780),
        'Bangkok': (13.7563, 100.5018),
        'Kuala Lumpur': (3.1390, 101.6869),
        'Delhi': (28.6139, 77.2090),
        'Sydney': (-33.8688, 151.2093),
        'London': (51.5074, -0.1278),
        'Paris': (48.8566, 2.3522),
        'Berlin': (52.5200, 13.4050),
        'New York': (40.7128, -74.0060),
        'San Francisco': (37.7749, -122.4194),
        'Dubai': (25.276987, 55.296249),
        'Moscow': (55.7558, 37.6173),
        'Cairo': (30.0444, 31.2357),
        'Rio de Janeiro': (-22.9068, -43.1729),
        'Jakarta': (-6.2088, 106.8456)
    }
    
    def __init__(self, db_connector: Optional[MongoConnector] = None):
        self.db = db_connector or MongoConnector()

    def get_all_users(self) -> List[Dict]:
        """Lấy tất cả users từ database"""
        return self.db.get_all_users()

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Lấy user theo ID"""
        return self.db.get_user(user_id)

    def delete_user(self, user_id: str) -> bool:
        """
        Xóa user khỏi database
        """
        try:
            user = self.db.get_user(user_id)
            if not user:
                print(f"❌ User {user_id} không tồn tại")
                return False
            
            # Xóa user từ database
            users_collection = self.db.db["users"]
            result = users_collection.delete_one({"userId": user_id})
            
            if result.deleted_count > 0:
                print(f"✅ Đã xóa user: {user_id}")
                return True
            else:
                print(f"❌ Không thể xóa user: {user_id}")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi khi xóa user {user_id}: {e}")
            return False

    def delete_all_users(self) -> bool:
        """
        Xóa tất cả users khỏi database
        """
        try:
            users_collection = self.db.db["users"]
            result = users_collection.delete_many({})
            print(f"✅ Đã xóa {result.deleted_count} users")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi xóa tất cả users: {e}")
            return False

    def add_user(self, user_data: Dict) -> bool:
        """
        Thêm user mới vào database
        """
        try:
            # Validate required fields
            required_fields = ['userId', 'userName', 'cityName', 'ipAddress', 'port']
            for field in required_fields:
                if field not in user_data:
                    print(f"❌ Thiếu trường bắt buộc: {field}")
                    return False

            # Check if user already exists
            existing_user = self.db.get_user(user_data['userId'])
            if existing_user:
                print(f"❌ User {user_data['userId']} đã tồn tại")
                return False

            # Set default values
            user_data.setdefault('isActive', True)
            user_data.setdefault('connectionStatus', 'DISCONNECTED')
            user_data.setdefault('bandwidthMbps', 100.0)
            user_data.setdefault('latencyMs', 50.0)
            user_data.setdefault('packetLossRate', 0.0)
            user_data.setdefault('sessionDuration', 0.0)
            user_data.setdefault('dataConsumedMB', 0.0)
            user_data.setdefault('securityLevel', 'MEDIUM')
            user_data.setdefault('allowedProtocols', ['TCP', 'UDP', 'HTTP', 'HTTPS'])
            user_data.setdefault('preferences', {
                'autoReconnect': True,
                'preferredNodes': [],
                'qualityOfService': 'HIGH',
                'encryption': True
            })

            # Set coordinates based on city
            city_name = user_data['cityName']
            if city_name in self.CITY_COORDINATES:
                user_data['latitude'], user_data['longitude'] = self.CITY_COORDINATES[city_name]
            else:
                # Fallback coordinates
                user_data['latitude'] = random.uniform(-90, 90)
                user_data['longitude'] = random.uniform(-180, 180)
                print(f"⚠️  Sử dụng tọa độ ngẫu nhiên cho {city_name}")

            # Add timestamps
            user_data['lastUpdated'] = datetime.now(timezone.utc).isoformat()
            user_data['lastActivity'] = datetime.now(timezone.utc).isoformat()

            # Insert user
            result = self.db.insert_user(user_data)
            if result:
                print(f"✅ Đã thêm user: {user_data['userName']} ({user_data['userId']})")
                print(f"   📍 {user_data['cityName']} - ({user_data['latitude']}, {user_data['longitude']})")
                print(f"   🌐 {user_data['ipAddress']}:{user_data['port']}")
                return True
            else:
                print(f"❌ Không thể thêm user: {user_data['userId']}")
                return False

        except Exception as e:
            print(f"❌ Lỗi khi thêm user: {e}")
            return False

    def update_user_location(self, user_id: str, lat: float, lon: float) -> bool:
        """
        Cập nhật vị trí cho user
        """
        try:
            updates = {
                'latitude': lat,
                'longitude': lon,
                'lastUpdated': datetime.now(timezone.utc).isoformat()
            }
            success = self.db.update_user(user_id, updates)
            if success:
                print(f"✅ Đã cập nhật vị trí cho {user_id}: ({lat}, {lon})")
            else:
                print(f"❌ Không thể cập nhật vị trí cho {user_id}")
            return success
        except Exception as e:
            print(f"❌ Lỗi khi cập nhật vị trí: {e}")
            return False

    def update_user_city(self, user_id: str, city_name: str) -> bool:
        """
        Cập nhật thành phố và tự động set tọa độ
        """
        try:
            if city_name in self.CITY_COORDINATES:
                lat, lon = self.CITY_COORDINATES[city_name]
            else:
                lat, lon = random.uniform(-90, 90), random.uniform(-180, 180)
                print(f"⚠️  Sử dụng tọa độ ngẫu nhiên cho {city_name}")

            updates = {
                'cityName': city_name,
                'latitude': lat,
                'longitude': lon,
                'lastUpdated': datetime.now(timezone.utc).isoformat()
            }
            success = self.db.update_user(user_id, updates)
            if success:
                print(f"✅ Đã cập nhật thành phố cho {user_id}: {city_name} ({lat}, {lon})")
            else:
                print(f"❌ Không thể cập nhật thành phố cho {user_id}")
            return success
        except Exception as e:
            print(f"❌ Lỗi khi cập nhật thành phố: {e}")
            return False

    def create_sample_users(self) -> bool:
        """
        Tạo danh sách users mẫu với vị trí thực tế
        """
        sample_users = [
            {
                'userId': 'user-hanoi',
                'userName': 'User Hanoi',
                'cityName': 'Hanoi',
                'ipAddress': '127.0.0.1',
                'port': 10001
            },
            {
                'userId': 'user-singapore',
                'userName': 'User Singapore', 
                'cityName': 'Singapore',
                'ipAddress': '127.0.0.1',
                'port': 10002
            },
            {
                'userId': 'user-tokyo',
                'userName': 'User Tokyo',
                'cityName': 'Tokyo',
                'ipAddress': '127.0.0.1', 
                'port': 10003
            },
            {
                'userId': 'user-london',
                'userName': 'User London',
                'cityName': 'London',
                'ipAddress': '127.0.0.1',
                'port': 10004
            },
            {
                'userId': 'user-ny',
                'userName': 'User New York',
                'cityName': 'New York',
                'ipAddress': '127.0.0.1',
                'port': 10005
            },
            {
                'userId': 'user-sydney',
                'userName': 'User Sydney',
                'cityName': 'Sydney',
                'ipAddress': '127.0.0.1',
                'port': 10006
            }
        ]

        success_count = 0
        for user_data in sample_users:
            if self.add_user(user_data):
                success_count += 1

        print(f"📊 Đã tạo {success_count}/{len(sample_users)} users mẫu")
        return success_count == len(sample_users)

    def display_all_users(self):
        """
        Hiển thị thông tin tất cả users
        """
        users = self.get_all_users()
        if not users:
            print("📭 Không có users nào trong database")
            return

        print(f"\n👥 DANH SÁCH USERS ({len(users)} users)")
        print("=" * 80)
        
        for i, user in enumerate(users, 1):
            user_id = user.get('userId', 'Unknown')
            user_name = user.get('userName', 'Unknown')
            city = user.get('cityName', 'Unknown')
            ip = user.get('ipAddress', 'Unknown')
            port = user.get('port', 'Unknown')
            lat = user.get('latitude', 0)
            lon = user.get('longitude', 0)
            status = user.get('connectionStatus', 'UNKNOWN')
            
            print(f"{i:2d}. {user_name} ({user_id})")
            print(f"    🏙️  {city} | 📍 ({lat:.4f}, {lon:.4f})")
            print(f"    🌐 {ip}:{port} | 📶 {status}")
            print()

    def find_users_by_city(self, city_name: str) -> List[Dict]:
        """
        Tìm users theo thành phố
        """
        users = self.get_all_users()
        return [user for user in users if user.get('cityName') == city_name]

    def get_user_statistics(self) -> Dict[str, Any]:
        """
        Thống kê về users
        """
        users = self.get_all_users()
        if not users:
            return {"total_users": 0}

        cities = [user.get('cityName', 'Unknown') for user in users]
        city_counts = {}
        for city in cities:
            city_counts[city] = city_counts.get(city, 0) + 1

        connected_users = [user for user in users if user.get('connectionStatus') == 'CONNECTED']
        active_users = [user for user in users if user.get('isActive', False)]

        return {
            "total_users": len(users),
            "connected_users": len(connected_users),
            "active_users": len(active_users),
            "cities_distribution": city_counts,
            "connection_rate": len(connected_users) / len(users) if users else 0
        }


class EnhancedPacketSender:
    """
    Enhanced packet sender với quản lý users tích hợp
    """
    
    def __init__(self, db_connector: Optional[MongoConnector] = None):
        self.db = db_connector or MongoConnector()
        self.user_manager = UserManager(db_connector)

    def _calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points in km"""
        R = 6371.0  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    def find_nearest_ground_station(self, user_id: str) -> Optional[Dict]:
        """
        Find nearest ground station from database for a user
        """
        # Get user from database
        user = self.db.get_user(user_id)
        if not user:
            print(f"❌ User {user_id} not found in database")
            return None

        # Get user's city and coordinates
        city_name = user.get('cityName', 'Unknown')
        user_lat = user.get('latitude')
        user_lon = user.get('longitude')
        
        # If user doesn't have coordinates, use city coordinates
        if user_lat is None or user_lon is None:
            if city_name in self.user_manager.CITY_COORDINATES:
                user_lat, user_lon = self.user_manager.CITY_COORDINATES[city_name]
            else:
                user_lat, user_lon = random.uniform(-90, 90), random.uniform(-180, 180)
            print(f"📍 Using coordinates for {city_name}: ({user_lat}, {user_lon})")

        # Get all ground stations from database
        all_nodes = self.db.get_all_nodes()
        ground_stations = [node for node in all_nodes if node.get('nodeType') == 'GROUND_STATION']

        if not ground_stations:
            print("❌ No ground stations found in database")
            return None

        # Find nearest ground station
        min_distance = float('inf')
        nearest_gs = None

        for gs in ground_stations:
            gs_pos = gs.get('position', {})
            gs_lat = gs_pos.get('latitude', 0)
            gs_lon = gs_pos.get('longitude', 0)

            distance = self._calculate_haversine_distance(user_lat, user_lon, gs_lat, gs_lon)

            if distance < min_distance:
                min_distance = distance
                nearest_gs = gs

        if nearest_gs:
            print(f"📍 Nearest ground station for {user_id}: {nearest_gs.get('nodeName')} ({min_distance:.2f} km)")
        return nearest_gs

    def create_packet_from_database(self, source_user_id: str, destination_user_id: str,
                                   payload: str = "Test payload", use_rl: bool = False) -> Optional[Packet]:
        """
        Create a realistic packet using database information
        """
        print(f"🚀 Creating packet from database: {source_user_id} -> {destination_user_id}")

        # Get users from database
        source_user = self.db.get_user(source_user_id)
        dest_user = self.db.get_user(destination_user_id)

        if not source_user or not dest_user:
            print(f"❌ Users not found in database")
            return None

        # Find nearest ground stations for both users
        source_gs = self.find_nearest_ground_station(source_user_id)
        dest_gs = self.find_nearest_ground_station(destination_user_id)

        if not source_gs or not dest_gs:
            print(f"❌ Could not find ground stations for users")
            return None

        print(f"📍 Source GS: {source_gs.get('nodeName')}")
        print(f"🎯 Destination GS: {dest_gs.get('nodeName')}")

        # Create QoS based on service type
        qos = QoS(
            service_type="REALTIME",
            default_priority=1,
            max_latency_ms=100.0,
            max_jitter_ms=10.0,
            min_bandwidth_mbps=50.0,
            max_loss_rate=0.01
        )

        # Create analysis data
        analysis_data = AnalysisData(
            avg_latency=0.0,
            avg_distance_km=0.0,
            route_success_rate=0.0,
            total_distance_km=0.0,
            total_latency_ms=0.0
        )

        # Create packet with database information
        packet = Packet(
            packet_id=f"pkt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            source_user_id=source_user_id,
            destination_user_id=destination_user_id,
            station_source=source_gs.get('nodeId', ''),
            station_dest=dest_gs.get('nodeId', ''),
            type="DATA",
            time_sent_from_source_ms=datetime.now(timezone.utc).timestamp() * 1000,
            payload_data_base64=payload.encode('utf-8').hex(),
            payload_size_byte=len(payload),
            service_qos=qos,
            current_holding_node_id=source_gs.get('nodeId', ''),
            next_hop_node_id="",
            priority_level=1,
            max_acceptable_latency_ms=100.0,
            max_acceptable_loss_rate=0.01,
            analysis_data=analysis_data,
            use_rl=use_rl,
            ttl=64,
            path_history=[source_gs.get('nodeId', '')]
        )

        print(f"✅ Packet created: {packet.packet_id}")
        print(f"   Algorithm: {'RL' if use_rl else 'Dijkstra'}")
        print(f"   Payload size: {len(payload)} bytes")
        print(f"   QoS: Latency<{packet.max_acceptable_latency_ms}ms, "
              f"Loss<{packet.max_acceptable_loss_rate}, "
              f"BW>{packet.service_qos.min_bandwidth_mbps}Mbps")

        return packet

    def send_packet_to_receiver(self, packet: Packet, receiver_host: str = 'localhost', receiver_port: int = 10004):
        """
        Send packet to TCP Receiver for processing and routing
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(10.0)  # 10 seconds timeout
                sock.connect((receiver_host, receiver_port))

                packet_json = json.dumps(packet, default=lambda o: o.__dict__, indent=2)
                sock.sendall(packet_json.encode('utf-8'))
                print(f"✅ Successfully sent packet {packet.packet_id} to TCP Receiver at {receiver_host}:{receiver_port}")

        except ConnectionRefusedError:
            print(f"❌ Connection refused at {receiver_host}:{receiver_port}. Is TCP Receiver running?")
        except socket.timeout:
            print(f"⏰ Connection timeout to {receiver_host}:{receiver_port}")
        except Exception as e:
            print(f"❌ Error sending packet to TCP Receiver: {e}")


def user_management_demo():
    """
    Demo quản lý users
    """
    print("=" * 80)
    print("👥 DEMO QUẢN LÝ USERS")
    print("=" * 80)
    
    user_manager = UserManager()
    
    while True:
        print("\n" + "=" * 50)
        print("MENU QUẢN LÝ USERS")
        print("=" * 50)
        print("1. Hiển thị tất cả users")
        print("2. Thêm user mới")
        print("3. Xóa user")
        print("4. Xóa tất cả users")
        print("5. Tạo users mẫu")
        print("6. Cập nhật vị trí user")
        print("7. Thống kê users")
        print("8. Thoát")
        print("=" * 50)
        
        choice = input("Chọn chức năng (1-8): ").strip()
        
        if choice == '1':
            user_manager.display_all_users()
            
        elif choice == '2':
            print("\n➕ THÊM USER MỚI")
            user_data = {
                'userId': input("User ID: ").strip(),
                'userName': input("User Name: ").strip(),
                'cityName': input("City: ").strip(),
                'ipAddress': input("IP Address: ").strip() or '127.0.0.1',
                'port': int(input("Port: ").strip() or '10000')
            }
            user_manager.add_user(user_data)
            
        elif choice == '3':
            print("\n🗑️ XÓA USER")
            user_id = input("User ID cần xóa: ").strip()
            user_manager.delete_user(user_id)
            
        elif choice == '4':
            print("\n⚠️ XÓA TẤT CẢ USERS")
            confirm = input("Bạn có chắc muốn xóa tất cả users? (y/N): ").strip().lower()
            if confirm == 'y':
                user_manager.delete_all_users()
                
        elif choice == '5':
            print("\n🎯 TẠO USERS MẪU")
            user_manager.create_sample_users()
            
        elif choice == '6':
            print("\n📍 CẬP NHẬT VỊ TRÍ")
            user_id = input("User ID: ").strip()
            try:
                lat = float(input("Latitude: ").strip())
                lon = float(input("Longitude: ").strip())
                user_manager.update_user_location(user_id, lat, lon)
            except ValueError:
                print("❌ Tọa độ không hợp lệ")
                
        elif choice == '7':
            print("\n📊 THỐNG KÊ USERS")
            stats = user_manager.get_user_statistics()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
        elif choice == '8':
            print("👋 Thoát chương trình")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ")


def main():
    """
    Main function với cả packet sending và user management
    """
    print("=" * 80)
    print("🚀 ENHANCED TCP PACKET SENDER & USER MANAGER")
    print("=" * 80)
    
    while True:
        print("\n" + "=" * 50)
        print("MENU CHÍNH")
        print("=" * 50)
        print("1. Quản lý Users")
        print("2. Gửi Packets (Demo)")
        print("3. Thoát")
        print("=" * 50)
        
        choice = input("Chọn chức năng (1-3): ").strip()
        
        if choice == '1':
            user_management_demo()
        elif choice == '2':
            # Gọi hàm gửi packets demo từ code trước
            send_packets_demo()
        elif choice == '3':
            print("👋 Thoát chương trình")
            break
        else:
            print("❌ Lựa chọn không hợp lệ")


def send_packets_demo():
    """
    Demo gửi packets
    """
    receiver_host = 'localhost'
    receiver_port = 10004
    
    print("\n" + "=" * 70)
    print("📦 DEMO GỬI PACKETS")
    print("=" * 70)
    
    sender = EnhancedPacketSender()
    user_manager = UserManager()
    
    # Hiển thị users hiện có
    user_manager.display_all_users()
    
    # Gửi test packets
    test_cases = [
        ("user-singapore", "user-hanoi", "RL"),
        ("user-tokyo", "user-london", "Dijkstra"),
        ("user-ny", "user-sydney", "RL")
    ]
    
    for source, dest, algorithm in test_cases:
        print(f"\n🔄 Gửi packet: {source} -> {dest} ({algorithm})")
        packet = sender.create_packet_from_database(
            source_user_id=source,
            destination_user_id=dest,
            payload=f"Hello from {source} to {dest} using {algorithm}!",
            use_rl=(algorithm == "RL")
        )
        if packet:
            sender.send_packet_to_receiver(packet, receiver_host, receiver_port)
        time.sleep(1)


if __name__ == '__main__':
    main()