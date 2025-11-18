import json
import math
import random
import sys
import os
from datetime import datetime, timezone
from typing import List, Tuple, Optional

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from python.utils.db_connector import MongoConnector

class Position:
    def __init__(self, latitude: float, longitude: float, altitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
    
    def to_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude
        }
    
    def to_xyz(self, R_earth: float = 6371.0) -> Tuple[float, float, float]:
        """Chuyển đổi tọa độ địa lý sang tọa độ Descartes 3D"""
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)
        R = R_earth + self.altitude
        x = R * math.cos(lat_rad) * math.cos(lon_rad)
        y = R * math.cos(lat_rad) * math.sin(lon_rad)
        z = R * math.sin(lat_rad)
        return x, y, z

class Orbit:
    def __init__(self, semiMajorAxisKm: float = 0, eccentricity: float = 0, 
                 inclinationDeg: float = 0, raanDeg: float = 0, 
                 argumentOfPerigeeDeg: float = 0, trueAnomalyDeg: float = 0):
        self.semiMajorAxisKm = semiMajorAxisKm
        self.eccentricity = eccentricity
        self.inclinationDeg = inclinationDeg
        self.raanDeg = raanDeg
        self.argumentOfPerigeeDeg = argumentOfPerigeeDeg
        self.trueAnomalyDeg = trueAnomalyDeg
    
    def to_dict(self):
        return {
            "semiMajorAxisKm": self.semiMajorAxisKm,
            "eccentricity": self.eccentricity,
            "inclinationDeg": self.inclinationDeg,
            "raanDeg": self.raanDeg,
            "argumentOfPerigeeDeg": self.argumentOfPerigeeDeg,
            "trueAnomalyDeg": self.trueAnomalyDeg
        }

class Velocity:
    def __init__(self, velocityX: float = 0, velocityY: float = 0, velocityZ: float = 0):
        self.velocityX = velocityX
        self.velocityY = velocityY
        self.velocityZ = velocityZ
    
    def to_dict(self):
        return {
            "velocityX": self.velocityX,
            "velocityY": self.velocityY,
            "velocityZ": self.velocityZ
        }

class Communication:
    def __init__(self, frequencyGHz: float, bandwidthMHz: float, transmitPowerDbW: float,
                 antennaGainDb: float, beamWidthDeg: float, maxRangeKm: float,
                 minElevationDeg: float, ipAddress: str, port: int, protocol: str = "TCP"):
        self.frequencyGHz = frequencyGHz
        self.bandwidthMHz = bandwidthMHz
        self.transmitPowerDbW = transmitPowerDbW
        self.antennaGainDb = antennaGainDb
        self.beamWidthDeg = beamWidthDeg
        self.maxRangeKm = maxRangeKm
        self.minElevationDeg = minElevationDeg
        self.ipAddress = ipAddress
        self.port = port
        self.protocol = protocol
    
    def to_dict(self):
        return {
            "frequencyGHz": self.frequencyGHz,
            "bandwidthMHz": self.bandwidthMHz,
            "transmitPowerDbW": self.transmitPowerDbW,
            "antennaGainDb": self.antennaGainDb,
            "beamWidthDeg": self.beamWidthDeg,
            "maxRangeKm": self.maxRangeKm,
            "minElevationDeg": self.minElevationDeg,
            "ipAddress": self.ipAddress,
            "port": self.port,
            "protocol": self.protocol
        }

class Node:
    # Constants
    R_EARTH = 6371.0  # km
    NODE_TYPE_MAX_RANGE = {
        "GROUND_STATION": 2000.0,
        "LEO_SATELLITE": 3000.0,
        "MEO_SATELLITE": 10000.0,
        "GEO_SATELLITE": 35000.0
    }
    
    MAX_RANGE_MAP = {
        ("LEO_SATELLITE", "MEO_SATELLITE"): 12000,
        ("LEO_SATELLITE", "GEO_SATELLITE"): 40000,
        ("MEO_SATELLITE", "GEO_SATELLITE"): 30000,
    }
    
    def __init__(self, nodeId: str, nodeName: str, nodeType: str, 
                 position: Position, orbit: Orbit, velocity: Velocity,
                 communication: Communication, isOperational: bool = True,
                 batteryChargePercent: int = 100, nodeProcessingDelayMs: float = 1.0,
                 packetLossRate: float = 0.0, resourceUtilization: float = 0.1,
                 packetBufferCapacity: int = 1000, currentPacketCount: int = 0,
                 weather: str = "CLEAR", healthy: bool = True, neighbors: Optional[List[str]] = None):
        
        self.nodeId = nodeId
        self.nodeName = nodeName
        self.nodeType = nodeType
        self.position = position
        self.orbit = orbit
        self.velocity = velocity
        self.communication = communication
        self.isOperational = isOperational
        self.batteryChargePercent = batteryChargePercent
        self.nodeProcessingDelayMs = nodeProcessingDelayMs
        self.packetLossRate = packetLossRate
        self.resourceUtilization = resourceUtilization
        self.packetBufferCapacity = packetBufferCapacity
        self.currentPacketCount = currentPacketCount
        self.weather = weather
        self.lastUpdated = datetime.now(timezone.utc).isoformat()
        self.healthy = healthy
        self.neighbors = neighbors if neighbors is not None else []
    
    def to_dict(self):
        """Chuyển đối tượng Node thành dictionary để lưu JSON"""
        return {
            "nodeId": self.nodeId,
            "nodeName": self.nodeName,
            "nodeType": self.nodeType,
            "position": self.position.to_dict(),
            "orbit": self.orbit.to_dict(),
            "velocity": self.velocity.to_dict(),
            "communication": self.communication.to_dict(),
            "isOperational": self.isOperational,
            "batteryChargePercent": self.batteryChargePercent,
            "nodeProcessingDelayMs": self.nodeProcessingDelayMs,
            "packetLossRate": self.packetLossRate,
            "resourceUtilization": self.resourceUtilization,
            "packetBufferCapacity": self.packetBufferCapacity,
            "currentPacketCount": self.currentPacketCount,
            "weather": self.weather,
            "lastUpdated": self.lastUpdated,
            "healthy": self.healthy,
            "neighbors": self.neighbors
        }
    
    def distance_to(self, other_node: 'Node') -> float:
        """Tính khoảng cách 3D đến node khác (km)"""
        pos1 = self.position.to_xyz(self.R_EARTH)
        pos2 = other_node.position.to_xyz(self.R_EARTH)
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def elevation_to(self, other_node: 'Node') -> float:
        """Tính góc elevation từ node này đến node khác (độ)"""
        gs_pos = self.position.to_xyz(self.R_EARTH)
        sat_pos = other_node.position.to_xyz(self.R_EARTH)
        
        dx = sat_pos[0] - gs_pos[0]
        dy = sat_pos[1] - gs_pos[1]
        dz = sat_pos[2] - gs_pos[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        elev_rad = math.asin(dz/distance)
        return math.degrees(elev_rad)
    
    def can_communicate_with(self, other_node: 'Node') -> bool:
        """Kiểm tra khả năng kết nối với node khác"""
        # Kiểm tra khoảng cách
        distance = self.distance_to(other_node)
        
        # Xác định max range dựa trên loại node
        node_types = (min(self.nodeType, other_node.nodeType), max(self.nodeType, other_node.nodeType))
        
        if node_types in self.MAX_RANGE_MAP:
            max_range = self.MAX_RANGE_MAP[node_types]
        else:
            max_range = min(self.communication.maxRangeKm, other_node.communication.maxRangeKm)
        
        if distance > max_range:
            return False
        
        # Kiểm tra góc elevation nếu một trong hai là ground station
        if self.nodeType == "GROUND_STATION" or other_node.nodeType == "GROUND_STATION":
            if self.nodeType == "GROUND_STATION":
                elev = self.elevation_to(other_node)
                min_elev = self.communication.minElevationDeg
            else:
                elev = other_node.elevation_to(self)
                min_elev = other_node.communication.minElevationDeg
            
            if elev < min_elev:
                return False
        
        return True
    
    def add_neighbor(self, neighbor_id: str):
        """Thêm neighbor vào danh sách"""
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
    
    def remove_neighbor(self, neighbor_id: str):
        """Xóa neighbor khỏi danh sách"""
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)
    
    def update_status(self, battery: int = None, delay: float = None, 
                     loss_rate: float = None, utilization: float = None,
                     packet_count: int = None, weather: str = None,
                     operational: bool = None, healthy: bool = None):
        """Cập nhật trạng thái của node"""
        if battery is not None:
            self.batteryChargePercent = max(0, min(100, battery))
        if delay is not None:
            self.nodeProcessingDelayMs = delay
        if loss_rate is not None:
            self.packetLossRate = loss_rate
        if utilization is not None:
            self.resourceUtilization = utilization
        if packet_count is not None:
            self.currentPacketCount = max(0, min(self.packetBufferCapacity, packet_count))
        if weather is not None:
            self.weather = weather
        if operational is not None:
            self.isOperational = operational
        if healthy is not None:
            self.healthy = healthy
        
        self.lastUpdated = datetime.now(timezone.utc).isoformat()

class NodeManager:
    """Quản lý danh sách nodes trong mạng"""

    def __init__(self):
        self.nodes = []

    def add_node(self, node: Node):
        """Thêm node mới vào mạng"""
        self.nodes.append(node)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Lấy node theo ID"""
        for node in self.nodes:
            if node.nodeId == node_id:
                return node
        return None

    def remove_node(self, node_id: str) -> bool:
        """Xóa node theo ID"""
        node = self.get_node_by_id(node_id)
        if node:
            self.nodes.remove(node)
            return True
        return False

    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Lấy danh sách nodes theo loại"""
        return [node for node in self.nodes if node.nodeType == node_type]

    def get_operational_nodes(self) -> List[Node]:
        """Lấy danh sách nodes đang hoạt động"""
        return [node for node in self.nodes if node.isOperational]

    def save_to_json(self, filename: str = "network_nodes.json"):
        """Lưu danh sách nodes vào file JSON"""
        data = [node.to_dict() for node in self.nodes]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(self.nodes)} nodes to {filename}")

    def load_from_json(self, filename: str = "network_nodes.json"):
        """Tải danh sách nodes từ file JSON"""
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            self.nodes = []
            for item in data:
                # Tạo các đối tượng từ dictionary
                position = Position(**item["position"])
                orbit = Orbit(**item["orbit"])
                velocity = Velocity(**item["velocity"])
                communication = Communication(**item["communication"])

                node = Node(
                    nodeId=item["nodeId"],
                    nodeName=item["nodeName"],
                    nodeType=item["nodeType"],
                    position=position,
                    orbit=orbit,
                    velocity=velocity,
                    communication=communication,
                    isOperational=item["isOperational"],
                    batteryChargePercent=item["batteryChargePercent"],
                    nodeProcessingDelayMs=item["nodeProcessingDelayMs"],
                    packetLossRate=item["packetLossRate"],
                    resourceUtilization=item["resourceUtilization"],
                    packetBufferCapacity=item["packetBufferCapacity"],
                    currentPacketCount=item["currentPacketCount"],
                    weather=item["weather"],
                    healthy=item["healthy"],
                    neighbors=item["neighbors"]
                )
                node.lastUpdated = item["lastUpdated"]
                self.nodes.append(node)

            print(f"✅ Loaded {len(self.nodes)} nodes from {filename}")
        except FileNotFoundError:
            print(f"⚠️ File {filename} not found, starting with empty node list")
        except Exception as e:
            print(f"❌ Error loading nodes: {e}")

class NetworkGenerator:
    """Lớp để tạo mạng lưới nodes"""
    
    @staticmethod
    def create_ground_station(node_id: str, node_name: str, lat: float, lon: float) -> Node:
        """Tạo ground station"""
        position = Position(lat, lon, 0.0)
        orbit = Orbit()
        velocity = Velocity()
        
        communication = Communication(
            frequencyGHz=random.uniform(2, 3),
            bandwidthMHz=random.randint(100, 300),
            transmitPowerDbW=random.randint(15, 30),
            antennaGainDb=random.randint(20, 35),
            beamWidthDeg=random.uniform(10, 30),
            maxRangeKm=Node.NODE_TYPE_MAX_RANGE["GROUND_STATION"],
            minElevationDeg=5,
            ipAddress=f"10.0.0.{random.randint(1, 254)}",
            port=7700 + random.randint(0, 99),
            protocol="TCP"
        )
        
        return Node(
            nodeId=node_id,
            nodeName=node_name,
            nodeType="GROUND_STATION",
            position=position,
            orbit=orbit,
            velocity=velocity,
            communication=communication,
            batteryChargePercent=random.randint(70, 100),
            nodeProcessingDelayMs=round(random.uniform(1, 10), 2),
            packetLossRate=round(random.uniform(0, 0.02), 4),
            resourceUtilization=round(random.uniform(0.1, 0.7), 2),
            packetBufferCapacity=random.randint(500, 5000),
            currentPacketCount=random.randint(0, 500),
            weather=random.choice(["CLEAR", "LIGHT_RAIN", "STORM"])
        )
    
    @staticmethod
    def create_leo_satellite(sat_id: str, alt_min: float = 500, alt_max: float = 600) -> Node:
        """Tạo LEO satellite"""
        position = Position(
            random.uniform(-90, 90),
            random.uniform(-180, 180),
            random.uniform(alt_min, alt_max)
        )
        
        orbit = Orbit(
            inclinationDeg=random.uniform(0, 98)
        )
        
        velocity = Velocity()
        
        communication = Communication(
            frequencyGHz=random.uniform(10, 15),
            bandwidthMHz=random.randint(100, 500),
            transmitPowerDbW=random.randint(20, 30),
            antennaGainDb=random.randint(20, 35),
            beamWidthDeg=random.uniform(5, 20),
            maxRangeKm=Node.NODE_TYPE_MAX_RANGE["LEO_SATELLITE"],
            minElevationDeg=5,
            ipAddress=f"10.1.0.{sat_id.split('-')[1]}",
            port=7800 + int(sat_id.split('-')[1]),
            protocol="TCP"
        )
        
        return Node(
            nodeId=sat_id,
            nodeName=sat_id,
            nodeType="LEO_SATELLITE",
            position=position,
            orbit=orbit,
            velocity=velocity,
            communication=communication,
            batteryChargePercent=random.randint(50, 100),
            nodeProcessingDelayMs=round(random.uniform(1, 10), 2),
            packetLossRate=round(random.uniform(0, 0.05), 4),
            resourceUtilization=round(random.uniform(0.1, 0.8), 2),
            packetBufferCapacity=random.randint(500, 5000),
            currentPacketCount=random.randint(0, 500),
            weather=random.choice(["CLEAR", "LIGHT_RAIN", "STORM"])
        )
    
    @staticmethod
    def create_meo_satellite(sat_id: str, alt_min: float = 10000, alt_max: float = 10500) -> Node:
        """Tạo MEO satellite"""
        position = Position(
            random.uniform(-90, 90),
            random.uniform(-180, 180),
            random.uniform(alt_min, alt_max)
        )
        
        orbit = Orbit(
            inclinationDeg=random.uniform(0, 98)
        )
        
        velocity = Velocity()
        
        communication = Communication(
            frequencyGHz=random.uniform(10, 15),
            bandwidthMHz=random.randint(100, 500),
            transmitPowerDbW=random.randint(20, 30),
            antennaGainDb=random.randint(20, 35),
            beamWidthDeg=random.uniform(5, 20),
            maxRangeKm=Node.NODE_TYPE_MAX_RANGE["MEO_SATELLITE"],
            minElevationDeg=5,
            ipAddress=f"10.2.0.{sat_id.split('-')[1]}",
            port=7900 + int(sat_id.split('-')[1]),
            protocol="TCP"
        )
        
        return Node(
            nodeId=sat_id,
            nodeName=sat_id,
            nodeType="MEO_SATELLITE",
            position=position,
            orbit=orbit,
            velocity=velocity,
            communication=communication,
            batteryChargePercent=random.randint(50, 100),
            nodeProcessingDelayMs=round(random.uniform(1, 10), 2),
            packetLossRate=round(random.uniform(0, 0.05), 4),
            resourceUtilization=round(random.uniform(0.1, 0.8), 2),
            packetBufferCapacity=random.randint(500, 5000),
            currentPacketCount=random.randint(0, 500),
            weather=random.choice(["CLEAR", "LIGHT_RAIN", "STORM"])
        )
    
    @staticmethod
    def create_geo_satellite(sat_id: str, alt: float = 35786) -> Node:
        """Tạo GEO satellite"""
        position = Position(
            random.uniform(-90, 90),
            random.uniform(-180, 180),
            alt
        )
        
        orbit = Orbit()
        velocity = Velocity()
        
        communication = Communication(
            frequencyGHz=random.uniform(10, 15),
            bandwidthMHz=random.randint(100, 500),
            transmitPowerDbW=random.randint(20, 30),
            antennaGainDb=random.randint(20, 35),
            beamWidthDeg=random.uniform(5, 20),
            maxRangeKm=Node.NODE_TYPE_MAX_RANGE["GEO_SATELLITE"],
            minElevationDeg=5,
            ipAddress=f"10.3.0.{sat_id.split('-')[1]}",
            port=8000 + int(sat_id.split('-')[1]),
            protocol="TCP"
        )
        
        return Node(
            nodeId=sat_id,
            nodeName=sat_id,
            nodeType="GEO_SATELLITE",
            position=position,
            orbit=orbit,
            velocity=velocity,
            communication=communication,
            batteryChargePercent=random.randint(50, 100),
            nodeProcessingDelayMs=round(random.uniform(1, 10), 2),
            packetLossRate=round(random.uniform(0, 0.05), 4),
            resourceUtilization=round(random.uniform(0.1, 0.8), 2),
            packetBufferCapacity=random.randint(500, 5000),
            currentPacketCount=random.randint(0, 500),
            weather=random.choice(["CLEAR", "LIGHT_RAIN", "STORM"])
        )

def generate_sample_network() -> List[Node]:
    """Tạo mạng lưới mẫu với các nodes"""
    nodes = []
    generator = NetworkGenerator()
    
    # Ground Stations
    gs_locations = [
        ("GS_HANOI", 21.0285, 105.8542),
        ("GS_HOCHIMINH", 10.7769, 106.7009),
        ("GS_DANANG", 16.0544, 108.2022),
        ("GS_JAKARTA", -6.2088, 106.8456),
        ("GS_SINGAPORE", 1.3521, 103.8198),
        ("GS_TOKYO", 35.6895, 139.6917),
        ("GS_SEOUL", 37.5665, 126.9780),
        ("GS_BANGKOK", 13.7563, 100.5018),
        ("GS_KUALALUMPUR", 3.1390, 101.6869),
        ("GS_DELHI", 28.6139, 77.2090),
        ("GS_SYDNEY", -33.8688, 151.2093),
        ("GS_LONDON", 51.5074, -0.1278),
        ("GS_PARIS", 48.8566, 2.3522),
        ("GS_BERLIN", 52.5200, 13.4050),
        ("GS_NEWYORK", 40.7128, -74.0060),
        ("GS_SANFRAN", 37.7749, -122.4194),
        ("GS_DUBAI", 25.276987, 55.296249),
        ("GS_MOSCOW", 55.7558, 37.6173),
        ("GS_CAIRE", 30.0444, 31.2357),
        ("GS_RIO", -22.9068, -43.1729)
    ]
    
    for gs_id, lat, lon in gs_locations:
        nodes.append(generator.create_ground_station(gs_id, gs_id, lat, lon))
    
    # LEO Satellites
    for i in range(1, 51):
        nodes.append(generator.create_leo_satellite(f"LEO-{i:02d}"))
    
    # MEO Satellites
    for i in range(1, 21):
        nodes.append(generator.create_meo_satellite(f"MEO-{i:02d}"))
    
    # GEO Satellites
    for i in range(1, 11):
        nodes.append(generator.create_geo_satellite(f"GEO-{i:02d}"))
    
    return nodes

def save_network_to_json(nodes: List[Node], filename: str = "network_nodes.json"):
    """Lưu mạng lưới nodes vào file JSON"""
    data = [node.to_dict() for node in nodes]
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Generated {len(nodes)} nodes in '{filename}'")

def load_network_from_json(filename: str = "network_nodes.json") -> List[Node]:
    """Tải mạng lưới nodes từ file JSON"""
    with open(filename, "r") as f:
        data = json.load(f)
    
    nodes = []
    for item in data:
        # Tạo các đối tượng từ dictionary
        position = Position(**item["position"])
        orbit = Orbit(**item["orbit"])
        velocity = Velocity(**item["velocity"])
        communication = Communication(**item["communication"])
        
        node = Node(
            nodeId=item["nodeId"],
            nodeName=item["nodeName"],
            nodeType=item["nodeType"],
            position=position,
            orbit=orbit,
            velocity=velocity,
            communication=communication,
            isOperational=item["isOperational"],
            batteryChargePercent=item["batteryChargePercent"],
            nodeProcessingDelayMs=item["nodeProcessingDelayMs"],
            packetLossRate=item["packetLossRate"],
            resourceUtilization=item["resourceUtilization"],
            packetBufferCapacity=item["packetBufferCapacity"],
            currentPacketCount=item["currentPacketCount"],
            weather=item["weather"],
            healthy=item["healthy"],
            neighbors=item["neighbors"]
        )
        node.lastUpdated = item["lastUpdated"]
        nodes.append(node)
    
    return nodes

def generate_rl_test_data() -> List[Node]:
    """Tạo dữ liệu test cho RL với 30 nodes."""
    nodes = []
    generator = NetworkGenerator()
    
    # 5 Ground Stations
    gs_locations = [
        ("GS_HANOI", 21.0285, 105.8542),
        ("GS_HOCHIMINH", 10.7769, 106.7009),
        ("GS_DANANG", 16.0544, 108.2022),
        ("GS_NEWYORK", 40.7128, -74.0060),
        ("GS_LONDON", 51.5074, -0.1278),
    ]
    for gs_id, lat, lon in gs_locations:
        nodes.append(generator.create_ground_station(gs_id, gs_id, lat, lon))
        
    # 15 LEO Satellites
    for i in range(1, 16):
        nodes.append(generator.create_leo_satellite(f"LEO-{i:02d}"))
        
    # 7 MEO Satellites
    for i in range(1, 8):
        nodes.append(generator.create_meo_satellite(f"MEO-{i:02d}"))
        
    # 3 GEO Satellites
    for i in range(1, 4):
        nodes.append(generator.create_geo_satellite(f"GEO-{i:02d}"))
        
    return nodes

def save_nodes_to_db(nodes: List[Node]):
    """Lưu danh sách các node vào local MongoDB."""
    connector = MongoConnector()
    nodes_data = [node.to_dict() for node in nodes]
    connector.clear_and_insert_nodes(nodes_data)
    print(f"✅ Saved {len(nodes)} nodes to local MongoDB.")

# Example usage
if __name__ == "__main__":
    # Tạo dữ liệu test cho RL
    rl_nodes = generate_rl_test_data()
    
    # Lưu vào local database
    save_nodes_to_db(rl_nodes)
    
    print("✅ RL test data generation and saving complete.")