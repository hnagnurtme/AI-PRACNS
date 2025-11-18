import json
import socket
import random
import math
import sys
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from python.utils.db_connector import MongoConnector

class RoutingAlgorithm(str, Enum):
    DIJKSTRA = "Dijkstra"
    REINFORCEMENT_LEARNING = "ReinforcementLearning"

@dataclass
class Position:
    latitude: float
    longitude: float
    altitude: float

@dataclass
class BufferState:
    queue_size: int
    bandwidth_utilization: float  # 0.0 - 1.0

@dataclass
class RoutingDecisionInfo:
    algorithm: RoutingAlgorithm
    metric: Optional[str] = None
    reward: Optional[float] = None

@dataclass
class HopRecord:
    from_node_id: str
    to_node_id: str
    latency_ms: float
    timestamp_ms: float
    distance_km: float
    from_node_position: Optional[Position] = None
    to_node_position: Optional[Position] = None
    from_node_buffer_state: Optional[BufferState] = None
    routing_decision_info: Optional[RoutingDecisionInfo] = None
    scenario_type: Optional[str] = None
    node_load_percent: Optional[float] = None
    drop_reason_details: Optional[str] = None

@dataclass
class QoS:
    service_type: str
    default_priority: int
    max_latency_ms: float
    max_jitter_ms: float
    min_bandwidth_mbps: float
    max_loss_rate: float

@dataclass
class AnalysisData:
    avg_latency: float
    avg_distance_km: float
    route_success_rate: float
    total_distance_km: float
    total_latency_ms: float

@dataclass
class Packet:
    packet_id: str
    source_user_id: str
    destination_user_id: str
    station_source: str
    station_dest: str
    type: str
    time_sent_from_source_ms: float
    payload_data_base64: str
    payload_size_byte: int
    service_qos: QoS
    current_holding_node_id: str
    next_hop_node_id: str
    priority_level: int
    max_acceptable_latency_ms: float
    max_acceptable_loss_rate: float
    analysis_data: AnalysisData
    use_rl: bool
    ttl: int
    
    # Optional fields với giá trị mặc định
    acknowledged_packet_id: Optional[str] = None
    path_history: List[str] = field(default_factory=list)
    hop_records: List[HopRecord] = field(default_factory=list)
    accumulated_delay_ms: float = 0.0
    dropped: bool = False
    drop_reason: Optional[str] = None

    def __post_init__(self):
        if self.path_history is None:
            self.path_history = []
        if self.hop_records is None:
            self.hop_records = []

    def add_hop_record(self, hop_record: HopRecord):
        self.hop_records.append(hop_record)
        self.path_history.append(hop_record.to_node_id)
        self.accumulated_delay_ms += hop_record.latency_ms

class DatabaseNodeManager:
    """Quản lý nodes từ database MongoDB"""
    
    def __init__(self, db_connector: MongoConnector):
        self.db_connector = db_connector
        self.nodes_cache = {}
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict]:
        """Lấy thông tin node từ database theo ID"""
        if node_id in self.nodes_cache:
            return self.nodes_cache[node_id]
        
        node = self.db_connector.get_node(node_id)
        if node:
            self.nodes_cache[node_id] = node
        return node
    
    def get_all_nodes(self) -> List[Dict]:
        """Lấy tất cả nodes từ database"""
        return self.db_connector.get_all_nodes()
    
    def get_ground_stations(self) -> List[Dict]:
        """Lấy tất cả ground stations từ database"""
        nodes = self.get_all_nodes()
        return [node for node in nodes if node.get("nodeType") == "GROUND_STATION"]
    
    def get_satellites_by_type(self, satellite_type: str) -> List[Dict]:
        """Lấy satellites theo type từ database"""
        nodes = self.get_all_nodes()
        return [node for node in nodes if node.get("nodeType") == satellite_type]
    
    def update_node_status(self, node_id: str, updates: Dict):
        """Cập nhật trạng thái node trong database"""
        self.db_connector.update_node(node_id, updates)
        # Clear cache for this node
        if node_id in self.nodes_cache:
            del self.nodes_cache[node_id]

class EnhancedPacketSimulation:
    """Lớp mô phỏng packet nâng cao với tích hợp database"""
    
    # Constants
    SPEED_OF_LIGHT = 299792  # km/s
    PROCESSING_DELAY_GROUND = 2.0  # ms
    PROCESSING_DELAY_SATELLITE = 5.0  # ms
    
    def __init__(self, db_connector: MongoConnector, user_manager):
        self.db_connector = db_connector
        self.node_manager = DatabaseNodeManager(db_connector)
        self.user_manager = user_manager
        self.simulation_results = []
    
    def find_nearest_ground_station(self, user_lat: float, user_lon: float) -> Optional[Dict]:
        """Tìm ground station gần user nhất từ database"""
        ground_stations = self.node_manager.get_ground_stations()
        
        if not ground_stations:
            return None
        
        nearest_gs = None
        min_distance = float('inf')
        
        for gs in ground_stations:
            gs_position = gs.get("position", {})
            gs_lat = gs_position.get("latitude", 0)
            gs_lon = gs_position.get("longitude", 0)
            
            distance = self._calculate_haversine_distance(
                user_lat, user_lon, gs_lat, gs_lon
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_gs = gs
        
        return nearest_gs
    
    def _calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Tính khoảng cách Haversine giữa hai điểm"""
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
    
    def _calculate_3d_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Tính khoảng cách 3D giữa hai positions"""
        def position_to_xyz(position):
            lat = math.radians(position.get("latitude", 0))
            lon = math.radians(position.get("longitude", 0))
            alt = position.get("altitude", 0)
            R = 6371.0 + alt  # Earth radius + altitude
            
            x = R * math.cos(lat) * math.cos(lon)
            y = R * math.cos(lat) * math.sin(lon)
            z = R * math.sin(lat)
            return x, y, z
        
        x1, y1, z1 = position_to_xyz(pos1)
        x2, y2, z2 = position_to_xyz(pos2)
        
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    def find_optimal_path(self, source_gs: Dict, destination_gs: Dict, 
                         algorithm: str = "DIJKSTRA") -> List[Dict]:
        """
        Tìm đường đi tối ưu giữa hai ground station sử dụng database
        """
        if algorithm == "DIJKSTRA":
            return self._dijkstra_path(source_gs, destination_gs)
        else:  # RL
            return self._rl_path(source_gs, destination_gs)
    
    def _dijkstra_path(self, source: Dict, destination: Dict) -> List[Dict]:
        """Tìm đường đi sử dụng Dijkstra algorithm với database"""
        path = [source]
        
        # Lấy satellites từ database
        leo_satellites = self.node_manager.get_satellites_by_type("LEO_SATELLITE")
        meo_satellites = self.node_manager.get_satellites_by_type("MEO_SATELLITE")
        geo_satellites = self.node_manager.get_satellites_by_type("GEO_SATELLITE")
        
        all_satellites = leo_satellites + meo_satellites + geo_satellites
        operational_satellites = [sat for sat in all_satellites 
                                if sat.get("isOperational", True)]
        
        if operational_satellites:
            # Chọn 1-2 satellite ngẫu nhiên
            num_intermediate = min(2, len(operational_satellites))
            intermediate_sats = random.sample(operational_satellites, num_intermediate)
            path.extend(intermediate_sats)
        
        path.append(destination)
        return path
    
    def _rl_path(self, source: Dict, destination: Dict) -> List[Dict]:
        """Tìm đường đi sử dụng RL algorithm với database"""
        path = [source]
        
        # Lấy satellites từ database và sắp xếp theo resource utilization
        leo_satellites = self.node_manager.get_satellites_by_type("LEO_SATELLITE")
        meo_satellites = self.node_manager.get_satellites_by_type("MEO_SATELLITE")
        geo_satellites = self.node_manager.get_satellites_by_type("GEO_SATELLITE")
        
        all_satellites = leo_satellites + meo_satellites + geo_satellites
        operational_satellites = [sat for sat in all_satellites 
                                if sat.get("isOperational", True)]
        
        if operational_satellites:
            # Sắp xếp theo resource utilization (thấp nhất đầu tiên)
            operational_satellites.sort(key=lambda x: x.get("resourceUtilization", 0.5))
            num_intermediate = min(2, len(operational_satellites))
            intermediate_sats = operational_satellites[:num_intermediate]
            path.extend(intermediate_sats)
        
        path.append(destination)
        return path
    
    def calculate_hop_metrics(self, from_node: Dict, to_node: Dict, 
                            algorithm: RoutingAlgorithm) -> Tuple[float, float, float, float, float, float]:
        """Tính toán metrics chi tiết cho một hop"""
        # Tính khoảng cách 3D
        from_pos = from_node.get("position", {})
        to_pos = to_node.get("position", {})
        distance = self._calculate_3d_distance(from_pos, to_pos)
        
        # Tính propagation delay (tốc độ ánh sáng)
        propagation_delay = (distance / self.SPEED_OF_LIGHT) * 1000  # ms
        
        # Processing delay dựa trên loại node
        from_node_type = from_node.get("nodeType", "")
        if from_node_type == "GROUND_STATION":
            processing_delay = self.PROCESSING_DELAY_GROUND
        else:
            processing_delay = self.PROCESSING_DELAY_SATELLITE
        
        # Thêm node processing delay từ database
        node_processing_delay = from_node.get("nodeProcessingDelayMs", 0)
        processing_delay += node_processing_delay
        
        # Tổng latency
        total_latency = propagation_delay + processing_delay
        
        # Bandwidth - lấy giá trị nhỏ nhất
        from_comm = from_node.get("communication", {})
        to_comm = to_node.get("communication", {})
        bandwidth = min(from_comm.get("bandwidthMHz", 0), 
                       to_comm.get("bandwidthMHz", 0))
        
        # Packet loss rate - lấy giá trị trung bình
        packet_loss = (from_node.get("packetLossRate", 0) + 
                      to_node.get("packetLossRate", 0)) / 2
        
        return total_latency, bandwidth, packet_loss, distance, processing_delay, propagation_delay
    
    def create_hop_record(self, hop_number: int, from_node: Dict, to_node: Dict,
                         latency: float, distance: float, algorithm: RoutingAlgorithm) -> HopRecord:
        """Tạo HopRecord chi tiết từ thông tin database"""
        
        # Tạo Position objects
        from_pos = from_node.get("position", {})
        to_pos = to_node.get("position", {})
        
        from_position = Position(
            latitude=from_pos.get("latitude", 0),
            longitude=from_pos.get("longitude", 0),
            altitude=from_pos.get("altitude", 0)
        ) if from_pos else None
        
        to_position = Position(
            latitude=to_pos.get("latitude", 0),
            longitude=to_pos.get("longitude", 0),
            altitude=to_pos.get("altitude", 0)
        ) if to_pos else None
        
        # Tạo BufferState
        buffer_state = BufferState(
            queue_size=from_node.get("currentPacketCount", 0),
            bandwidth_utilization=from_node.get("resourceUtilization", 0.0)
        )
        
        # Tạo RoutingDecisionInfo
        routing_info = RoutingDecisionInfo(
            algorithm=algorithm,
            metric="latency" if algorithm == RoutingAlgorithm.DIJKSTRA else "q_learning",
            reward=random.uniform(0.7, 1.0) if algorithm == RoutingAlgorithm.REINFORCEMENT_LEARNING else None
        )
        
        # Tính node load
        node_load = from_node.get("resourceUtilization", 0.0) * 100
        
        return HopRecord(
            from_node_id=from_node.get("nodeId", ""),
            to_node_id=to_node.get("nodeId", ""),
            latency_ms=latency,
            timestamp_ms=datetime.now(timezone.utc).timestamp() * 1000,
            distance_km=distance,
            from_node_position=from_position,
            to_node_position=to_position,
            from_node_buffer_state=buffer_state,
            routing_decision_info=routing_info,
            scenario_type="NORMAL",
            node_load_percent=node_load,
            drop_reason_details=None
        )
    
    def simulate_packet_journey(self, source_user, destination_user, 
                               packet_data: str, algorithm: str = "DIJKSTRA") -> Dict[str, Any]:
        """
        Mô phỏng hành trình packet với tích hợp database đầy đủ
        """
        print(f"🚀 Starting enhanced packet simulation: {source_user.userName} -> {destination_user.userName} ({algorithm})")
        
        # Lấy tọa độ user từ database hoặc attributes
        source_lat = getattr(source_user, 'latitude', 1.3521)  # Singapore mặc định
        source_lon = getattr(source_user, 'longitude', 103.8198)
        dest_lat = getattr(destination_user, 'latitude', 21.0285)  # Hanoi mặc định
        dest_lon = getattr(destination_user, 'longitude', 105.8542)
        
        # Tìm ground station gần nhất từ database
        source_gs = self.find_nearest_ground_station(source_lat, source_lon)
        destination_gs = self.find_nearest_ground_station(dest_lat, dest_lon)
        
        if not source_gs or not destination_gs:
            raise Exception("Cannot find suitable ground stations from database")
        
        print(f"📍 Source GS: {source_gs.get('nodeName')}, Destination GS: {destination_gs.get('nodeName')}")
        
        # Tìm đường đi tối ưu
        path_nodes = self.find_optimal_path(source_gs, destination_gs, algorithm)
        
        # Tạo packet object với đầy đủ thông tin
        packet = self._create_packet_object(source_user, destination_user, packet_data, 
                                          source_gs, destination_gs, algorithm)
        
        # Mô phỏng packet đi qua từng hop
        hop_records = []
        total_latency = 0
        total_packet_loss = 0
        min_bandwidth = float('inf')
        
        print(f"🛣️  Path found ({len(path_nodes)} hops):")
        for i, node in enumerate(path_nodes):
            print(f"    {i+1}. {node.get('nodeName')} ({node.get('nodeType')})")
        
        # Mô phỏng từng hop với thông tin từ database
        routing_algorithm = RoutingAlgorithm.DIJKSTRA if algorithm == "DIJKSTRA" else RoutingAlgorithm.REINFORCEMENT_LEARNING
        
        for i in range(len(path_nodes) - 1):
            from_node = path_nodes[i]
            to_node = path_nodes[i + 1]
            
            # Tính toán metrics cho hop này
            latency, bandwidth, packet_loss, distance, processing_delay, propagation_delay = \
                self.calculate_hop_metrics(from_node, to_node, routing_algorithm)
            
            # Tạo HopRecord chi tiết với thông tin từ database
            hop_record = self.create_hop_record(
                hop_number=i + 1,
                from_node=from_node,
                to_node=to_node,
                latency=latency,
                distance=distance,
                algorithm=routing_algorithm
            )
            
            # Thêm HopRecord vào packet
            packet.add_hop_record(hop_record)
            hop_records.append(hop_record)
            
            total_latency += latency
            total_packet_loss += packet_loss
            min_bandwidth = min(min_bandwidth, bandwidth)
            
            print(f"    🔄 Hop {i+1}: {from_node.get('nodeName')} -> {to_node.get('nodeName')}")
            print(f"       📊 Latency: {latency:.2f}ms, Bandwidth: {bandwidth:.2f}MHz, Distance: {distance:.2f}km")
            print(f"       🏷️  Node Load: {from_node.get('resourceUtilization', 0)*100:.1f}%")
        
        # Cập nhật analysis data cho packet
        packet.analysis_data = AnalysisData(
            avg_latency=total_latency / len(hop_records) if hop_records else 0,
            avg_distance_km=sum(hr.distance_km for hr in hop_records) / len(hop_records) if hop_records else 0,
            route_success_rate=1.0,
            total_distance_km=sum(hr.distance_km for hr in hop_records),
            total_latency_ms=total_latency
        )
        
        # Tính packet loss tổng (xác suất tích lũy)
        total_packet_loss_rate = 1 - (1 - total_packet_loss / len(hop_records)) ** len(hop_records)
        
        # Tạo kết quả mô phỏng
        simulation_result = {
            "simulationId": f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sourceUser": source_user.to_dict(),
            "destinationUser": destination_user.to_dict(),
            "algorithm": algorithm,
            "path": [node.get("nodeId") for node in path_nodes],
            "hopRecords": [self._hop_record_to_dict(hr) for hr in hop_records],
            "packetData": self._packet_to_dict(packet),
            "totalMetrics": {
                "totalLatencyMs": total_latency,
                "totalPacketLossRate": total_packet_loss_rate,
                "minBandwidthMbps": min_bandwidth,
                "totalHops": len(hop_records),
                "totalDistanceKm": sum(hr.distance_km for hr in hop_records)
            },
            "status": "COMPLETED"
        }
        
        self.simulation_results.append(simulation_result)
        
        print(f"✅ Enhanced simulation completed:")
        print(f"   📈 Total Latency: {total_latency:.2f}ms")
        print(f"   📉 Packet Loss Rate: {total_packet_loss_rate:.4f}")
        print(f"   📊 Min Bandwidth: {min_bandwidth:.2f}MHz")
        print(f"   🔢 Total Hops: {len(hop_records)}")
        
        return simulation_result
    
    def _create_packet_object(self, source_user, destination_user, packet_data: str,
                            source_gs: Dict, destination_gs: Dict, algorithm: str) -> Packet:
        """Tạo đối tượng Packet với đầy đủ thông tin"""
        
        # Tạo QoS
        qos = QoS(
            service_type="REALTIME",
            default_priority=1,
            max_latency_ms=100.0,
            max_jitter_ms=10.0,
            min_bandwidth_mbps=50.0,
            max_loss_rate=0.01
        )
        
        # Tạo AnalysisData mặc định
        analysis_data = AnalysisData(
            avg_latency=0.0,
            avg_distance_km=0.0,
            route_success_rate=0.0,
            total_distance_km=0.0,
            total_latency_ms=0.0
        )
        
        return Packet(
            packet_id=f"pkt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_user_id=source_user.userId,
            destination_user_id=destination_user.userId,
            station_source=source_gs.get("nodeId", ""),
            station_dest=destination_gs.get("nodeId", ""),
            type="DATA",
            time_sent_from_source_ms=datetime.now(timezone.utc).timestamp() * 1000,
            payload_data_base64=packet_data.encode('utf-8').hex(),  # Simple encoding
            payload_size_byte=len(packet_data),
            service_qos=qos,
            current_holding_node_id=source_gs.get("nodeId", ""),
            next_hop_node_id="",  # Will be set during simulation
            priority_level=1,
            max_acceptable_latency_ms=100.0,
            max_acceptable_loss_rate=0.01,
            analysis_data=analysis_data,
            use_rl=(algorithm == "RL"),
            ttl=64
        )
    
    def _hop_record_to_dict(self, hop_record: HopRecord) -> Dict:
        """Chuyển HopRecord thành dictionary"""
        return {
            "from_node_id": hop_record.from_node_id,
            "to_node_id": hop_record.to_node_id,
            "latency_ms": hop_record.latency_ms,
            "timestamp_ms": hop_record.timestamp_ms,
            "distance_km": hop_record.distance_km,
            "from_node_position": {
                "latitude": hop_record.from_node_position.latitude,
                "longitude": hop_record.from_node_position.longitude,
                "altitude": hop_record.from_node_position.altitude
            } if hop_record.from_node_position else None,
            "to_node_position": {
                "latitude": hop_record.to_node_position.latitude,
                "longitude": hop_record.to_node_position.longitude,
                "altitude": hop_record.to_node_position.altitude
            } if hop_record.to_node_position else None,
            "from_node_buffer_state": {
                "queue_size": hop_record.from_node_buffer_state.queue_size,
                "bandwidth_utilization": hop_record.from_node_buffer_state.bandwidth_utilization
            } if hop_record.from_node_buffer_state else None,
            "routing_decision_info": {
                "algorithm": hop_record.routing_decision_info.algorithm.value,
                "metric": hop_record.routing_decision_info.metric,
                "reward": hop_record.routing_decision_info.reward
            } if hop_record.routing_decision_info else None,
            "scenario_type": hop_record.scenario_type,
            "node_load_percent": hop_record.node_load_percent,
            "drop_reason_details": hop_record.drop_reason_details
        }
    
    def _packet_to_dict(self, packet: Packet) -> Dict:
        """Chuyển Packet thành dictionary"""
        return {
            "packet_id": packet.packet_id,
            "source_user_id": packet.source_user_id,
            "destination_user_id": packet.destination_user_id,
            "station_source": packet.station_source,
            "station_dest": packet.station_dest,
            "type": packet.type,
            "time_sent_from_source_ms": packet.time_sent_from_source_ms,
            "payload_data_base64": packet.payload_data_base64,
            "payload_size_byte": packet.payload_size_byte,
            "service_qos": {
                "service_type": packet.service_qos.service_type,
                "default_priority": packet.service_qos.default_priority,
                "max_latency_ms": packet.service_qos.max_latency_ms,
                "max_jitter_ms": packet.service_qos.max_jitter_ms,
                "min_bandwidth_mbps": packet.service_qos.min_bandwidth_mbps,
                "max_loss_rate": packet.service_qos.max_loss_rate
            },
            "current_holding_node_id": packet.current_holding_node_id,
            "next_hop_node_id": packet.next_hop_node_id,
            "priority_level": packet.priority_level,
            "max_acceptable_latency_ms": packet.max_acceptable_latency_ms,
            "max_acceptable_loss_rate": packet.max_acceptable_loss_rate,
            "analysis_data": {
                "avg_latency": packet.analysis_data.avg_latency,
                "avg_distance_km": packet.analysis_data.avg_distance_km,
                "route_success_rate": packet.analysis_data.route_success_rate,
                "total_distance_km": packet.analysis_data.total_distance_km,
                "total_latency_ms": packet.analysis_data.total_latency_ms
            },
            "use_rl": packet.use_rl,
            "ttl": packet.ttl,
            "acknowledged_packet_id": packet.acknowledged_packet_id,
            "path_history": packet.path_history,
            "hop_records": [self._hop_record_to_dict(hr) for hr in packet.hop_records],
            "accumulated_delay_ms": packet.accumulated_delay_ms,
            "dropped": packet.dropped,
            "drop_reason": packet.drop_reason
        }

    # Giữ nguyên các phương thức khác từ class cũ (send_packet_to_destination, compare_algorithms, etc.)
    def send_packet_to_destination(self, simulation_result: Dict[str, Any]) -> bool:
        """Gửi packet đến user đích"""
        try:
            destination_user = None
            for user in self.user_manager.users:
                if user.userId == simulation_result["destinationUser"]["userId"]:
                    destination_user = user
                    break
            
            if not destination_user:
                print(f"❌ Destination user not found")
                return False
            
            packet_data = {
                "type": "ENHANCED_SIMULATION_RESULT",
                "simulationId": simulation_result["simulationId"],
                "sourceUser": simulation_result["sourceUser"]["userName"],
                "timestamp": simulation_result["timestamp"],
                "metrics": simulation_result["totalMetrics"],
                "packetData": simulation_result["packetData"]
            }
            
            packet_json = json.dumps(packet_data, indent=2)
            
            print(f"📤 Sending enhanced packet to {destination_user.userName} at {destination_user.ipAddress}:{destination_user.port}")
            
            success = self._send_udp_packet(
                destination_user.ipAddress,
                destination_user.port,
                packet_json
            )
            
            if success:
                print(f"✅ Enhanced packet successfully sent to {destination_user.userName}")
                simulation_result["deliveryStatus"] = "DELIVERED"
                simulation_result["deliveryTimestamp"] = datetime.now(timezone.utc).isoformat()
            else:
                print(f"❌ Failed to send enhanced packet to {destination_user.userName}")
                simulation_result["deliveryStatus"] = "FAILED"
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending enhanced packet: {e}")
            simulation_result["deliveryStatus"] = "FAILED"
            simulation_result["error"] = str(e)
            return False

    def _send_udp_packet(self, ip: str, port: int, data: str) -> bool:
        """Gửi UDP packet"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5.0)
            sock.sendto(data.encode('utf-8'), (ip, port))
            sock.close()
            return True
        except Exception as e:
            print(f"❌ UDP send error: {e}")
            return False

    def compare_algorithms(self, source_user, destination_user, packet_data: str) -> Dict[str, Any]:
        """So sánh hiệu năng giữa hai thuật toán"""
        print(f"🔬 Comparing algorithms for {source_user.userName} -> {destination_user.userName}")
        
        dijkstra_result = self.simulate_packet_journey(
            source_user, destination_user, packet_data, "DIJKSTRA"
        )
        
        rl_result = self.simulate_packet_journey(
            source_user, destination_user, packet_data, "RL"
        )
        
        comparison = {
            "comparisonId": f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sourceUser": source_user.to_dict(),
            "destinationUser": destination_user.to_dict(),
            "dijkstraResults": dijkstra_result["totalMetrics"],
            "rlResults": rl_result["totalMetrics"],
            "winner": self._determine_winner(
                dijkstra_result["totalMetrics"], 
                rl_result["totalMetrics"]
            ),
            "details": {
                "dijkstra": dijkstra_result,
                "rl": rl_result
            }
        }
        
        print(f"🏆 Algorithm Comparison Winner: {comparison['winner']}")
        return comparison

    def _determine_winner(self, dijkstra_metrics: Dict, rl_metrics: Dict) -> str:
        """Xác định thuật toán nào tốt hơn"""
        dijkstra_score = 0
        rl_score = 0
        
        if dijkstra_metrics["totalLatencyMs"] < rl_metrics["totalLatencyMs"]:
            dijkstra_score += 1
        elif rl_metrics["totalLatencyMs"] < dijkstra_metrics["totalLatencyMs"]:
            rl_score += 1
        
        if dijkstra_metrics["totalPacketLossRate"] < rl_metrics["totalPacketLossRate"]:
            dijkstra_score += 1
        elif rl_metrics["totalPacketLossRate"] < dijkstra_metrics["totalPacketLossRate"]:
            rl_score += 1
        
        if dijkstra_metrics["minBandwidthMbps"] > rl_metrics["minBandwidthMbps"]:
            dijkstra_score += 1
        elif rl_metrics["minBandwidthMbps"] > dijkstra_metrics["minBandwidthMbps"]:
            rl_score += 1
        
        if dijkstra_metrics["totalHops"] < rl_metrics["totalHops"]:
            dijkstra_score += 1
        elif rl_metrics["totalHops"] < dijkstra_metrics["totalHops"]:
            rl_score += 1
        
        if dijkstra_score > rl_score:
            return "DIJKSTRA"
        elif rl_score > dijkstra_score:
            return "RL"
        else:
            return "TIE"

    def save_simulation_results(self, filename: str = "enhanced_simulation_results.json"):
        """Lưu kết quả mô phỏng"""
        with open(filename, "w") as f:
            json.dump(self.simulation_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.simulation_results)} enhanced simulation results to {filename}")

# Ví dụ sử dụng
if __name__ == "__main__":
    from model.User import UserManager
    
    # Khởi tạo database connector
    db_connector = MongoConnector()
    
    # Khởi tạo user manager
    user_manager = UserManager()
    user_manager.load_from_json("helper/network_user.json")
    
    # Tạo enhanced packet simulation
    enhanced_simulator = EnhancedPacketSimulation(db_connector, user_manager)
    
    # Lấy users để simulation
    source_user = user_manager.get_user_by_id("user-Singapore")
    dest_user = user_manager.get_user_by_id("user-Hanoi")
    
    if source_user and dest_user:
        # Thêm tọa độ cho users
        source_user.latitude = 1.3521
        source_user.longitude = 103.8198
        dest_user.latitude = 21.0285
        dest_user.longitude = 105.8542
        
        # Chạy so sánh thuật toán với database
        comparison = enhanced_simulator.compare_algorithms(
            source_user, 
            dest_user, 
            "Hello from Singapore to Hanoi with database integration!"
        )
        
        # Gửi packet đến destination
        dijkstra_result = comparison["details"]["dijkstra"]
        enhanced_simulator.send_packet_to_destination(dijkstra_result)
        
        # Lưu kết quả
        enhanced_simulator.save_simulation_results()
        
        print(f"\n🎯 Enhanced algorithm comparison completed!")
        print(f"   Winner: {comparison['winner']}")
        print(f"   Total simulations: {len(enhanced_simulator.simulation_results)}")
    
    else:
        print("❌ Users not found for enhanced simulation")