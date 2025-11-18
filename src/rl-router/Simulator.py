import json
import socket
import random
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from model.Node import Node, NodeManager
from model.User import User, UserManager

class HopRecord:
    """Lớp ghi lại thông tin của từng hop trong đường đi"""
    
    def __init__(self, hop_number: int, from_node_id: str, to_node_id: str, 
                 node_type: str, latency_ms: float, bandwidth_mbps: float,
                 packet_loss_rate: float, distance_km: float, 
                 processing_delay_ms: float, propagation_delay_ms: float):
        
        self.hopNumber = hop_number
        self.fromNodeId = from_node_id
        self.toNodeId = to_node_id
        self.nodeType = node_type
        self.latencyMs = latency_ms
        self.bandwidthMbps = bandwidth_mbps
        self.packetLossRate = packet_loss_rate
        self.distanceKm = distance_km
        self.processingDelayMs = processing_delay_ms
        self.propagationDelayMs = propagation_delay_ms
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hopNumber": self.hopNumber,
            "fromNodeId": self.fromNodeId,
            "toNodeId": self.toNodeId,
            "nodeType": self.nodeType,
            "latencyMs": self.latencyMs,
            "bandwidthMbps": self.bandwidthMbps,
            "packetLossRate": self.packetLossRate,
            "distanceKm": self.distanceKm,
            "processingDelayMs": self.processingDelayMs,
            "propagationDelayMs": self.propagationDelayMs,
            "timestamp": self.timestamp
        }

class PacketSimulation:
    """Lớp mô phỏng việc gửi packet qua network"""
    
    # Constants
    SPEED_OF_LIGHT = 299792  # km/s
    PROCESSING_DELAY_GROUND = 2.0  # ms
    PROCESSING_DELAY_SATELLITE = 5.0  # ms
    
    def __init__(self, node_manager: NodeManager, user_manager: UserManager):
        self.node_manager = node_manager
        self.user_manager = user_manager
        self.simulation_results = []
    
    def find_nearest_ground_station(self, user: User) -> Optional[Node]:
        """Tìm ground station gần user nhất dựa trên vị trí địa lý"""
        nearest_gs = None
        min_distance = float('inf')
        
        user_lat = getattr(user, 'latitude', 0)
        user_lon = getattr(user, 'longitude', 0)
        
        for node in self.node_manager.nodes:
            if node.nodeType == "GROUND_STATION" and node.isOperational:
                # Tính khoảng cách đơn giản dựa trên tọa độ
                distance = self._calculate_distance(
                    user_lat, user_lon, 
                    node.position.latitude, node.position.longitude
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_gs = node
        
        return nearest_gs
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Tính khoảng cách Great-circle giữa hai điểm (đơn giản hóa)"""
        # Sử dụng công thức Haversine đơn giản hóa
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
    
    def find_optimal_path(self, source_gs: Node, destination_gs: Node, 
                         algorithm: str = "DIJKSTRA") -> List[Node]:
        """
        Tìm đường đi tối ưu giữa hai ground station
        algorithm: "DIJKSTRA" hoặc "RL"
        """
        # Đây là implementation đơn giản
        # Trong thực tế, bạn sẽ sử dụng Dijkstra hoặc RL thực sự
        
        if algorithm == "DIJKSTRA":
            return self._dijkstra_path(source_gs, destination_gs)
        else:  # RL
            return self._rl_path(source_gs, destination_gs)
    
    def _dijkstra_path(self, source: Node, destination: Node) -> List[Node]:
        """Tìm đường đi sử dụng Dijkstra algorithm (đơn giản hóa)"""
        # Implementation đơn giản - trong thực tế cần đồ thị đầy đủ
        path = [source]
        
        # Thêm một vài satellite ngẫu nhiên làm trung gian
        satellites = [node for node in self.node_manager.nodes 
                     if node.nodeType.endswith("SATELLITE") and node.isOperational]
        
        if satellites:
            # Chọn 1-2 satellite ngẫu nhiên
            num_intermediate = min(2, len(satellites))
            intermediate_sats = random.sample(satellites, num_intermediate)
            path.extend(intermediate_sats)
        
        path.append(destination)
        return path
    
    def _rl_path(self, source: Node, destination: Node) -> List[Node]:
        """Tìm đường đi sử dụng RL algorithm (đơn giản hóa)"""
        # Implementation đơn giản - trong thực tế sẽ có model RL
        path = [source]
        
        # Ưu tiên các satellite có resource utilization thấp
        satellites = [node for node in self.node_manager.nodes 
                     if node.nodeType.endswith("SATELLITE") and node.isOperational]
        
        if satellites:
            # Sắp xếp theo resource utilization (thấp nhất đầu tiên)
            satellites.sort(key=lambda x: x.resourceUtilization)
            num_intermediate = min(2, len(satellites))
            intermediate_sats = satellites[:num_intermediate]
            path.extend(intermediate_sats)
        
        path.append(destination)
        return path
    
    def calculate_hop_metrics(self, from_node: Node, to_node: Node) -> Tuple[float, float, float, float, float, float]:
        """Tính toán metrics cho một hop giữa hai node"""
        # Tính khoảng cách
        distance = from_node.distance_to(to_node)
        
        # Tính propagation delay (tốc độ ánh sáng)
        propagation_delay = (distance / self.SPEED_OF_LIGHT) * 1000  # ms
        
        # Processing delay dựa trên loại node
        if from_node.nodeType == "GROUND_STATION":
            processing_delay = self.PROCESSING_DELAY_GROUND
        else:
            processing_delay = self.PROCESSING_DELAY_SATELLITE
        
        # Tổng latency
        total_latency = propagation_delay + processing_delay
        
        # Bandwidth - lấy giá trị nhỏ nhất
        bandwidth = min(from_node.communication.bandwidthMHz, 
                       to_node.communication.bandwidthMHz)
        
        # Packet loss rate - lấy giá trị trung bình
        packet_loss = (from_node.packetLossRate + to_node.packetLossRate) / 2
        
        return total_latency, bandwidth, packet_loss, distance, processing_delay, propagation_delay
    
    def simulate_packet_journey(self, source_user: User, destination_user: User, 
                               packet_data: str, algorithm: str = "DIJKSTRA") -> Dict[str, Any]:
        """
        Mô phỏng hành trình của packet từ user nguồn đến user đích
        """
        print(f"🚀 Starting packet simulation: {source_user.userName} -> {destination_user.userName} ({algorithm})")
        
        # Tìm ground station gần nhất cho source và destination
        source_gs = self.find_nearest_ground_station(source_user)
        destination_gs = self.find_nearest_ground_station(destination_user)
        
        if not source_gs or not destination_gs:
            raise Exception("Cannot find suitable ground stations")
        
        print(f"📍 Source GS: {source_gs.nodeName}, Destination GS: {destination_gs.nodeName}")
        
        # Tìm đường đi tối ưu
        path_nodes = self.find_optimal_path(source_gs, destination_gs, algorithm)
        
        # Mô phỏng packet đi qua từng hop
        hop_records = []
        total_latency = 0
        total_packet_loss = 0
        min_bandwidth = float('inf')
        
        print(f"🛣️  Path found ({len(path_nodes)} hops):")
        for i, node in enumerate(path_nodes):
            print(f"    {i+1}. {node.nodeName} ({node.nodeType})")
        
        # Mô phỏng từng hop
        for i in range(len(path_nodes) - 1):
            from_node = path_nodes[i]
            to_node = path_nodes[i + 1]
            
            # Tính toán metrics cho hop này
            latency, bandwidth, packet_loss, distance, processing_delay, propagation_delay = \
                self.calculate_hop_metrics(from_node, to_node)
            
            # Tạo hop record
            hop_record = HopRecord(
                hop_number=i + 1,
                from_node_id=from_node.nodeId,
                to_node_id=to_node.nodeId,
                node_type=to_node.nodeType,
                latency_ms=latency,
                bandwidth_mbps=bandwidth,
                packet_loss_rate=packet_loss,
                distance_km=distance,
                processing_delay_ms=processing_delay,
                propagation_delay_ms=propagation_delay
            )
            
            hop_records.append(hop_record)
            total_latency += latency
            total_packet_loss += packet_loss
            min_bandwidth = min(min_bandwidth, bandwidth)
            
            print(f"    🔄 Hop {i+1}: {from_node.nodeName} -> {to_node.nodeName}")
            print(f"       📊 Latency: {latency:.2f}ms, Bandwidth: {bandwidth:.2f}MHz, Distance: {distance:.2f}km")
        
        # Tính packet loss tổng (xác suất tích lũy)
        total_packet_loss_rate = 1 - (1 - total_packet_loss / len(hop_records)) ** len(hop_records)
        
        # Tạo kết quả mô phỏng
        simulation_result = {
            "simulationId": f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sourceUser": source_user.to_dict(),
            "destinationUser": destination_user.to_dict(),
            "algorithm": algorithm,
            "path": [node.nodeId for node in path_nodes],
            "hopRecords": [hop.to_dict() for hop in hop_records],
            "totalMetrics": {
                "totalLatencyMs": total_latency,
                "totalPacketLossRate": total_packet_loss_rate,
                "minBandwidthMbps": min_bandwidth,
                "totalHops": len(hop_records),
                "totalDistanceKm": sum(hop.distanceKm for hop in hop_records)
            },
            "packetData": packet_data,
            "status": "COMPLETED"
        }
        
        self.simulation_results.append(simulation_result)
        
        print(f"✅ Simulation completed:")
        print(f"   📈 Total Latency: {total_latency:.2f}ms")
        print(f"   📉 Packet Loss Rate: {total_packet_loss_rate:.4f}")
        print(f"   📊 Min Bandwidth: {min_bandwidth:.2f}MHz")
        print(f"   🔢 Total Hops: {len(hop_records)}")
        
        return simulation_result
    
    def send_packet_to_destination(self, simulation_result: Dict[str, Any]) -> bool:
        """
        Gửi packet đến user đích dựa trên thông tin từ database
        """
        try:
            destination_user = None
            for user in self.user_manager.users:
                if user.userId == simulation_result["destinationUser"]["userId"]:
                    destination_user = user
                    break
            
            if not destination_user:
                print(f"❌ Destination user not found")
                return False
            
            # Tạo packet data để gửi
            packet_data = {
                "type": "SIMULATION_RESULT",
                "simulationId": simulation_result["simulationId"],
                "sourceUser": simulation_result["sourceUser"]["userName"],
                "timestamp": simulation_result["timestamp"],
                "metrics": simulation_result["totalMetrics"],
                "originalData": simulation_result["packetData"]
            }
            
            # Chuyển đổi thành JSON string
            packet_json = json.dumps(packet_data, indent=2)
            
            print(f"📤 Sending packet to {destination_user.userName} at {destination_user.ipAddress}:{destination_user.port}")
            
            # Gửi packet sử dụng socket (UDP)
            success = self._send_udp_packet(
                destination_user.ipAddress,
                destination_user.port,
                packet_json
            )
            
            if success:
                print(f"✅ Packet successfully sent to {destination_user.userName}")
                simulation_result["deliveryStatus"] = "DELIVERED"
                simulation_result["deliveryTimestamp"] = datetime.now(timezone.utc).isoformat()
            else:
                print(f"❌ Failed to send packet to {destination_user.userName}")
                simulation_result["deliveryStatus"] = "FAILED"
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending packet: {e}")
            simulation_result["deliveryStatus"] = "FAILED"
            simulation_result["error"] = str(e)
            return False
    
    def _send_udp_packet(self, ip: str, port: int, data: str) -> bool:
        """
        Gửi UDP packet đến địa chỉ đích
        """
        try:
            # Tạo UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5.0)  # 5 seconds timeout
            
            # Gửi data
            sock.sendto(data.encode('utf-8'), (ip, port))
            
            # Đóng socket
            sock.close()
            return True
            
        except Exception as e:
            print(f"❌ UDP send error: {e}")
            return False
    
    def compare_algorithms(self, source_user: User, destination_user: User, 
                          packet_data: str) -> Dict[str, Any]:
        """
        So sánh hiệu năng giữa hai thuật toán DIJKSTRA và RL
        """
        print(f"🔬 Comparing algorithms for {source_user.userName} -> {destination_user.userName}")
        
        # Chạy mô phỏng cho cả hai thuật toán
        dijkstra_result = self.simulate_packet_journey(
            source_user, destination_user, packet_data, "DIJKSTRA"
        )
        
        rl_result = self.simulate_packet_journey(
            source_user, destination_user, packet_data, "RL"
        )
        
        # So sánh kết quả
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
        """Xác định thuật toán nào tốt hơn dựa trên multiple metrics"""
        dijkstra_score = 0
        rl_score = 0
        
        # So sánh latency (thấp hơn tốt hơn)
        if dijkstra_metrics["totalLatencyMs"] < rl_metrics["totalLatencyMs"]:
            dijkstra_score += 1
        elif rl_metrics["totalLatencyMs"] < dijkstra_metrics["totalLatencyMs"]:
            rl_score += 1
        
        # So sánh packet loss (thấp hơn tốt hơn)
        if dijkstra_metrics["totalPacketLossRate"] < rl_metrics["totalPacketLossRate"]:
            dijkstra_score += 1
        elif rl_metrics["totalPacketLossRate"] < dijkstra_metrics["totalPacketLossRate"]:
            rl_score += 1
        
        # So sánh bandwidth (cao hơn tốt hơn)
        if dijkstra_metrics["minBandwidthMbps"] > rl_metrics["minBandwidthMbps"]:
            dijkstra_score += 1
        elif rl_metrics["minBandwidthMbps"] > dijkstra_metrics["minBandwidthMbps"]:
            rl_score += 1
        
        # So sánh số hop (ít hơn tốt hơn)
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
    
    def save_simulation_results(self, filename: str = "simulation_results.json"):
        """Lưu kết quả mô phỏng vào file JSON"""
        with open(filename, "w") as f:
            json.dump(self.simulation_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.simulation_results)} simulation results to {filename}")
    
    def print_simulation_summary(self):
        """In tổng quan kết quả mô phỏng"""
        print("\n" + "="*50)
        print("📊 SIMULATION SUMMARY")
        print("="*50)
        
        for i, result in enumerate(self.simulation_results):
            print(f"\n{i+1}. {result['simulationId']}")
            print(f"   From: {result['sourceUser']['userName']}")
            print(f"   To: {result['destinationUser']['userName']}")
            print(f"   Algorithm: {result['algorithm']}")
            print(f"   Latency: {result['totalMetrics']['totalLatencyMs']:.2f}ms")
            print(f"   Packet Loss: {result['totalMetrics']['totalPacketLossRate']:.4f}")
            print(f"   Bandwidth: {result['totalMetrics']['minBandwidthMbps']:.2f}MHz")
            print(f"   Hops: {result['totalMetrics']['totalHops']}")

# UDP Server để nhận packet
class PacketReceiver:
    """UDP Server để nhận packet từ các simulation"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        self.host = host
        self.port = port
        self.received_packets = []
        self.is_running = False
        self.server_socket = None
    
    def start_server(self):
        """Khởi động UDP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((self.host, self.port))
            self.is_running = True
            
            print(f"📡 UDP Server started on {self.host}:{self.port}")
            
            while self.is_running:
                try:
                    data, addr = self.server_socket.recvfrom(1024)
                    packet_data = data.decode('utf-8')
                    
                    # Xử lý packet nhận được
                    self._handle_packet(packet_data, addr)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"❌ Server error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
    
    def _handle_packet(self, packet_data: str, addr: tuple):
        """Xử lý packet nhận được"""
        try:
            packet_dict = json.loads(packet_data)
            packet_dict["receivedAt"] = datetime.now(timezone.utc).isoformat()
            packet_dict["sourceAddress"] = addr[0]
            packet_dict["sourcePort"] = addr[1]
            
            self.received_packets.append(packet_dict)
            
            print(f"📥 Received packet from {addr[0]}:{addr[1]}")
            print(f"   Type: {packet_dict.get('type', 'UNKNOWN')}")
            print(f"   Simulation ID: {packet_dict.get('simulationId', 'N/A')}")
            
        except json.JSONDecodeError:
            print(f"📥 Received raw packet: {packet_data[:100]}...")
    
    def stop_server(self):
        """Dừng UDP server"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        print("🛑 UDP Server stopped")
    
    def get_received_packets(self) -> List[Dict]:
        """Lấy danh sách packet đã nhận"""
        return self.received_packets

# Ví dụ sử dụng
if __name__ == "__main__":
    import random
    
    # Khởi tạo managers
    node_manager = NodeManager()
    user_manager = UserManager()
    
    # Load dữ liệu mẫu (cần có files trước)
    try:
        node_manager.load_from_json("helper/network_nodes.json")
        user_manager.load_from_json("helper/network_user.json")
    except:
        print("⚠️ Could not load data files, creating sample data...")
        # Tạo dữ liệu mẫu nếu files không tồn tại
        from model.Node import generate_sample_network

        # Tạo nodes
        nodes = generate_sample_network()
        node_manager.nodes = nodes
        
        # Tạo users
        user_manager.create_user("Singapore", "127.0.0.1", 10000, "user-Singapore", "User_Singapore")
        user_manager.create_user("Hanoi", "127.0.0.1", 10001, "user-Hanoi", "User_Hanoi")
        user_manager.create_user("Tokyo", "127.0.0.1", 10002, "user-Tokyo", "User_Tokyo")
    
    # Khởi tạo packet simulation
    simulator = PacketSimulation(node_manager, user_manager)
    
    # Lấy users để simulation
    source_user = user_manager.get_user_by_id("user-Singapore")
    dest_user = user_manager.get_user_by_id("user-Hanoi")
    
    if source_user and dest_user:
        # Thêm tọa độ cho users (giả lập)
        source_user.latitude = 1.3521
        source_user.longitude = 103.8198
        dest_user.latitude = 21.0285
        dest_user.longitude = 105.8542
        
        # Chạy so sánh thuật toán
        comparison = simulator.compare_algorithms(
            source_user, 
            dest_user, 
            "Hello from Singapore to Hanoi!"
        )
        
        # Gửi packet đến destination
        dijkstra_result = comparison["details"]["dijkstra"]
        simulator.send_packet_to_destination(dijkstra_result)
        
        # In summary
        simulator.print_simulation_summary()
        
        # Lưu kết quả
        simulator.save_simulation_results()
        
        print(f"\n🎯 Algorithm comparison completed!")
        print(f"   Winner: {comparison['winner']}")
        print(f"   Total simulations: {len(simulator.simulation_results)}")
    
    else:
        print("❌ Users not found for simulation")