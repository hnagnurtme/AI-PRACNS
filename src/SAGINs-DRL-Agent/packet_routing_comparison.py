# test_rl_vs_dijkstra.py
import torch
import numpy as np
import time
from typing import List, Dict, Tuple
import heapq
from data.mongo_manager import MongoDataManager
from env.link_metrics_calculator import LinkMetricsCalculator
from env.state_processor import StateProcessor
from env.action_mapper import ActionMapper
from agents.dpn_agent import DqnAgent

class DijkstraRouter:
    """Dijkstra algorithm for shortest path routing"""
    
    def __init__(self, mongo_manager: MongoDataManager, link_calculator: LinkMetricsCalculator):
        self.mongo_manager = mongo_manager
        self.link_calculator = link_calculator
        self.nodes = {}
        self._load_network()
    
    def _load_network(self):
        """Load network topology từ MongoDB"""
        snapshot = self.mongo_manager.get_training_snapshot()
        self.nodes = snapshot.get('nodes', {})
        print(f"📡 Dijkstra loaded {len(self.nodes)} nodes")
    
    def calculate_shortest_path(self, source: str, destination: str) -> Tuple[List[str], float]:
        """Tính shortest path using Dijkstra"""
        if source not in self.nodes or destination not in self.nodes:
            return [], float('inf')
        
        # Khởi tạo distances và predecessors
        distances = {node: float('inf') for node in self.nodes}
        predecessors = {node: None for node in self.nodes}
        distances[source] = 0
        
        # Priority queue
        pq = [(0, source)]
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # Nếu đã đến destination
            if current_node == destination:
                break
                
            # Nếu distance hiện tại lớn hơn distance đã biết, bỏ qua
            if current_distance > distances[current_node]:
                continue
            
            # Duyệt qua tất cả neighbors
            for neighbor_id, neighbor_node in self.nodes.items():
                if neighbor_id == current_node:
                    continue
                
                # Tính link cost (dựa trên latency)
                link_metrics = self.link_calculator.calculate_link_metrics(
                    self.nodes[current_node], neighbor_node
                )
                
                if not link_metrics.get('isLinkActive', True):
                    continue
                
                # Cost = latency (ms)
                cost = link_metrics.get('latencyMs', 1000.0)
                distance = current_distance + cost
                
                # Nếu tìm được đường ngắn hơn
                if distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    predecessors[neighbor_id] = current_node
                    heapq.heappush(pq, (distance, neighbor_id))
        
        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        path.reverse()
        
        # Nếu không tìm được path
        if path[0] != source:
            return [], float('inf')
        
        return path, distances[destination]

class RLRouter:
    """RL-based router using trained model"""
    
    def __init__(self, mongo_manager: MongoDataManager, model_path: str):
        self.mongo_manager = mongo_manager
        self.model_path = model_path
        
        # Khởi tạo components
        self.link_calculator = LinkMetricsCalculator()
        self.state_processor = StateProcessor(max_neighbors=10)
        self.action_mapper = ActionMapper(mongo_manager)
        
        # Load trained model
        self.agent = self._load_trained_model()
        self.agent.action_mapper = self.action_mapper
        
        print(f"🧠 RL Router loaded from: {model_path}")
    
    def _load_trained_model(self) -> DqnAgent:
        """Load trained DQN model"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        state_size = self.state_processor.get_state_size()
        action_size = self.action_mapper.get_action_size()
        
        agent = DqnAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,  # Not used in inference
            gamma=0.99
        )
        
        # Load trained weights
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.policy_net.eval()  # Set to evaluation mode
        
        # Set epsilon to 0 for pure exploitation
        agent.epsilon = 0.0
        
        return agent
    
    def find_rl_path(self, source: str, destination: str, max_hops: int = 8) -> Tuple[List[str], float]:
        """Tìm path using RL agent với improved loop prevention"""
        if source not in self.action_mapper.get_available_nodes() or destination not in self.action_mapper.get_available_nodes():
            return [], float('inf')
        
        path = [source]
        total_latency = 0.0
        current_node = source
        visited_nodes = set([source])  # Track visited nodes
        
        print(f"🧭 RL Routing: {source} → {destination}")
        
        for hop in range(max_hops):
            if current_node == destination:
                print(f"🎉 RL SUCCESS: Reached destination in {hop} hops")
                break
            
            try:
                # Tạo state
                state_data = self._create_state_data(current_node, destination)
                state_vector = self.state_processor.json_to_state_vector(state_data)
                
                # Lấy available neighbors (tránh loops)
                available_neighbors = self._get_available_neighbors(current_node, visited_nodes)
                
                if not available_neighbors:
                    print(f"❌ No available neighbors from {current_node}")
                    break
                
                # RL agent chọn next hop từ available neighbors
                next_hop = self._select_best_rl_action(state_vector, available_neighbors)
                
                if not next_hop:
                    print(f"❌ No valid RL action from {current_node}")
                    break
                
                # Tính latency của link này
                snapshot = self.mongo_manager.get_training_snapshot()
                current_node_data = snapshot['nodes'][current_node]
                next_node_data = snapshot['nodes'][next_hop]
                
                link_metrics = self.link_calculator.calculate_link_metrics(
                    current_node_data, next_node_data
                )
                
                if not link_metrics.get('isLinkActive', True):
                    print(f"❌ Link {current_node} -> {next_hop} is inactive")
                    # Đánh dấu node này không available
                    visited_nodes.add(next_hop)
                    continue  # Thử lại với node khác
                
                link_latency = link_metrics.get('latencyMs', 1000.0)
                total_latency += link_latency
                
                # Cập nhật path và visited nodes
                path.append(next_hop)
                visited_nodes.add(next_hop)
                current_node = next_hop
                
                print(f"   Hop {hop+1}: {path[-2]} → {next_hop} (latency: {link_latency:.2f}ms)")
                
                # Early success check
                if next_hop == destination:
                    print(f"🎉 RL SUCCESS: Reached destination in {hop+1} hops")
                    break
                    
            except Exception as e:
                print(f"❌ Error in RL routing at hop {hop}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Kiểm tra kết quả
        success = (path[-1] == destination)
        if success:
            print(f"✅ RL COMPLETED: {source} → {destination} in {len(path)-1} hops")
            print(f"   Total Latency: {total_latency:.2f}ms, Path: {' → '.join(path)}")
        else:
            print(f"❌ RL FAILED: Stopped at {path[-1]}")
            total_latency = float('inf')
        
        return path, total_latency

    def _get_available_neighbors(self, current_node: str, visited_nodes: set) -> List[str]:
        """Lấy danh sách neighbors có thể kết nối, tránh visited nodes"""
        snapshot = self.mongo_manager.get_training_snapshot()
        nodes = snapshot['nodes']
        current_node_data = nodes[current_node]
        
        available_neighbors = []
        
        for neighbor_id, neighbor_node in nodes.items():
            if neighbor_id == current_node or neighbor_id in visited_nodes:
                continue
                
            # Kiểm tra link có active không
            link_metrics = self.link_calculator.calculate_link_metrics(
                current_node_data, neighbor_node
            )
            
            if link_metrics.get('isLinkActive', True):
                available_neighbors.append(neighbor_id)
        
        return available_neighbors

    def _select_best_rl_action(self, state_vector: np.ndarray, available_nodes: List[str]) -> str:
        """Chọn action tốt nhất từ available nodes"""
        if not available_nodes:
            return None
        
        # Nếu chỉ có 1 node available, chọn nó
        if len(available_nodes) == 1:
            return available_nodes[0]
        
        # Dùng RL để chọn từ available nodes
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.agent.policy_net(state_tensor)
        
        # Tìm node có Q-value cao nhất trong available nodes
        best_value = -float('inf')
        best_node = None
        
        for node_id in available_nodes:
            node_index = self.action_mapper.get_action_index(node_id)
            if node_index is not None and q_values[0, node_index] > best_value:
                best_value = q_values[0, node_index]
                best_node = node_id
        
        return best_node
    
    def _create_state_data(self, source: str, destination: str) -> Dict:
        """Tạo state data cho RL agent"""
        snapshot = self.mongo_manager.get_training_snapshot()
        nodes = snapshot['nodes']
        
        source_node = nodes[source]
        dest_node = nodes[destination]
        
        # Tính neighbor links
        neighbor_links = {}
        for neighbor_id, neighbor_node in nodes.items():
            if neighbor_id != source:
                link_metrics = self.link_calculator.calculate_link_metrics(source_node, neighbor_node)
                if link_metrics['isLinkActive']:
                    neighbor_links[neighbor_id] = link_metrics
        
        return {
            "sourceNodeId": source,
            "destinationNodeId": destination,
            "targetQoS": {
                "serviceType": "VIDEO_STREAMING",
                "maxLatencyMs": 50.0,
                "minBandwidthMbps": 500.0,
                "maxLossRate": 0.02
            },
            "sourceNodeInfo": source_node,
            "destinationNodeInfo": dest_node,
            "neighborLinkMetrics": neighbor_links
        }

class PacketSender:
    """Simulate packet sending between ground/sea stations"""
    
    def __init__(self, mongo_manager: MongoDataManager):
        self.mongo_manager = mongo_manager
        self.dijkstra_router = DijkstraRouter(mongo_manager, LinkMetricsCalculator())
        
        # Tìm model mới nhất
        self.rl_router = self._find_latest_rl_model()
    
    def _find_latest_rl_model(self) -> RLRouter:
        """Tìm RL model mới nhất"""
        import glob
        import os
        
        model_patterns = [
            "models/best_model_*.pth",
            "models/latest_checkpoint.pth",
            "models/checkpoint_final_*.pth"
        ]
        
        latest_model = None
        latest_time = 0
        
        for pattern in model_patterns:
            for model_path in glob.glob(pattern):
                mtime = os.path.getmtime(model_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_model = model_path
        
        if latest_model:
            return RLRouter(self.mongo_manager, latest_model)
        else:
            raise FileNotFoundError("No trained RL model found!")
    
    def send_packet(self, source_station: str, destination_station: str):
        """Gửi packet từ source đến destination và so sánh RL vs Dijkstra"""
        print(f"\n📦 SENDING PACKET: {source_station} → {destination_station}")
        print("=" * 60)
        
        # Test Dijkstra
        print("🧮 DIJKSTRA ROUTING:")
        start_time = time.time()
        dijkstra_path, dijkstra_latency = self.dijkstra_router.calculate_shortest_path(
            source_station, destination_station
        )
        dijkstra_time = max(time.time() - start_time, 0.001)  # FIX: Đảm bảo không chia cho 0
        
        if dijkstra_path:
            print(f"   Path: {' → '.join(dijkstra_path)}")
            print(f"   Total Latency: {dijkstra_latency:.2f} ms")
            print(f"   Hop Count: {len(dijkstra_path) - 1}")
        else:
            print("   ❌ No path found")
            dijkstra_latency = float('inf')
        
        print(f"   Computation Time: {dijkstra_time*1000:.2f} ms")
        
        # Test RL
        print("\n🧠 RL ROUTING:")
        start_time = time.time()
        rl_path, rl_latency = self.rl_router.find_rl_path(source_station, destination_station)
        rl_time = time.time() - start_time
        
        if rl_path and rl_latency < float('inf'):
            print(f"   Path: {' → '.join(rl_path)}")
            print(f"   Total Latency: {rl_latency:.2f} ms")
            print(f"   Hop Count: {len(rl_path) - 1}")
        else:
            print("   ❌ No path found")
            rl_latency = float('inf')
        
        print(f"   Computation Time: {rl_time*1000:.2f} ms")
        
        # Comparison - FIX: Safe division
        print("\n📊 COMPARISON RESULTS:")
        print("=" * 40)
        
        if dijkstra_latency < float('inf') and rl_latency < float('inf'):
            latency_diff = rl_latency - dijkstra_latency
            latency_ratio = (rl_latency / max(dijkstra_latency, 0.001) - 1) * 100  # FIX: safe division
            
            if latency_diff < -0.1:  # RL nhanh hơn 0.1ms
                print(f"🎉 RL WINS: {-latency_diff:.2f}ms faster ({-latency_ratio:.1f}% improvement)")
            elif latency_diff > 0.1:  # Dijkstra nhanh hơn 0.1ms
                print(f"📈 Dijkstra WINS: {latency_diff:.2f}ms faster ({latency_ratio:.1f}% better)")
            else:
                print("🤝 TIE: Similar latency (±0.1ms)")
            
            # Compare hop count
            dijkstra_hops = len(dijkstra_path) - 1
            rl_hops = len(rl_path) - 1
            hop_diff = rl_hops - dijkstra_hops
            
            if hop_diff < 0:
                print(f"🔄 RL uses {-hop_diff} fewer hops")
            elif hop_diff > 0:
                print(f"🔄 Dijkstra uses {hop_diff} fewer hops")
            else:
                print("🔄 Same hop count")
                
        elif dijkstra_latency < float('inf'):
            print("❌ RL failed to find a path")
        elif rl_latency < float('inf'):
            print("❌ Dijkstra failed to find a path")
        else:
            print("❌ Both algorithms failed to find paths")
        
        # FIX: Safe time comparison
        if dijkstra_time > 0 and rl_time > 0:
            time_ratio = rl_time / dijkstra_time
            speed_desc = "slower" if time_ratio > 1 else "faster"
            print(f"⏱️  RL computation was {time_ratio:.1f}x {speed_desc}")
        else:
            print("⏱️  Computation time comparison not available")

def main():
    """Main test function"""
    print("🚀 SAGINs RL vs Dijkstra Routing Test")
    print("=" * 50)
    
    # Kết nối MongoDB
    mongo_manager = MongoDataManager()
    
    # Test connection
    try:
        snapshot = mongo_manager.get_training_snapshot()
        nodes = snapshot.get('nodes', {})
        print(f"✅ Connected to MongoDB - {len(nodes)} nodes available")
        
        # Hiển thị available ground/sea stations
        ground_stations = [nid for nid, ndata in nodes.items() 
                          if ndata.get('nodeType') in ['GROUND_STATION', 'SEA_STATION']]
        print(f"📍 Available Stations: {ground_stations}")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return
    
    # Tạo packet sender
    try:
        packet_sender = PacketSender(mongo_manager)
    except Exception as e:
        print(f"❌ Failed to initialize routers: {e}")
        return
    
    # Test cases
    test_cases = [
        # (source, destination)
        ("GS_SINGAPORE", "GS_TOKYO"),
        ("GS_LONDON", "GS_SEOUL"), 
        ("SS_ARABIAN_SEA_01", "GS_BANGKOK"),
        ("GS_DUBLIN", "SS_PACIFIC_01"),
        ("GS_MUMBAI", "SS_NORTH_ATLANTIC_02")
    ]
    
    # Chạy test cases
    for i, (source, dest) in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}/{len(test_cases)}")
        print("-" * 40)
        
        if source in nodes and dest in nodes:
            packet_sender.send_packet(source, dest)
        else:
            print(f"❌ Invalid nodes: {source} or {dest} not found")
        
        # Pause between tests
        if i < len(test_cases):
            input("\n⏸️  Press Enter to run next test...")

if __name__ == "__main__":
    main()