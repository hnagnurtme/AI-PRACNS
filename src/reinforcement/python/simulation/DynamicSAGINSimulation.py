import math
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import random
from models.Node import Node
from models.Position import Position
from models.Orbit import Orbit
from models.Velocity import Velocity
from models.Communication import Communication
from env.DynamicEnvironmentManager import DynamicEnvironmentManager


class DynamicSAGINSimulation:
    def __init__(self):
        self.nodes = []
        self.env_manager = DynamicEnvironmentManager(self.nodes)
        self.simulation_time = 0.0
        self.time_step = 1.0  # 1 giây mỗi bước
        self.episode_length = 1000  # 1000 bước mỗi episode
        
    def create_sample_network(self):
        """Tạo mạng SAGIN mẫu với các node động"""
        
        # 1. Ground Stations
        gs1 = Node(
            nodeId="gs1", nodeName="Ground-Station-1", nodeType="GROUND_STATION",
            position=Position(21.0, 105.8, 0.0),  # Hà Nội
            orbit=Orbit(),
            velocity=Velocity(0, 0, 0),
            communication=Communication(
                frequencyGHz=12.0, bandwidthMHz=100, transmitPowerDbW=20,
                antennaGainDb=30, beamWidthDeg=10, maxRangeKm=2000,
                minElevationDeg=5, ipAddress="192.168.1.1", port=8080
            )
        )
        
        gs2 = Node(
            nodeId="gs2", nodeName="Ground-Station-2", nodeType="GROUND_STATION", 
            position=Position(10.8, 106.7, 0.0),  # Hồ Chí Minh
            orbit=Orbit(),
            velocity=Velocity(0, 0, 0),
            communication=Communication(
                frequencyGHz=12.0, bandwidthMHz=100, transmitPowerDbW=20,
                antennaGainDb=30, beamWidthDeg=10, maxRangeKm=2000,
                minElevationDeg=5, ipAddress="192.168.1.2", port=8080
            )
        )
        
        # 2. LEO Satellites
        leo1 = Node(
            nodeId="leo1", nodeName="LEO-Sat-1", nodeType="LEO_SATELLITE",
            position=Position(20.0, 105.0, 800.0),  # Quỹ đạo LEO
            orbit=Orbit(semiMajorAxisKm=7171.0, inclinationDeg=45.0, trueAnomalyDeg=0.0),
            velocity=Velocity(7.5, 0, 0),
            communication=Communication(
                frequencyGHz=20.0, bandwidthMHz=500, transmitPowerDbW=15,
                antennaGainDb=25, beamWidthDeg=15, maxRangeKm=3000,
                minElevationDeg=0, ipAddress="10.0.1.1", port=8080
            )
        )
        
        leo2 = Node(
            nodeId="leo2", nodeName="LEO-Sat-2", nodeType="LEO_SATELLITE",
            position=Position(15.0, 107.0, 800.0),
            orbit=Orbit(semiMajorAxisKm=7171.0, inclinationDeg=45.0, trueAnomalyDeg=120.0),
            velocity=Velocity(0, 7.5, 0),
            communication=Communication(
                frequencyGHz=20.0, bandwidthMHz=500, transmitPowerDbW=15,
                antennaGainDb=25, beamWidthDeg=15, maxRangeKm=3000,
                minElevationDeg=0, ipAddress="10.0.1.2", port=8080
            )
        )
        
        # 3. MEO Satellite
        meo1 = Node(
            nodeId="meo1", nodeName="MEO-Sat-1", nodeType="MEO_SATELLITE",
            position=Position(15.0, 105.0, 10000.0),
            orbit=Orbit(semiMajorAxisKm=16371.0, inclinationDeg=55.0, trueAnomalyDeg=0.0),
            velocity=Velocity(3.0, 0, 0),
            communication=Communication(
                frequencyGHz=15.0, bandwidthMHz=300, transmitPowerDbW=18,
                antennaGainDb=28, beamWidthDeg=8, maxRangeKm=10000,
                minElevationDeg=0, ipAddress="10.0.2.1", port=8080
            )
        )
        
        # 4. GEO Satellite
        geo1 = Node(
            nodeId="geo1", nodeName="GEO-Sat-1", nodeType="GEO_SATELLITE",
            position=Position(15.0, 105.0, 35786.0),
            orbit=Orbit(semiMajorAxisKm=42164.0, inclinationDeg=0.0, trueAnomalyDeg=0.0),
            velocity=Velocity(0, 3.0, 0),
            communication=Communication(
                frequencyGHz=10.0, bandwidthMHz=200, transmitPowerDbW=25,
                antennaGainDb=35, beamWidthDeg=2, maxRangeKm=35000,
                minElevationDeg=0, ipAddress="10.0.3.1", port=8080
            )
        )
        
        self.nodes = [gs1, gs2, leo1, leo2, meo1, geo1]
        self.env_manager = DynamicEnvironmentManager(self.nodes)
        
    def get_network_state(self) -> Dict:
        """Lấy trạng thái hiện tại của toàn mạng cho RL agent"""
        state = {
            'time': self.simulation_time,
            'nodes': {},
            'adjacency_matrix': self._get_adjacency_matrix(),
            'link_qualities': self._get_link_qualities(),
            'node_statuses': self._get_node_statuses(),
            'traffic_load': self.env_manager.traffic_model.base_traffic,
            'weather_intensity': self.env_manager.weather_model.weather_intensity
        }
        
        for node in self.nodes:
            state['nodes'][node.nodeId] = {
                'position': node.position.to_dict(),
                'battery': node.batteryChargePercent,
                'congestion': node.congestion_level,
                'operational': node.isOperational,
                'weather': node.weather,
                'packet_count': node.currentPacketCount,
                'delay': node.get_current_delay(),
                'link_quality': node.communication.link_quality
            }
            
        return state
    
    def _get_adjacency_matrix(self) -> np.ndarray:
        """Ma trận kề - liên kết nào đang hoạt động"""
        n = len(self.nodes)
        adj_matrix = np.zeros((n, n))
        
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j and node1.isOperational and node2.isOperational:
                    quality = node1.get_link_quality_to(node2)
                    if quality > 0.1:  # Ngưỡng chất lượng tối thiểu
                        adj_matrix[i][j] = 1
                        
        return adj_matrix
    
    def _get_link_qualities(self) -> np.ndarray:
        """Ma trận chất lượng liên kết"""
        n = len(self.nodes)
        quality_matrix = np.zeros((n, n))
        
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j:
                    quality_matrix[i][j] = node1.get_link_quality_to(node2)
                    
        return quality_matrix
    
    def _get_node_statuses(self) -> List[float]:
        """Trạng thái các node (chuẩn hóa 0-1)"""
        statuses = []
        for node in self.nodes:
            status = 0.0
            if node.isOperational:
                status = 0.5 + (node.batteryChargePercent / 200.0)  # 0.5-1.0
            statuses.append(status)
        return statuses
    
    def calculate_routing_metrics(self, path: List[str]) -> Dict:
        """Tính metrics cho một đường đi cụ thể"""
        if len(path) < 2:
            return {'valid': False, 'total_delay': float('inf'), 'min_quality': 0}
            
        total_delay = 0
        min_quality = 1.0
        valid_path = True
        
        node_map = {node.nodeId: node for node in self.nodes}
        
        for i in range(len(path) - 1):
            node1 = node_map.get(path[i])
            node2 = node_map.get(path[i + 1])
            
            if not node1 or not node2 or not node1.isOperational or not node2.isOperational:
                valid_path = False
                break
                
            link_quality = node1.get_link_quality_to(node2)
            if link_quality < 0.1:  # Liên kết quá yếu
                valid_path = False
                break
                
            delay = node1.get_current_delay() + node2.get_current_delay()
            total_delay += delay
            min_quality = min(min_quality, link_quality)
            
        return {
            'valid': valid_path,
            'total_delay': total_delay if valid_path else float('inf'),
            'min_quality': min_quality if valid_path else 0,
            'hop_count': len(path) - 1
        }
    
    def step(self, action=None):
        """Thực hiện một bước mô phỏng"""
        # Cập nhật môi trường
        self.env_manager.update_environment(self.time_step)
        self.simulation_time += self.time_step
        
        # Lấy trạng thái mới
        new_state = self.get_network_state()
        
        # Tính reward (ví dụ: negative của total delay trong network)
        reward = self._calculate_reward()
        
        # Kiểm tra kết thúc episode
        done = self.simulation_time >= self.episode_length
        
        # Info for debugging
        info = {
            'simulation_time': self.simulation_time,
            'operational_nodes': sum(1 for node in self.nodes if node.isOperational),
            'avg_battery': np.mean([node.batteryChargePercent for node in self.nodes]),
            'avg_congestion': np.mean([node.congestion_level for node in self.nodes])
        }
        
        return new_state, reward, done, info
    
    def _calculate_reward(self) -> float:
        """Tính reward cho RL agent"""
        # Reward dựa trên hiệu suất mạng
        operational_nodes_count = sum(1 for node in self.nodes if node.isOperational)
        operational_bonus = operational_nodes_count * 0.1
        
        # Penalty cho non-operational nodes
        non_operational_penalty = (len(self.nodes) - operational_nodes_count) * 0.5

        # Penalty cho congestion cao
        congestion_penalty = np.mean([node.congestion_level for node in self.nodes]) * 2.0
        
        # Bonus cho battery level cao
        battery_bonus = np.mean([node.batteryChargePercent for node in self.nodes]) / 100.0

        # Penalty cho delay cao
        avg_delay = np.mean([node.get_current_delay() for node in self.nodes])
        delay_penalty = avg_delay * 0.01 # Adjust multiplier as needed

        total_reward = operational_bonus - non_operational_penalty - congestion_penalty + battery_bonus - delay_penalty
        return float(total_reward)
    
    def reset(self):
        """Reset môi trường về trạng thái ban đầu"""
        self.simulation_time = 0.0
        self.create_sample_network()  # Recreate network
        return self.get_network_state()

# 3. RL Agent đơn giản (Q-Learning)
class SimpleRLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def choose_action(self, state):
        """Chọn action theo epsilon-greedy policy"""
        state_idx = self._state_to_index(state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)  # Exploration
        else:
            return np.argmax(self.q_table[state_idx])  # Exploitation
            
    def learn(self, state, action, reward, next_state, done):
        """Cập nhật Q-table"""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        
        # Q-learning update
        best_next_action = np.max(self.q_table[next_state_idx])
        td_target = reward + self.discount_factor * best_next_action * (1 - done)
        td_error = td_target - self.q_table[state_idx][action]
        
        self.q_table[state_idx][action] += self.learning_rate * td_error
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _state_to_index(self, state):
        """Chuyển state thành index (đơn giản hóa)"""
        # Trong thực tế, bạn cần mã hóa state phức tạp hơn
        operational_count = sum(1 for node_id, node_data in state['nodes'].items() 
                              if node_data['operational'])
        return min(operational_count, self.state_size - 1)

# 4. Dijkstra Algorithm cho so sánh
class DijkstraRouter:
    def __init__(self, simulation):
        self.simulation = simulation
        
    def find_shortest_path(self, start_node_id: str, end_node_id: str) -> List[str]:
        """Tìm đường đi ngắn nhất sử dụng Dijkstra"""
        nodes = self.simulation.nodes
        node_map = {node.nodeId: node for node in nodes}
        
        if start_node_id not in node_map or end_node_id not in node_map:
            return []
            
        # Khởi tạo
        distances = {node_id: float('inf') for node_id in node_map}
        previous = {node_id: None for node_id in node_map}
        distances[start_node_id] = 0
        unvisited = set(node_map.keys())
        
        while unvisited:
            # Tìm node có distance nhỏ nhất
            current = min(unvisited, key=lambda node_id: distances[node_id])
            
            if distances[current] == float('inf'):
                break  # Không còn node nào có thể đến
                
            unvisited.remove(current)
            
            if current == end_node_id:
                break  # Đã tìm thấy đích
                
            # Cập nhật khoảng cách đến neighbors
            current_node = node_map[current]
            for neighbor_id in unvisited:
                neighbor_node = node_map[neighbor_id]
                
                if (current_node.isOperational and neighbor_node.isOperational and
                    current_node.get_link_quality_to(neighbor_node) > 0.1):
                    
                    # Cost = delay (có thể thay đổi thành metric khác)
                    cost = current_node.get_current_delay()
                    new_distance = distances[current] + cost
                    
                    if new_distance < distances[neighbor_id]:
                        distances[neighbor_id] = new_distance
                        previous[neighbor_id] = current
        
        # Truy vết đường đi
        path = []
        current = end_node_id
        while current is not None:
            path.append(current)
            current = previous[current]
            
        path.reverse()
        return path if path[0] == start_node_id else []

# 5. Chạy Demo
def run_demo():
    print("=== SAGIN DYNAMIC SIMULATION DEMO ===")
    
    # Khởi tạo simulation
    sim = DynamicSAGINSimulation()
    sim.create_sample_network()
    
    # Khởi tạo router
    dijkstra = DijkstraRouter(sim)
    
    print("Initial Network State:")
    initial_state = sim.get_network_state()
    print(f"Number of nodes: {len(sim.nodes)}")
    print(f"Operational nodes: {sum(1 for node in sim.nodes if node.isOperational)}")
    
    # Chạy simulation trong 10 bước
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        
        # Cập nhật môi trường
        state, reward, done, info = sim.step()
        
        # Hiển thị thông tin
        print(f"Simulation Time: {info['simulation_time']:.1f}s")
        print(f"Operational Nodes: {info['operational_nodes']}")
        print(f"Average Battery: {info['avg_battery']:.1f}%")
        print(f"Average Congestion: {info['avg_congestion']:.2f}")
        
        # Tìm đường đi từ gs1 đến gs2 bằng Dijkstra
        path = dijkstra.find_shortest_path("gs1", "gs2")
        metrics = sim.calculate_routing_metrics(path)
        
        print(f"Dijkstra Path: {path}")
        print(f"Path Valid: {metrics['valid']}")
        if metrics['valid']:
            print(f"Total Delay: {metrics['total_delay']:.2f}ms")
            print(f"Min Link Quality: {metrics['min_quality']:.2f}")
        
        # Hiển thị trạng thái một vài node
        print("\nNode Status:")
        for node in sim.nodes[:3]:  # Chỉ hiển thị 3 node đầu
            print(f"  {node.nodeName}: "
                  f"Battery={node.batteryChargePercent}%, "
                  f"Operational={node.isOperational}, "
                  f"Weather={node.weather}")
    
    print("\n=== SIMULATION COMPLETE ===")

# 6. So sánh hiệu năng RL vs Dijkstra
def compare_performance():
    """So sánh hiệu năng giữa RL và Dijkstra"""
    sim = DynamicSAGINSimulation()
    sim.create_sample_network()
    dijkstra = DijkstraRouter(sim)
    
    results = []
    
    for episode in range(5):  # 5 episodes
        sim.reset()
        dijkstra_success = 0
        rl_success = 0
        
        for step in range(100):  # 100 steps mỗi episode
            sim.step()
            
            # Test Dijkstra
            path = dijkstra.find_shortest_path("gs1", "gs2")
            metrics = sim.calculate_routing_metrics(path)
            if metrics['valid']:
                dijkstra_success += 1
            
            # Ở đây bạn sẽ test RL agent thực tế
            # rl_path = rl_agent.choose_path(state)
            # rl_metrics = sim.calculate_routing_metrics(rl_path)
            # if rl_metrics['valid']:
            #     rl_success += 1
        
        success_rate_dijkstra = dijkstra_success / 100
        # success_rate_rl = rl_success / 100
        
        results.append({
            'episode': episode,
            'dijkstra_success_rate': success_rate_dijkstra,
            # 'rl_success_rate': success_rate_rl
        })
        
        print(f"Episode {episode}: Dijkstra Success Rate = {success_rate_dijkstra:.2f}")
    
    return results

if __name__ == "__main__":
    # Chạy demo
    run_demo()
    
    # So sánh hiệu năng
    print("\n=== PERFORMANCE COMPARISON ===")
    results = compare_performance()