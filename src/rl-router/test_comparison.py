import torch
import numpy as np
import heapq
import random
import math
from typing import List, Dict, Tuple

# Import modules
from python.utils.mock_db import MockDBConnector
from python.utils.state_builder import StateBuilder
from python.utils.constants import SPEED_OF_LIGHT
from python.utils.math_utils import to_cartesian_ecef
from python.env.satellite_simulator import SatelliteEnv
from python.rl_agent.trainer import DQNAgent

# Đảm bảo đường dẫn đúng tới file model tốt nhất của bạn
MODEL_PATH = "models/checkpoints/dqn_latest.pth" 

# ==============================================================================
# 1. DIJKSTRA (GLOBAL KNOWLEDGE)
# ==============================================================================
class DijkstraSolver:
    def __init__(self, db: MockDBConnector):
        self.db = db
        self.graph = self._build_graph()

    def _calculate_weight(self, node_a, node_b):
        pos_a = to_cartesian_ecef(node_a['position'])
        pos_b = to_cartesian_ecef(node_b['position'])
        dist_km = np.linalg.norm(pos_a - pos_b)
        prop_delay = (dist_km / SPEED_OF_LIGHT) * 1000.0
        proc_delay = node_b.get('nodeProcessingDelayMs', 5.0)
        return prop_delay + proc_delay

    def _build_graph(self):
        nodes = self.db.get_all_nodes()
        node_map = {n['nodeId']: n for n in nodes}
        graph = {}
        for node in nodes:
            nid = node['nodeId']
            graph[nid] = []
            for neighbor_id in node['neighbors']:
                if neighbor_id in node_map:
                    weight = self._calculate_weight(node, node_map[neighbor_id])
                    graph[nid].append((neighbor_id, weight))
        return graph

    def find_path(self, start_node: str, end_node: str) -> Tuple[List[str], float]:
        pq = [(0.0, start_node, [start_node])]
        visited = set()
        while pq:
            cost, current, path = heapq.heappop(pq)
            if current == end_node:
                return path, cost
            if current in visited: continue
            visited.add(current)
            if current in self.graph:
                for neighbor, weight in self.graph[current]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))
        return [], float('inf')

# ==============================================================================
# 2. RL AGENT RUNNER (WITH FULL LOOP AVOIDANCE)
# ==============================================================================
def run_rl_episode(agent: DQNAgent, env: SatelliteEnv, src: str, dest: str):
    packet = {
        "packetId": "TEST", "currentHoldingNodeId": src, "stationDest": dest,
        "ttl": 50, "accumulatedDelayMs": 0.0, "serviceQoS": {"maxLatencyMs": 5000.0},
        "dropped": False, "path": [src]
    }
    
    state = env.reset(packet)
    done = False
    path = [src]
    total_delay = 0.0
    hops = 0

    # Dùng set để tra cứu O(1) các node đã đi qua
    visited_set = {src}

    while not done and hops < 50:
        curr_id = env.current_packet_state["currentHoldingNodeId"]
        node_data = env.state_builder.db.get_node(curr_id)
        neighbors = node_data.get('neighbors', [])
        num_neighbors = len(neighbors)

        # Lấy Q-values từ mạng Neural
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.q_network.fc1.weight.device)
            q_values = agent.q_network(state_tensor).squeeze(0) # [10]

        # --- MASKING LOGIC (QUAN TRỌNG) ---
        
        # 1. Mask các action index không tồn tại
        q_values[num_neighbors:] = float('-inf')

        # 2. Mask các Neighbor ĐÃ ĐI QUA (Full Loop Prevention)
        # Nếu neighbor đã có trong path, cấm gửi tới đó
        all_masked = True
        for idx, n_id in enumerate(neighbors):
            if n_id in visited_set:
                q_values[idx] = float('-inf')
            else:
                all_masked = False # Vẫn còn đường đi

        # --- XỬ LÝ NGÕ CỤT (DEAD END) ---
        if all_masked:
            # Nếu tất cả hàng xóm đều đã đi qua -> Buộc phải quay lại hoặc Drop
            # Ở đây ta chọn Drop để kết thúc vòng lặp vô nghĩa
            path.append("DROP_LOOP")
            done = True
        else:
            # Chọn đường tốt nhất trong các đường chưa đi
            action = q_values.argmax().item()

            if action < len(neighbors):
                next_node = neighbors[action]
                new_packet = env._simulate_hop(env.current_packet_state, next_node)
                
                state = env.state_builder.build(new_packet)
                env.current_packet_state = new_packet
                
                path.append(next_node)
                visited_set.add(next_node) # Đánh dấu đã thăm
                total_delay = new_packet['accumulatedDelayMs']
                done = env._is_terminal(new_packet)
            else:
                path.append("DROP_INVALID")
                done = True
                
        hops += 1

    success = (path[-1] == dest)
    return path, total_delay, success
# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    print("🚀 LOADING SYSTEM FOR COMPARISON...")
    
    db = MockDBConnector()
    dijkstra = DijkstraSolver(db)
    state_builder = StateBuilder(db)
    
    env = SatelliteEnv(state_builder)
    agent = DQNAgent(env, use_legacy_architecture=False)

    # --- LOAD MODEL FIX ---
    try:
        print(f"📥 Loading model from: {MODEL_PATH}")
        # SỬA LỖI CHÍNH: weights_only=False
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['model'])
        else:
            agent.q_network.load_state_dict(checkpoint)
            
        agent.q_network.eval()
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print(f"❌ File not found: {MODEL_PATH}. Please train first!")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️ Running with RANDOM weights (expect failures)")

    # Test Cases
    test_cases = [
        ("GS_HANOI", "GS_HCM"),
        ("SAT_1", "SAT_8"),
        ("SAT_2", "SAT_7"),
        ("GS_HANOI", "SAT_6"),
        ("SAT_3", "SAT_1"), # Case gần
        ("SAT_2", "SAT_4")
    ]

    print("\n" + "="*110)
    print(f"{'SRC':<10} | {'DEST':<10} | {'ALG':<8} | {'STATUS':<8} | {'HOPS':<4} | {'DELAY':<8} | {'PATH'}")
    print("="*110)

    stats = {"rl": 0, "dij": 0, "tie": 0}

    for src, dest in test_cases:
        # Dijkstra
        d_path, d_delay = dijkstra.find_path(src, dest)
        d_hops = len(d_path) - 1
        
        # RL Agent
        r_path, r_delay, r_success = run_rl_episode(agent, env, src, dest)
        r_hops = len(r_path) - 1
        r_status = "OK" if r_success else "FAIL"

        print(f"{src:<10} | {dest:<10} | {'Dijkstra':<8} | {'OK':<8} | {d_hops:<4} | {d_delay:<8.1f} | {d_path}")
        print(f"{'':<10} | {'':<10} | {'RL':<8} | {r_status:<8} | {r_hops:<4} | {r_delay:<8.1f} | {r_path}")
        
        if r_success:
            if r_hops < d_hops:
                print(f"   >>> 🏆 RL WINS (Shorter)")
                stats["rl"] += 1
            elif r_hops > d_hops:
                print(f"   >>> 🤖 DIJKSTRA WINS")
                stats["dij"] += 1
            else:
                print(f"   >>> 🤝 TIE (Optimal)")
                stats["tie"] += 1
        else:
             print(f"   >>> ❌ RL FAILS")
        print("-" * 110)

    print(f"\nSUMMARY: RL Wins: {stats['rl']} | Dijkstra Wins: {stats['dij']} | Ties: {stats['tie']}")

if __name__ == "__main__":
    main()