# env/satellite_simulator.py

from python.utils.state_builder import (
    StateBuilder, 
    MAX_NEIGHBORS,             
    MAX_SYSTEM_LATENCY_MS,     
)
from typing import Dict, Any, Tuple, Optional
import numpy as np

# ======================== TRỌNG SỐ REWARD ==========================
DEFAULT_WEIGHTS = {
    'goal': 10.0,             # Thưởng khi đến đích
    'drop': 100.0,            # Phạt khi rớt gói
    'latency': -5.0,          # Phạt trễ hop
    'latency_violation': -50.0, # Phạt vượt QoS
    'utilization': 2.0,       # Thưởng tài nguyên thấp
    'bandwidth': 1.0,         # Thưởng băng thông cao
    'reliability': 3.0,       # Thưởng độ tin cậy
    'fspl': -0.1,             # Phạt suy hao
    'hop_cost': -1.0,         # Phạt mỗi hop để ưu tiên đường ngắn
    'operational': 5.0        # Thưởng node đang hoạt động
}


# ======================== ENVIRONMENT ==============================
class SatelliteEnv:
    """Môi trường mô phỏng nhiều bước định tuyến trong mạng vệ tinh."""

    def __init__(self, state_builder: StateBuilder, weights: Optional[Dict[str, float]] = None):
        self.state_builder = state_builder
        self.weights = DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        self.current_packet_state: Dict[str, Any] = {} 
        
        # Cấu trúc vector S
        # S = V_G (10) + V_P (8) + V_C (6) + V_N (MAX_NEIGHBORS * 7)
        self.NEIGHBOR_SLOT_SIZE = 7
        self.START_INDEX_VN = 10 + 8 + 6 # = 24

    # =============================================================
    # Reset và Step
    # =============================================================

    def reset(self, initial_packet_data: Dict[str, Any]) -> np.ndarray:
        """Khởi tạo trạng thái ban đầu (Vector S)"""
        self.current_packet_state = initial_packet_data
        return self.state_builder.get_state_vector(initial_packet_data)

    def step(self, action_index: int, next_node_id: str, next_packet: Dict[str, Any]) -> Tuple[np.ndarray, float, bool]:
        """
        Tính reward cho 1 bước di chuyển giữa 2 node.
        """
        current_id = next_packet["path"][-2] if len(next_packet["path"]) > 1 else None
        dest_id = next_packet["stationDest"]

        # --- CÁC TRƯỜNG HỢP ĐẶC BIỆT ---
        reached_destination = (next_node_id == dest_id)
        ttl_exhausted = next_packet["ttl"] <= 0
        dropped = next_packet.get("dropped", False)

        # --- REWARD SHAPING ---
        reward = 0.0

        # 1️⃣ Đạt đến đích
        if reached_destination:
            reward += self.weights.get("goal", 300.0)

        # 2️⃣ Hết TTL hoặc bị drop
        elif ttl_exhausted or dropped:
            reward += self.weights.get("drop", -200.0)

        # 3️⃣ Phạt nếu quay lại node cũ (loop)
        elif next_node_id in next_packet["path"][:-1]:
            reward -= 50.0

        # 4️⃣ Thưởng nếu tiến gần đích hơn
        if current_id is not None:
            dist_prev = self.state_builder.get_distance(current_id, dest_id)
            dist_next = self.state_builder.get_distance(next_node_id, dest_id)
            reward += (dist_prev - dist_next) * 2.0  # thưởng nếu gần hơn

        # 5️⃣ Phạt nhẹ cho mỗi hop để khuyến khích đường ngắn
        reward += self.weights.get("hop_cost", -1.0)

        # 6️⃣ Phạt delay cao
        reward -= next_packet["accumulatedDelayMs"] * 0.01

        # Kết thúc nếu đạt đích / TTL hết / drop
        done = reached_destination or ttl_exhausted or dropped

        # Sinh state kế tiếp từ next_packet
        next_state = self.state_builder.get_state_vector(next_packet)
        return next_state, reward, done

    # =============================================================
    # Mô phỏng nhiều bước (multi-hop)
    # =============================================================

    def simulate_episode(self, agent, initial_packet_data: Dict[str, Any], max_hops: int = 10) -> float:
        """
        Cho phép Agent đi qua nhiều hop đến đích hoặc hết TTL.
        Không optimize model từng bước, tối ưu sau khi thu thập transitions.
        """
        state = self.reset(initial_packet_data)
        total_reward = 0.0
        hop = 0
        transitions = []

        while hop < max_hops:
            current_node_id = self.current_packet_state["currentHoldingNodeId"]
            current_node_data = self.state_builder.db.get_node(current_node_id, projection={"neighbors": 1})
            neighbor_ids = current_node_data.get("neighbors", []) if current_node_data else []

            if not neighbor_ids:
                # Node không có neighbor → kết thúc sớm
                new_packet_data = self.current_packet_state.copy()
                new_packet_data["dropped"] = True
                next_state = self.state_builder.get_state_vector(new_packet_data)
                reward = -self.weights['drop']
                transitions.append((state, None, reward, next_state, True))
                total_reward += reward
                break

            action_index = agent.select_action(state)
            # Nếu agent chọn invalid index → random hợp lệ
            if not (0 <= action_index < len(neighbor_ids)):
                action_index = np.random.randint(len(neighbor_ids))

            neighbor_id = neighbor_ids[action_index]
            new_packet_data = self._simulate_hop(self.current_packet_state, neighbor_id)

            next_state, reward, done = self.step(action_index, neighbor_id, new_packet_data)

            transitions.append((state, action_index, reward, next_state, done))
            total_reward += reward

            state = next_state
            self.current_packet_state = new_packet_data
            hop += 1

            if done:
                break

        # Optimize sau khi kết thúc episode
        for s, a, r, s_next, done in transitions:
            if a is not None:
                agent.memory.push(s, a, r, s_next, done)
        agent.optimize_model()

        return total_reward

    # =============================================================
    # Logic phụ
    # =============================================================

    def _is_terminal(self, packet_data: Dict[str, Any]) -> bool:
        """Kết thúc khi gói đến đích hoặc bị drop / hết TTL"""
        is_at_dest = packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest')
        is_dropped = packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0
        return is_at_dest or is_dropped

    def _simulate_hop(self, packet_data: Dict[str, Any], next_node_id: Optional[str]) -> Dict[str, Any]:
        """
        Mô phỏng việc chuyển gói tin qua 1 hop.
        Tùy bạn có thể thay bằng dữ liệu thật từ simulator.
        """
        new_packet = packet_data.copy()
        new_packet["currentHoldingNodeId"] = next_node_id or packet_data["currentHoldingNodeId"]
        new_packet["ttl"] = packet_data.get("ttl", 10) - 1
        new_packet["accumulatedDelayMs"] = packet_data.get("accumulatedDelayMs", 0) + np.random.uniform(1, 5)
        # random drop nhỏ
        new_packet["dropped"] = np.random.rand() < 0.02
        return new_packet

    # =============================================================
    # Reward Function
    # =============================================================

    def _calculate_reward(self, action_index: int, packet_data: Dict[str, Any]) -> float:
        """
        Tính reward dựa trên trạng thái trước và sau hop.
        """
        w = self.weights

        # 1️⃣ Goal / Drop
        if packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest'):
            return w['goal']
        if packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0:
            return -w['drop']

        # 2️⃣ Lấy slot metrics từ state trước đó
        prev_S = self.state_builder.get_state_vector(self.current_packet_state)
        SLOT_START = self.START_INDEX_VN + (action_index * self.NEIGHBOR_SLOT_SIZE)

        if action_index >= MAX_NEIGHBORS or (SLOT_START + self.NEIGHBOR_SLOT_SIZE) > len(prev_S):
            return -w['drop'] / 2

        slot_metrics = prev_S[SLOT_START:SLOT_START + self.NEIGHBOR_SLOT_SIZE]

        is_op = slot_metrics[0]
        total_latency_ratio = slot_metrics[1]
        avail_bw_ratio = slot_metrics[2]
        dest_util_ratio = slot_metrics[3]
        loss_rate_neighbor = slot_metrics[4]
        fspl_ratio = slot_metrics[6]

        # 3️⃣ Reward tính toán
        reward = 0.0
        reward += w['hop_cost']
        reward += w['latency'] * total_latency_ratio

        max_lat = packet_data.get('serviceQoS', {}).get('maxLatencyMs', MAX_SYSTEM_LATENCY_MS)
        curr_delay = packet_data.get('accumulatedDelayMs', 0.0)
        if curr_delay / max_lat > 0.9:
            reward += w['latency_violation']

        reward += w['bandwidth'] * avail_bw_ratio
        reward += w['utilization'] * (1.0 - dest_util_ratio)
        reward += w['reliability'] * (1.0 - loss_rate_neighbor)
        reward += w['fspl'] * fspl_ratio
        reward += w['operational'] * is_op

        return float(reward)
