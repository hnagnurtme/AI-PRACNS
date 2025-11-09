# env/satellite_simulator.py

from python.utils.state_builder import (
    StateBuilder, 
    MAX_NEIGHBORS,             
    MAX_SYSTEM_LATENCY_MS,     
)
from typing import Dict, Any, Tuple, Optional
import numpy as np
import random # Dùng thư viện 'random' chuẩn

# ======================== TRỌNG SỐ REWARD (TỐI ƯU HÓA MỚI) ==========================
DEFAULT_WEIGHTS = {
    # --- Trọng số Mục tiêu ---
    'goal': 150.0,             # Thưởng LỚN khi đến đích (tăng từ 100)
    'drop': 200.0,             # Phạt NẶNG khi drop (giữ nguyên)

    # --- Trọng số Định hình Hướng đi (ƯU TIÊN LATENCY THẤP & ĐƯỜNG NGẮN) ---
    'hop_cost': -80.0,         # Phạt cho mỗi hop (giảm từ -100 để khuyến khích đa dạng)
    'progress_penalty': -150.0, # Phạt nếu đi xa đích (giảm từ -200)
    
    # --- Trọng số QoS & Resource (TĂNG CƯỜNG ĐỂ CÂN BẰNG) ---
    'latency': -8.0,           # Phạt độ trễ cao (tăng từ -5 để ưu tiên latency thấp)
    'latency_violation': -100.0, # Phạt RẤT NẶNG nếu vượt QoS (tăng từ -50)
    'bandwidth': 3.0,          # Thưởng băng thông khả dụng cao (tăng từ 1)
    'utilization': 8.0,        # Thưởng node utilization thấp (tăng từ 2)
    'reliability': 5.0,        # Thưởng packet loss thấp (tăng từ 3)
    'fspl': -0.2,              # Phạt FSPL cao (tăng từ -0.1)
    'operational': 10.0,       # Thưởng node hoạt động tốt (tăng từ 5)
    
    # --- Trọng số Resource mới (THEO DÕI TÀI NGUYÊN NODE) ---
    'node_load': -10.0,        # Phạt node có load cao (tỷ lệ packet/capacity)
    'resource_balance': 5.0,   # Thưởng nếu resource utilization cân bằng
}

# ======================== ENVIRONMENT ==============================
class SatelliteEnv:
    """
    Môi trường mô phỏng (Optimized Environment).
    Quản lý toàn bộ vòng đời của một episode và tự động huấn luyện agent.
    """

    def __init__(self, state_builder: StateBuilder, weights: Optional[Dict[str, float]] = None):
        self.state_builder = state_builder
        self.weights = DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        self.current_packet_state: Dict[str, Any] = {} 
        
        # Cấu trúc vector S (94 chiều)
        self.NEIGHBOR_SLOT_SIZE = 7
        self.START_INDEX_VN = 24 # V_G (10) + V_P (8) + V_C (6)

    # =============================================================
    # Reset
    # =============================================================

    def reset(self, initial_packet_data: Dict[str, Any]) -> np.ndarray:
        """Khởi tạo trạng thái ban đầu (Vector S) cho một episode mới."""
        self.current_packet_state = initial_packet_data
        return self.state_builder.get_state_vector(initial_packet_data)

    # =============================================================
    # Mô phỏng nhiều bước (multi-hop)
    # =============================================================

    def simulate_episode(self, agent, initial_packet_data: Dict[str, Any], max_hops: int = 10) -> float:
        """
        (TỐI ƯU) Chạy toàn bộ episode.
        Đây là hàm duy nhất mà `main_train` cần gọi.
        """
        state = self.reset(initial_packet_data)
        total_reward = 0.0
        hop = 0
        transitions = [] # Lưu trữ (s, a, r, s') để học sau
        visited_nodes = set()  # Track visited nodes to detect loops

        while hop < max_hops:
            current_node_id = self.current_packet_state["currentHoldingNodeId"]
            current_node_data = self.state_builder.db.get_node(current_node_id, projection={"neighbors": 1})
            
            neighbor_ids_full = current_node_data.get("neighbors", []) if current_node_data else []
            neighbor_ids = neighbor_ids_full[:MAX_NEIGHBORS] # Khớp policy (10 neighbors)
            neighbor_count = len(neighbor_ids)

            # --- TRƯỜNG HỢP 1: BỊ KẸT (Không có neighbor) ---
            if neighbor_count == 0:
                reward = -self.weights['drop']
                done = True
                next_state = self.state_builder.get_state_vector(
                    self._simulate_hop(self.current_packet_state, None, dropped=True)
                )
                transitions.append((state, None, reward, next_state, done))
                total_reward += reward
                break 

            # --- TRƯỜNG HỢP 2: AGENT CHỌN HÀNH ĐỘNG ---
            action_index = agent.select_action(state, num_valid_actions=neighbor_count)  # Agent trả về 0-(neighbor_count-1)

            # --- TRƯỜNG HỢP 2A: HÀNH ĐỘNG KHÔNG HỢP LỆ ---
            # (TỐI ƯU) Phạt nặng agent nếu chọn action không tồn tại
            if not (0 <= action_index < neighbor_count):
                reward = -self.weights['drop']
                done = True
                next_state = state # Trạng thái không đổi
                
                transitions.append((state, action_index, reward, next_state, done))
                total_reward += reward
                break 

            # --- TRƯỜNG HỢP 2B: HÀNH ĐỘNG HỢP LỆ ---
            neighbor_id = neighbor_ids[action_index]
            
            # (MỚI) Check for loop - penalize revisiting nodes
            if neighbor_id in visited_nodes:
                # Penalize loop but don't end episode immediately
                reward = -self.weights['hop_cost'] * 3  # Triple penalty for loops
                new_packet_data = self._simulate_hop(self.current_packet_state, neighbor_id)
                next_state = self.state_builder.get_state_vector(new_packet_data)
                done = False
            else:
                visited_nodes.add(current_node_id)
                new_packet_data = self._simulate_hop(self.current_packet_state, neighbor_id)
                next_state = self.state_builder.get_state_vector(new_packet_data)
                done = self._is_terminal(new_packet_data)
                
                # Tính reward DỰA TRÊN (state, action) -> next_state
                reward = self._calculate_reward(state, action_index, new_packet_data)

            transitions.append((state, action_index, reward, next_state, done))
            total_reward += reward

            # Cập nhật cho vòng lặp tiếp theo
            state = next_state
            self.current_packet_state = new_packet_data
            hop += 1

            if done:
                break

        # --- TỐI ƯU SAU KHI KẾT THÚC EPISODE ---
        for s, a, r, s_next, d in transitions:
            if a is not None: 
                agent.memory.push(s, a, r, s_next, d)
        
        # Chỉ gọi optimize 1 lần sau mỗi episode
        agent.optimize_model()

        return total_reward

    # =============================================================
    # Logic phụ (Helpers)
    # =============================================================

    def _is_terminal(self, packet_data: Dict[str, Any]) -> bool:
        """Kiểm tra điều kiện kết thúc episode."""
        is_at_dest = packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest')
        is_dropped = packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0
        return is_at_dest or is_dropped

    def _simulate_hop(self, packet_data: Dict[str, Any], next_node_id: Optional[str], dropped: bool = False) -> Dict[str, Any]:
        """
        Mô phỏng việc chuyển gói tin qua 1 hop.
        """
        new_packet = packet_data.copy()
        
        new_packet["currentHoldingNodeId"] = next_node_id or packet_data["currentHoldingNodeId"]
        new_packet["ttl"] = packet_data.get("ttl", 10) - 1
        new_packet["accumulatedDelayMs"] = packet_data.get("accumulatedDelayMs", 0) + random.uniform(1, 5)
        
        if dropped or (random.random() < 0.02): # 2% chance to drop
             new_packet["dropped"] = True
        
        return new_packet

    # =============================================================
    # Reward Function (Trái tim của sự tối ưu)
    # =============================================================

    def _calculate_reward(self, prev_state_vector: np.ndarray, action_index: int, packet_data: Dict[str, Any]) -> float:
        """
        (TỐI ƯU HÓA MỚI) Tính reward cho (state, action) -> next_state.
        Cân bằng giữa: Mục tiêu, Latency thấp, Resource optimization, và QoS.
        """
        w = self.weights

        # 1️⃣ Kiểm tra Goal / Drop (dựa trên trạng thái MỚI)
        if packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest'):
            return w['goal']
        if packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0:
            return -w['drop']

        # 2️⃣ Lấy slot metrics từ state TRƯỚC ĐÓ (prev_state_vector)
        SLOT_START = self.START_INDEX_VN + (action_index * self.NEIGHBOR_SLOT_SIZE)

        if action_index >= MAX_NEIGHBORS or (SLOT_START + self.NEIGHBOR_SLOT_SIZE) > len(prev_state_vector):
            return -w['drop'] / 2 # Lỗi logic, phạt

        slot_metrics = prev_state_vector[SLOT_START : SLOT_START + self.NEIGHBOR_SLOT_SIZE]

        is_op = slot_metrics[0]
        total_latency_ratio = slot_metrics[1]
        avail_bw_ratio = slot_metrics[2]
        dest_util_ratio = slot_metrics[3]
        loss_rate_neighbor = slot_metrics[4]
        dist_to_dest_neighbor_ratio = slot_metrics[5] # Đặc trưng khoảng cách
        fspl_ratio = slot_metrics[6]

        # 3️⃣ Tính toán Reward chi tiết (Improved Shaping)
        reward = 0.0
        
        # A. Định hướng đường đi (Progress towards destination)
        reward += w['hop_cost'] 
        reward += w['progress_penalty'] * dist_to_dest_neighbor_ratio

        # B. Ưu tiên Latency thấp
        # Phạt vi phạm QoS nghiêm trọng
        max_lat = packet_data.get('serviceQoS', {}).get('maxLatencyMs', MAX_SYSTEM_LATENCY_MS)
        curr_delay = packet_data.get('accumulatedDelayMs', 0.0)
        if max_lat > 0:
            lat_ratio = curr_delay / max_lat
            if lat_ratio > 0.9: # Vượt 90% threshold
                reward += w['latency_violation']
            elif lat_ratio > 0.7: # Cảnh báo khi vượt 70%
                reward += w['latency_violation'] * 0.3

        # Phạt latency của link hiện tại
        reward += w['latency'] * total_latency_ratio 

        # C. Tối ưu hóa Resource (Node Load & Utilization)
        # Thưởng cho node có utilization thấp (ít tải)
        reward += w['utilization'] * (1.0 - dest_util_ratio)
        
        # (MỚI) Thêm phạt cho node có load cao (dựa trên buffer usage)
        # Lấy thông tin buffer từ current packet state
        current_node_id = packet_data.get('currentHoldingNodeId')
        if current_node_id:
            current_node = self.state_builder.db.get_node(
                current_node_id, 
                projection={"currentPacketCount": 1, "packetBufferCapacity": 1}
            )
            if current_node:
                curr_count = current_node.get("currentPacketCount", 0)
                capacity = max(current_node.get("packetBufferCapacity", 1), 1)
                node_load_ratio = curr_count / capacity
                reward += w['node_load'] * node_load_ratio
        
        # (MỚI) Thưởng cho resource balance (tránh overload một node)
        if dest_util_ratio > 0.1 and dest_util_ratio < 0.7:
            # Sweet spot: Node đang hoạt động nhưng không quá tải
            reward += w['resource_balance']

        # D. QoS Metrics khác
        reward += w['bandwidth'] * avail_bw_ratio
        reward += w['reliability'] * (1.0 - loss_rate_neighbor)
        reward += w['fspl'] * fspl_ratio
        reward += w['operational'] * is_op

        return float(reward)