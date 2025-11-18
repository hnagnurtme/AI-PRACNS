# env/satellite_simulator.py

import numpy as np
import random
from typing import Dict, Any, Optional, Tuple

# Import từ module nội bộ
from python.utils.state_builder import StateBuilder
from python.utils.constants import (
    MAX_NEIGHBORS, 
    NEIGHBOR_FEAT_SIZE,       # Đảm bảo là 14
    MAX_PROCESSING_DELAY_MS,
    MAX_PROPAGATION_DELAY_MS
)

# Định nghĩa fallback nếu chưa có trong constants
MAX_SYSTEM_LATENCY_MS = 5000.0 

# ======================== TRỌNG SỐ REWARD ==========================
DEFAULT_WEIGHTS = {
    'goal': 200.0,
    'drop': 300.0,
    'hop_cost': -150.0,
    'progress_penalty': -250.0,
    'latency': -25.0,
    'latency_violation': -200.0,
    'snr_reward': 5.0,         # Thay cho bandwidth (vì Neighbor feat chứa SNR)
    'utilization': 8.0,
    'reliability': 5.0,
    'operational': 10.0,
    'node_load': -20.0,
    'resource_balance': 5.0,
}

class SatelliteEnv:
    """
    Môi trường mô phỏng vệ tinh (Đã đồng bộ với StateBuilder).
    """

    def __init__(self, state_builder: StateBuilder, weights: Optional[Dict[str, float]] = None):
        self.state_builder = state_builder
        self.weights = DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        self.current_packet_state: Dict[str, Any] = {} 
        
        # === CẤU HÌNH VECTOR STATE (RẤT QUAN TRỌNG) ===
        # Phải khớp chính xác với StateBuilder.build()
        # Part A (Self): 12 features
        # Part B (Dest): 8 features
        self.START_INDEX_NEIGHBORS = 12 + 8  # = 20
        self.NEIGHBOR_FEAT_SIZE = NEIGHBOR_FEAT_SIZE # = 14 (theo constants mới)

    # =============================================================
    # Core Methods
    # =============================================================

    def reset(self, initial_packet_data: Dict[str, Any]) -> np.ndarray:
        """Khởi tạo trạng thái ban đầu."""
        self.current_packet_state = initial_packet_data
        # SỬA LỖI: Gọi đúng tên hàm 'build' thay vì 'get_state_vector'
        return self.state_builder.build(initial_packet_data)

    def simulate_episode(self, agent, initial_packet_data: Dict[str, Any], max_hops: int = 10) -> float:
        """Chạy 1 episode hoàn chỉnh."""
        state = self.reset(initial_packet_data)
        total_reward = 0.0
        hop = 0
        transitions = []
        visited_nodes = set()
        current_node_id = initial_packet_data.get("currentHoldingNodeId")
        visited_nodes.add(current_node_id)

        while hop < max_hops:
            # Lấy thông tin node hiện tại để biết số lượng neighbor thực tế
            current_node_id = self.current_packet_state["currentHoldingNodeId"]
            current_node_data = self.state_builder.db.get_node(current_node_id)
            
            # Lấy danh sách neighbor IDs (cần xử lý trường hợp node chết/mất DB)
            neighbor_ids = current_node_data.get("neighbors", []) if current_node_data else []
            # Cắt danh sách nếu vượt quá MAX (StateBuilder đã sort, nhưng Env cần biết số lượng valid)
            valid_neighbor_count = min(len(neighbor_ids), MAX_NEIGHBORS)

            # --- CASE 1: DEAD END (Không có hàng xóm) ---
            if valid_neighbor_count == 0:
                reward = -self.weights['drop']
                # Next state là state drop
                next_packet = self._simulate_hop(self.current_packet_state, None, dropped=True)
                next_state = self.state_builder.build(next_packet)
                transitions.append((state, None, reward, next_state, True))
                total_reward += reward
                break 

            # --- CASE 2: AGENT CHỌN ACTION ---
            # Masking: Chỉ cho phép chọn trong range [0, valid_neighbor_count-1]
            action_index = agent.select_action(state, num_valid_actions=valid_neighbor_count)

            # Kiểm tra tính hợp lệ của action
            if not (0 <= action_index < valid_neighbor_count):
                # Phạt nặng nếu Agent cố tình chọn slot trống (đã padding)
                reward = -self.weights['drop']
                transitions.append((state, action_index, reward, state, True))
                total_reward += reward
                break

            # Lấy ID neighbor được chọn
            # Lưu ý: Thứ tự neighbor_ids ở đây phải KHỚP với logic sort trong StateBuilder.
            # StateBuilder sort theo Distance. Để chính xác nhất, ta nên lấy lại list đã sort từ StateBuilder logic,
            # nhưng để đơn giản hóa, ta giả định DB trả về hoặc logic sort là nhất quán.
            # TỐI ƯU: Truy vấn neighbor cụ thể từ ID list
            selected_neighbor_id = neighbor_ids[action_index]
            
            # --- CASE 3: LOOP DETECTION ---
            if selected_neighbor_id in visited_nodes:
                reward = -self.weights['hop_cost'] * 3.0 # Phạt gấp 3 lần
                # Vẫn cho đi tiếp nhưng đánh dấu đây là bước đi tồi
                new_packet_data = self._simulate_hop(self.current_packet_state, selected_neighbor_id)
                next_state = self.state_builder.build(new_packet_data)
                done = False # Không end game, để agent tự tìm đường ra
            else:
                visited_nodes.add(selected_neighbor_id)
                new_packet_data = self._simulate_hop(self.current_packet_state, selected_neighbor_id)
                next_state = self.state_builder.build(new_packet_data)
                done = self._is_terminal(new_packet_data)
                
                # Tính Reward chuẩn
                reward = self._calculate_reward(state, action_index, new_packet_data)

            transitions.append((state, action_index, reward, next_state, done))
            total_reward += reward

            state = next_state
            self.current_packet_state = new_packet_data
            hop += 1

            if done:
                break

        # --- HỌC (EXPERIENCE REPLAY) ---
        for s, a, r, s_next, d in transitions:
            if a is not None: 
                agent.memory.push(s, a, r, s_next, d)
        
        agent.optimize_model()
        return total_reward

    # =============================================================
    # Helpers
    # =============================================================

    def _is_terminal(self, packet: Dict[str, Any]) -> bool:
        is_dest = packet.get('currentHoldingNodeId') == packet.get('stationDest')
        is_dead = packet.get('dropped', False) or packet.get('ttl', 0) <= 0
        return is_dest or is_dead

    def _simulate_hop(self, packet: Dict[str, Any], next_node_id: Optional[str], dropped: bool = False) -> Dict[str, Any]:
        new_p = packet.copy()
        if next_node_id:
            new_p["currentHoldingNodeId"] = next_node_id
        
        new_p["ttl"] = new_p.get("ttl", 10) - 1
        
        # Simulation delay ngẫu nhiên (hoặc dùng math_utils nếu muốn chính xác)
        # Giả lập delay = Processing + Propagation
        step_delay = random.uniform(1.0, MAX_PROCESSING_DELAY_MS/10.0) + random.uniform(5.0, 50.0)
        new_p["accumulatedDelayMs"] = new_p.get("accumulatedDelayMs", 0) + step_delay

        if dropped:
            new_p["dropped"] = True
        
        return new_p

    # =============================================================
    # REWARD FUNCTION (ĐÃ FIX MAPPING)
    # =============================================================

    def _calculate_reward(self, prev_state: np.ndarray, action_idx: int, new_packet: Dict[str, Any]) -> float:
        w = self.weights

        # 1. Terminal Rewards
        if new_packet.get('currentHoldingNodeId') == new_packet.get('stationDest'):
            return w['goal']
        if new_packet.get('dropped') or new_packet.get('ttl', 0) <= 0:
            return -w['drop']

        # 2. Extract Neighbor Features (Mapping theo StateBuilder)
        # Vị trí bắt đầu của neighbor được chọn trong mảng state
        start_idx = self.START_INDEX_NEIGHBORS + (action_idx * self.NEIGHBOR_FEAT_SIZE)
        
        # Guard check: tránh index out of bound
        if start_idx + self.NEIGHBOR_FEAT_SIZE > len(prev_state):
            return -w['drop'] 

        # Lấy slice features của neighbor
        feats = prev_state[start_idx : start_idx + self.NEIGHBOR_FEAT_SIZE]
        
        # --- MAPPING CHÍNH XÁC TỪ STATE BUILDER ---
        # [0:Valid, 1:Dist, 2:Cos, 3:SNR, 4:Batt, 5:Queue, 6:CPU, 7:Loss, 8:Delay, 9:Op, 10-12:Pos, 13:Type]
        
        dist_score = feats[1]       # Khoảng cách tới đích (Normalized)
        snr_score = feats[3]        # Chất lượng tín hiệu (thay cho Bandwidth)
        queue_score = feats[5]      # Queue Load
        cpu_score = feats[6]        # CPU Utilization
        loss_score = feats[7]       # Packet Loss Rate
        delay_score = feats[8]      # Node Processing Delay
        is_op = feats[9]            # Operational Status

        reward = 0.0

        # A. Hướng đi & Khoảng cách
        reward += w['hop_cost']
        # Càng gần đích (dist_score thấp) càng ít bị phạt
        reward += w['progress_penalty'] * dist_score 

        # B. Chất lượng Link (SNR)
        # Ưu tiên SNR cao (tín hiệu tốt)
        reward += w['snr_reward'] * snr_score

        # C. Trạng thái Node (Load Balancing)
        # Queue score càng cao (đầy) -> phạt
        reward += w['node_load'] * queue_score 
        # Ưu tiên CPU rảnh (cpu_score thấp)
        reward += w['utilization'] * (1.0 - cpu_score)

        # D. Độ tin cậy
        reward += w['reliability'] * (1.0 - loss_score)
        reward += w['operational'] * is_op

        # E. QoS (Latency)
        # Kiểm tra tổng latency tích lũy
        curr_delay = new_packet.get('accumulatedDelayMs', 0)
        max_lat = new_packet.get('serviceQoS', {}).get('maxLatencyMs', MAX_SYSTEM_LATENCY_MS)
        
        if max_lat > 0:
            ratio = curr_delay / max_lat
            if ratio > 1.0:
                reward += w['latency_violation']
            elif ratio > 0.8:
                reward += w['latency_violation'] * 0.5
        
        # Phạt delay của node này
        reward += w['latency'] * delay_score

        return float(reward)