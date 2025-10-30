# env/satellite_simulator.py

from python.utils.state_builder import (
    StateBuilder, 
    MAX_NEIGHBORS,             
    MAX_SYSTEM_LATENCY_MS,     
)
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Giả định các hằng số trọng số
DEFAULT_WEIGHTS = {
    'goal': 10.0,
    'drop': 100.0,
    'latency': -5.0,        # Phạt cho độ trễ hop cao
    'latency_violation': -50.0, # Phạt vượt ngưỡng QoS
    'utilization': 2.0,     # Thưởng tài nguyên thấp
    'bandwidth': 1.0,       # Thưởng băng thông khả dụng cao
    'reliability': 3.0,     # Thưởng độ tin cậy cao
    'fspl': -0.1,           # Phạt suy hao cao
    'hop_cost': -1.0        # 💡 PHẠT MỚI: Chi phí cố định cho mỗi hop (khuyến khích đường ngắn)
}

class SatelliteEnv:
    """Môi trường mô phỏng hop định tuyến trong mạng vệ tinh."""

    def __init__(self, state_builder: StateBuilder, weights: Optional[Dict[str, float]] = None):
        """
        Khởi tạo và thiết lập các chỉ số bắt đầu cố định của Vector S.
        """
        self.state_builder = state_builder
        self.weights = DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        self.current_packet_state: Dict[str, Any] = {} 
        
        # Tính toán các chỉ số bắt đầu cố định của Vector S (ĐỘNG)
        # S = V_G (10) + V_P (8) + V_C (6) + V_N (28)
        self.NEIGHBOR_SLOT_SIZE = 7
        self.START_INDEX_VN = 10 + 8 + 6 # 24

    def reset(self, initial_packet_data: Dict[str, Any]) -> np.ndarray:
        """Khởi tạo trạng thái ban đầu (Vector S)"""
        self.current_packet_state = initial_packet_data
        return self.state_builder.get_state_vector(initial_packet_data)

    def step(self, action_index: int, neighbor_id: str, new_packet_data: Dict[str, Any]) -> Tuple[np.ndarray, float, bool]:
        """
        Thực hiện hành động, trả về (next_state, reward, done)
        """
        reward = self._calculate_reward(action_index, new_packet_data) 
        
        self.current_packet_state = new_packet_data
        next_state = self.state_builder.get_state_vector(new_packet_data)
        done = self._is_terminal(new_packet_data)
        
        return next_state, reward, done

    def _is_terminal(self, packet_data: Dict[str, Any]) -> bool:
        """Kiểm tra gói tin đã đến đích hoặc bị drop / TTL hết"""
        is_at_dest = packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest')
        is_dropped = packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0
        return is_at_dest or is_dropped

    # ================== Hàm Reward Tối ưu (FIXED) ==================

    def _calculate_reward(self, action_index: int, packet_data: Dict[str, Any]) -> float:
        """
        Tính reward cân bằng.
        """
        w = self.weights
        
        # --- 1. Goal / Drop (Phần thưởng/Phạt cuối cùng) ---
        if packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest'):
            return w.get('goal', 10.0)
        
        if packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0:
            return -w.get('drop', 100.0)

        # --- Lấy Vector S cũ và Trích xuất Neighbor Slot ---
        # NOTE: get_state_vector được gọi trên current_packet_state (trạng thái TRƯỚC khi hop)
        prev_S = self.state_builder.get_state_vector(self.current_packet_state)
        
        SLOT_START = self.START_INDEX_VN + (action_index * self.NEIGHBOR_SLOT_SIZE)

        # Kiểm tra tính hợp lệ của hành động
        if action_index >= MAX_NEIGHBORS or (SLOT_START + self.NEIGHBOR_SLOT_SIZE) > len(prev_S):
             # Phạt nặng hơn nếu chọn padding slot
             return -w.get('drop', 100.0) / 2 

        # Trích xuất 7 chỉ số (ĐÃ CHUẨN HÓA) của slot được chọn
        slot_metrics = prev_S[SLOT_START:SLOT_START + self.NEIGHBOR_SLOT_SIZE]
        
        # Ánh xạ các chỉ số (Indices)
        is_op = slot_metrics[0]
        total_latency_ratio = slot_metrics[1]
        avail_bw_ratio = slot_metrics[2]
        dest_util_ratio = slot_metrics[3]
        loss_rate_neighbor = slot_metrics[4]
        fspl_ratio = slot_metrics[6]
        
        # --- BẮT ĐẦU TÍNH TOÁN REWARD ---
        reward = 0.0
        
        # 1. 💡 PHẠT CHI PHÍ HOP (SOLUTION MỚI)
        # Phạt cố định cho mỗi hop để Agent ưu tiên đường ngắn hơn.
        reward += w.get('hop_cost', -1.0) 
        
        # 2. Phạt Độ trễ (Tránh Vi phạm QoS)
        max_lat = packet_data.get('serviceQoS', {}).get('maxLatencyMs', MAX_SYSTEM_LATENCY_MS)
        curr_delay = packet_data.get('accumulatedDelayMs', 0.0)
        
        if curr_delay / max_lat > 0.9:
             reward += w.get('latency_violation', -50.0)

        # Phạt/Thưởng dựa trên Độ trễ Hop (Vì total_latency_ratio càng cao càng xấu)
        reward += w.get('latency', -5.0) * total_latency_ratio
        
        # 3. Cân bằng Tài nguyên
        reward += w.get('bandwidth', 1.0) * avail_bw_ratio
        reward += w.get('utilization', 2.0) * (1.0 - dest_util_ratio) # Thưởng cho utilization thấp
        
        # 4. Độ tin cậy
        reward += w.get('reliability', 3.0) * (1.0 - loss_rate_neighbor) # Thưởng khi loss rate thấp
        reward += w.get('fspl', -0.1) * fspl_ratio # Phạt suy hao cao
        
        # 5. Trạng thái hoạt động
        reward += w.get('operational', 5.0) * is_op 
        
        return float(reward)