# env/satellite_simulator.py

from python.utils.state_builder import StateBuilder
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from python.utils.state_builder import (
    StateBuilder, 
    MAX_NEIGHBORS,             # Import hằng số trực tiếp
    MAX_SYSTEM_LATENCY_MS,     # Import hằng số trực tiếp
)

# Giả định các hằng số trọng số
DEFAULT_WEIGHTS = {
    'goal': 10.0,
    'drop': 100.0,
    'latency': -5.0,        # Phạt cho độ trễ hop cao
    'latency_violation': -50.0, # Phạt vượt ngưỡng QoS
    'utilization': 2.0,     # Thưởng tài nguyên thấp
    'bandwidth': 1.0,       # Thưởng băng thông khả dụng cao
    'reliability': 3.0,     # Thưởng độ tin cậy cao
    'fspl': -0.1            # Phạt suy hao cao
}

class SatelliteEnv:
    """Môi trường mô phỏng hop định tuyến trong mạng vệ tinh."""

    def __init__(self, state_builder: StateBuilder, weights: Optional[Dict[str, float]] = None):
        """
        :param weights: Dict các hệ số cho reward (goal, drop, delay, utilization, bandwidth, fspl, packet_loss, operational)
        """
        self.state_builder = state_builder
        # Tối ưu: Merge weights với default nếu weights không phải là None
        self.weights = DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        # FIX PY LANCE: Khởi tạo là Dict rỗng, sau đó được gán trong reset/step
        self.current_packet_state: Dict[str, Any] = {} 

    def reset(self, initial_packet_data: Dict[str, Any]) -> np.ndarray:
        """Khởi tạo trạng thái ban đầu (Vector S)"""
        self.current_packet_state = initial_packet_data
        return self.state_builder.get_state_vector(initial_packet_data)

    def step(self, action_index: int, neighbor_id: str, new_packet_data: Dict[str, Any]) -> Tuple[np.ndarray, float, bool]:
        """
        Thực hiện hành động, trả về (next_state, reward, done)
        """
        # Tối ưu: Truyền action_index để tính toán các chỉ số neighbor từ S cũ
        reward = self._calculate_reward(action_index, new_packet_data) 
        
        # Cập nhật trạng thái
        self.current_packet_state = new_packet_data
        next_state = self.state_builder.get_state_vector(new_packet_data)
        done = self._is_terminal(new_packet_data)
        
        return next_state, reward, done

    def _is_terminal(self, packet_data: Dict[str, Any]) -> bool:
        """Kiểm tra gói tin đã đến đích hoặc bị drop / TTL hết"""
        is_at_dest = packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest')
        is_dropped = packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0
        return is_at_dest or is_dropped

    # ================== Hàm Reward Tối ưu ==================

    def _calculate_reward(self, action_index: int, packet_data: Dict[str, Any]) -> float:
        """
        Tính reward cân bằng.
        LƯU Ý: Hàm này chỉ dựa vào trạng thái cũ (self.current_packet_state) và trạng thái gói tin mới.
        """
        w = self.weights
        reward = 0.0

        # Lấy chỉ số neighbor từ Vector S TRƯỚC HÀNH ĐỘNG (current_packet_state)
        # Giả định StateBuilder đã được chạy trên current_packet_state (từ hàm reset/step trước)
        prev_S = self.state_builder.get_state_vector(self.current_packet_state)
        
        # Tính toán vị trí bắt đầu của slot neighbor được chọn trong Vector S (V_N)
        NEIGHBOR_SLOT_SIZE = 7 # Kích thước 1 slot trong V_N
        V_G_SIZE = 10
        V_P_SIZE = 8
        V_C_SIZE = 6
        START_INDEX_VN = V_G_SIZE + V_P_SIZE + V_C_SIZE # 10 + 8 + 6 = 24
        SLOT_START = START_INDEX_VN + (action_index * NEIGHBOR_SLOT_SIZE)
        
        # Nếu action_index không hợp lệ (ví dụ: slot padding), phạt nhẹ
        if action_index >= MAX_NEIGHBORS:
             return -w.get('drop', 100.0) / 2 # Giảm nửa phạt
        
        # Trích xuất các chỉ số quan trọng của hop được chọn (ĐÃ CHUẨN HÓA)
        try:
            is_op = prev_S[SLOT_START]
            total_latency_ratio = prev_S[SLOT_START + 1]
            avail_bw_ratio = prev_S[SLOT_START + 2]
            dest_util_ratio = prev_S[SLOT_START + 3]
            loss_rate_neighbor = prev_S[SLOT_START + 4]
            fspl_ratio = prev_S[SLOT_START + 6]
        except IndexError:
             # Lỗi Index nếu kích thước S không khớp (bắt lỗi trong môi trường thực)
             return -w.get('drop', 100.0)

        # --- 1. Goal / Drop (Phần thưởng/Phạt cuối cùng) ---
        if packet_data.get('currentHoldingNodeId') == packet_data.get('stationDest'):
            return w.get('goal', 10.0) # TRẢ VỀ NGAY nếu ĐẾN ĐÍCH
        
        if packet_data.get('dropped', False) or packet_data.get('ttl', 0) <= 0:
            return -w.get('drop', 100.0) # TRẢ VỀ NGAY nếu DROP

        # --- 2. Phạt Độ trễ (Tránh Vi phạm QoS) ---
        max_lat = packet_data.get('serviceQoS', {}).get('maxLatencyMs', MAX_SYSTEM_LATENCY_MS)
        curr_delay = packet_data.get('accumulatedDelayMs', 0.0)
        
        # Phạt nếu độ trễ tích lũy VƯỢT QUÁ 90% ngưỡng QoS
        if curr_delay / max_lat > 0.9:
             reward += w.get('latency_violation', -50.0)

        # Phạt/Thưởng dựa trên Độ trễ Hop (Dùng giá trị ratio đã chuẩn hóa)
        # Vì total_latency_ratio đã là chỉ số tốt, ta phạt nó.
        reward += w.get('latency', -5.0) * total_latency_ratio
        
        # --- 3. Cân bằng Tài nguyên (Thưởng/Phạt Hop được chọn) ---
        
        # Thưởng khi băng thông khả dụng cao
        reward += w.get('bandwidth', 1.0) * avail_bw_ratio
        
        # Thưởng khi node lân cận không tắc nghẽn (utilization thấp)
        reward += w.get('utilization', 2.0) * (1.0 - dest_util_ratio)
        
        # --- 4. Độ tin cậy (Reliability) ---
        
        # Phạt mất gói (tỷ lệ mất gói cao)
        reward += w.get('reliability', 3.0) * (1.0 - loss_rate_neighbor)
        
        # Phạt suy hao (FSPL cao)
        reward += w.get('fspl', -0.1) * fspl_ratio

        # Phạt nếu node không hoạt động (is_op là 0.0 nếu không hoạt động)
        reward += w.get('operational', 5.0) * is_op 
        
        return float(reward)