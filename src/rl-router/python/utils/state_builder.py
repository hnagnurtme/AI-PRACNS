# utils/state_builder.py

import numpy as np
import math
import time
from typing import Dict, List, Any, Tuple
from .db_connector import MongoConnector 
from datetime import datetime

# --- I. HẰNG SỐ VÀ CÁC GIÁ TRỊ CHUẨN HÓA ---

SPEED_OF_LIGHT_KM_PER_S = 299792.458
MAX_SYSTEM_DISTANCE_KM = 50000.0  
MAX_SYSTEM_LATENCY_MS = 2000.0
MAX_LINK_BANDWIDTH_MBPS = 500.0
MAX_PROCESSING_DELAY_MS = 250.0
MAX_STALENESS_SEC = 30.0  # Threshold for data staleness

MAX_NEIGHBORS = 4 
NUM_SERVICE_TYPES = 5
NUM_NODE_TYPES = 4 

SERVICE_TYPE_MAP = {"VIDEO_STREAM": 0, "AUDIO_CALL": 1, "IMAGE_TRANSFER": 2, "FILE_TRANSFER": 3, "TEXT_MESSAGE": 4}
NODE_TYPE_MAP = {"GROUND_STATION": 0, "LEO_SATELLITE": 1, "MEO_SATELLITE": 2, "GEO_SATELLITE": 3}

# Kích thước cố định cho mỗi slot neighbor
NEIGHBOR_SLOT_SIZE = 7


# --- II. HÀM HELPER VẬT LÝ VÀ MÃ HÓA (Giữ nguyên) ---

def convert_to_ecef(pos: Dict[str, float]) -> Tuple[float, float, float]:
    """Chuyển đổi vị trí thô sang ECEF (Placeholder)."""
    return pos.get('longitude', 0.0), pos.get('latitude', 0.0), pos.get('altitude', 0.0)

def calculate_distance_km(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Tính khoảng cách Euclidean 3D."""
    dx, dy, dz = pos2[0]-pos1[0], pos2[1]-pos1[1], pos2[2]-pos1[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def calculate_direction_vector(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> List[float]:
    """Tính vector đơn vị chỉ hướng."""
    dist = calculate_distance_km(pos1, pos2)
    if dist == 0: return [0.0, 0.0, 0.0]
    return [(pos2[i]-pos1[i])/dist for i in range(3)]

def calculate_fspl_attenuation_db(distance_km: float, freq_ghz: float) -> float:
    """Tính suy hao đường truyền cơ bản (FSPL)."""
    if distance_km <= 0 or freq_ghz <= 0: return 0.0
    return 20 * math.log10(distance_km) + 20 * math.log10(freq_ghz) + 92.45

def get_ohe_vector(value: str, mapping: Dict[str,int], size: int) -> List[float]:
    """Tạo vector One-Hot Encoding."""
    vector = [0.0] * size
    idx = mapping.get(value)
    if idx is not None and 0 <= idx < size:
        vector[idx] = 1.0
    return vector


# --- III. STATE BUILDER (TỐI ƯU HÓA) ---

class StateBuilder:
    """Xây dựng vector trạng thái chuẩn hóa, tối ưu hóa hiệu suất truy cập."""
    
    def __init__(self, mongo_connector: MongoConnector):
        self.db = mongo_connector

    # Rút gọn logic tính toán độ cũ
    def _calculate_staleness(self, last_updated: Any) -> float:
        """Tính toán tỷ lệ độ cũ của dữ liệu."""
        now_ms = time.time() * 1000
        last_ms = now_ms
        if isinstance(last_updated, datetime):
            last_ms = last_updated.timestamp() * 1000
        
        staleness_sec = (now_ms - last_ms) / 1000.0
        return min(staleness_sec / MAX_STALENESS_SEC, 1.0)

    def get_state_vector(self, packet_data: Dict[str, Any]) -> np.ndarray:
        S = []

        # --- 1. FETCH VÀ XỬ LÝ DỮ LIỆU ĐẦU VÀO ---
        current_id = packet_data.get('currentHoldingNodeId')
        dest_id = packet_data.get('stationDest')
        if not isinstance(current_id, str):
            raise ValueError("Lỗi: 'currentHoldingNodeId' bị thiếu hoặc không phải chuỗi.")
        
        if not isinstance(dest_id, str):
            raise ValueError("Lỗi: 'stationDest' bị thiếu hoặc không phải chuỗi.")

        # Sử dụng get_node để fetch dữ liệu chi tiết
        current_node = self.db.get_node(current_id)
        dest_node = self.db.get_node(dest_id, projection={"position":1})
        
        if not current_node or not dest_node:
            raise ValueError("Node hiện tại hoặc đích đến không tồn tại.")

        # Lấy Neighbor Data
        neighbor_ids = current_node.get('neighbors', [])
        neighbors_data = self.db.get_neighbor_status_batch(neighbor_ids)

        # Vị trí ECEF (Sẽ được dùng lặp lại)
        cur_pos = convert_to_ecef(current_node.get('position', {}))
        dest_pos = convert_to_ecef(dest_node.get('position', {}))
        
        # Giá trị Packet cốt lõi
        qos = packet_data.get('serviceQoS', {})
        acc_delay = packet_data.get('accumulatedDelayMs', 0.0)
        max_lat = qos.get('maxLatencyMs', MAX_SYSTEM_LATENCY_MS)
        
        # Tối ưu hóa: Lấy các giá trị current_node cần dùng lặp lại
        cur_proc_delay = current_node.get('nodeProcessingDelayMs', 0.0)

        # --- A. QoS + Tiến trình (V_G) ---
        S.extend(get_ohe_vector(qos.get('serviceType','UNKNOWN'), SERVICE_TYPE_MAP, NUM_SERVICE_TYPES))
        S.extend([
            qos.get('maxLatencyMs',0.0) / MAX_SYSTEM_LATENCY_MS,
            qos.get('minBandwidthMbps',0.0) / MAX_LINK_BANDWIDTH_MBPS,
            qos.get('maxLossRate',0.0),
            min(acc_delay/max_lat,1.0) if max_lat > 0 else 0.0,
            packet_data.get('ttl',10) / 20.0
        ])

        # --- B. Vị trí + Hướng (V_P) ---
        dist_to_dest = calculate_distance_km(cur_pos, dest_pos)
        dir_vec = calculate_direction_vector(cur_pos, dest_pos)

        S.extend(get_ohe_vector(current_node.get('nodeType','UNKNOWN'), NODE_TYPE_MAP, NUM_NODE_TYPES))
        S.extend([
            dist_to_dest / MAX_SYSTEM_DISTANCE_KM,
            *dir_vec # Mở rộng vector hướng
        ])

        # --- C. Trạng thái Node hiện tại (V_C) ---
        curr_count = current_node.get('currentPacketCount', 0)
        capacity = current_node.get('packetBufferCapacity', 1)
        
        S.extend([
            current_node.get('resourceUtilization', 0.0),
            curr_count / capacity,
            current_node.get('packetLossRate', 0.0),
            cur_proc_delay / MAX_PROCESSING_DELAY_MS,
            self._calculate_staleness(current_node.get('lastUpdated')),
            1.0 if current_node.get('operational', False) else 0.0
        ])

        # --- D. Trạng thái Neighbors (V_N) ---
        k_slots = []
        
        # Tối ưu hóa: Lặp qua chỉ số cố định để đảm bảo kích thước vector
        for i in range(MAX_NEIGHBORS):
            if i < len(neighbor_ids):
                # Tải dữ liệu Neighbor
                n_data = neighbors_data.get(neighbor_ids[i], {})
                
                # Tránh lỗi nếu dữ liệu neighbor bị thiếu
                if not n_data:
                    k_slots.extend([0.0] * NEIGHBOR_SLOT_SIZE)
                    continue 

                n_pos = convert_to_ecef(n_data.get('position', {}))
                
                # Tính toán Metrics
                prop_dist = calculate_distance_km(cur_pos, n_pos)
                prop_latency = prop_dist / SPEED_OF_LIGHT_KM_PER_S * 1000
                
                # Total Latency = Prop + Source Proc + Dest Proc
                total_latency = prop_latency + cur_proc_delay + n_data.get('nodeProcessingDelayMs', 0.0)
                
                # Băng thông
                max_bw = n_data.get('communication', {}).get('bandwidthMHz', MAX_LINK_BANDWIDTH_MBPS)
                utilization = n_data.get('resourceUtilization', 0.0)
                avail_bw = max_bw * (1.0 - utilization)
                
                # Suy hao
                freq = n_data.get('communication', {}).get('frequencyGHz', 10.0)
                fspl = calculate_fspl_attenuation_db(prop_dist, freq)

                # Add Slot Data
                k_slots.extend([
                    1.0 if n_data.get('operational', False) else 0.0,
                    total_latency / MAX_SYSTEM_LATENCY_MS,
                    avail_bw / MAX_LINK_BANDWIDTH_MBPS,
                    utilization,
                    n_data.get('packetLossRate', 0.0),
                    calculate_distance_km(n_pos, dest_pos) / MAX_SYSTEM_DISTANCE_KM,
                    fspl / 300.0 # Chuẩn hóa FSPL
                ])
            else:
                # Padding bằng 0
                k_slots.extend([0.0] * NEIGHBOR_SLOT_SIZE)

        S.extend(k_slots)
        
        # --- 3. OUTPUT ---
        return np.array(S, dtype=np.float32)