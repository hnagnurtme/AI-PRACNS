import numpy as np
from .constants import (
    MAX_DIST_KM, MAX_BW_MHZ, MIN_SNR_DB, MAX_SNR_DB,
    MAX_PROCESSING_DELAY_MS, MAX_NEIGHBORS, DEFAULT_BUFFER_CAPACITY,
    NEIGHBOR_FEAT_SIZE
)
from .math_utils import to_cartesian_ecef, calculate_link_budget_snr

class StateBuilder:
    def __init__(self, db_connector):
        self.db = db_connector
        # Pre-allocate PAD_VEC as float32 to avoid casting later
        self.PAD_VEC = np.array(
            [0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            dtype=np.float32
        )
        
        # Validate kích thước ngay khi init
        assert len(self.PAD_VEC) == NEIGHBOR_FEAT_SIZE, \
            f"PAD_VEC size {len(self.PAD_VEC)} != NEIGHBOR_FEAT_SIZE {NEIGHBOR_FEAT_SIZE}"

        # Tính toán trước kích thước output để dùng cho placeholder
        self.TOTAL_STATE_SIZE = 14 + 8 + (MAX_NEIGHBORS * NEIGHBOR_FEAT_SIZE)

    def get_state_vector(self, packet: dict) -> np.ndarray:
        """
        Tạo State Vector chuẩn hóa (alias for build method).
        Output shape: (22 + 14*MAX_NEIGHBORS,)
        """
        return self.build(packet)

    def build(self, packet: dict) -> np.ndarray:
        """
        Tạo State Vector chuẩn hóa.
        Output shape: (22 + 14*MAX_NEIGHBORS,)
        """
        # 1. Fetch Data
        curr_node_id = packet.get('currentHoldingNodeId')
        dest_node_id = packet.get('stationDest')
        
        curr_node = self.db.get_node(curr_node_id)
        dest_node = self.db.get_node(dest_node_id)
        
        if not curr_node or not dest_node:
            return np.zeros(self.TOTAL_STATE_SIZE, dtype=np.float32)

        # 2. Geometry Calculations
        try:
            curr_pos = to_cartesian_ecef(curr_node.get('position', {}))
            dest_pos = to_cartesian_ecef(dest_node.get('position', {}))
        except Exception:
            return np.zeros(self.TOTAL_STATE_SIZE, dtype=np.float32)
        
        vec_to_dest = dest_pos - curr_pos
        dist_to_dest = np.linalg.norm(vec_to_dest)
        # Tránh chia cho 0 nếu trùng vị trí
        dir_to_dest = vec_to_dest / (dist_to_dest + 1e-6)

        # ==========================================================
        # PART A: Self State (14 features)
        # ==========================================================
        comm = curr_node.get('communication', {})
        # Safety: đảm bảo cap > 0
        cap = max(curr_node.get('packetBufferCapacity', DEFAULT_BUFFER_CAPACITY) or DEFAULT_BUFFER_CAPACITY, 1.0)
        
        norm_ttl = packet.get('ttl', 0) / 50.0
        max_lat = packet.get('serviceQoS', {}).get('maxLatencyMs', 500.0)
        norm_delay = min(packet.get('accumulatedDelayMs', 0) / max_lat, 1.0)

        self_state = np.array([
            curr_node.get('batteryChargePercent', 0) / 100.0,
            curr_node.get('currentPacketCount', 0) / cap,
            curr_node.get('resourceUtilization', 0.0),
            curr_node.get('packetLossRate', 0.0),
            min(curr_node.get('nodeProcessingDelayMs', 0) / MAX_PROCESSING_DELAY_MS, 1.0),
            1.0 if curr_node.get('isOperational') else 0.0,
            1.0 if curr_node.get('healthy') else 0.0,
            min(comm.get('bandwidthMHz', 0) / MAX_BW_MHZ, 1.0),
            dir_to_dest[0],
            dir_to_dest[1],
            dir_to_dest[2],
            min(dist_to_dest / MAX_DIST_KM, 1.0),
            norm_ttl,
            norm_delay
        ], dtype=np.float32)

        # ==========================================================
        # PART B: Destination State (8 features)
        # ==========================================================
        dest_cap = max(dest_node.get('packetBufferCapacity', DEFAULT_BUFFER_CAPACITY) or DEFAULT_BUFFER_CAPACITY, 1.0)
        node_type = dest_node.get('nodeType', 'UNKNOWN')
        
        dest_state = np.array([
            dest_node.get('resourceUtilization', 0.0),
            dest_node.get('currentPacketCount', 0) / dest_cap,
            1.0 if dest_node.get('isOperational') else 0.0,
            vec_to_dest[0] / MAX_DIST_KM,
            vec_to_dest[1] / MAX_DIST_KM,
            vec_to_dest[2] / MAX_DIST_KM,
            1.0 if node_type == 'GROUND_STATION' else 0.0,
            1.0 if 'SATELLITE' in node_type else 0.0
        ], dtype=np.float32)

        # ==========================================================
        # PART C: Neighbor State (MAX_NEIGHBORS * 14 features)
        # ==========================================================
        neighbor_ids = curr_node.get('neighbors', [])
        neighbors_raw = self.db.get_nodes(neighbor_ids)
        
        # Dùng list tuple để sort trước
        candidates = []

        for n in neighbors_raw:
            if not n: continue

            n_pos = to_cartesian_ecef(n.get('position', {}))
            vec_to_n = n_pos - curr_pos
            dist_to_n = np.linalg.norm(vec_to_n)
            
            # Vector từ Neighbor -> Dest
            n_to_dest_pos = dest_pos - n_pos
            dist_n_to_dest = np.linalg.norm(n_to_dest_pos)
            
            vec_to_n_unit = vec_to_n / (dist_to_n + 1e-6)
            cosine_sim = np.dot(vec_to_n_unit, dir_to_dest)

            weather = curr_node.get('weather', 'CLEAR')
            snr = calculate_link_budget_snr(curr_node, n, float(dist_to_n), weather)
            norm_snr = np.clip((snr - MIN_SNR_DB) / (MAX_SNR_DB - MIN_SNR_DB), 0.0, 1.0)

            n_cap = max(n.get('packetBufferCapacity', DEFAULT_BUFFER_CAPACITY) or DEFAULT_BUFFER_CAPACITY, 1.0)
            n_type = n.get('nodeType', 'UNKNOWN')

            # Feature tuple (raw values)
            feat = (
                1.0,                                      # Valid
                min(dist_n_to_dest / MAX_DIST_KM, 1.0),   # Dist to Dest
                cosine_sim,                               # Cos
                norm_snr,                                 # SNR
                n.get('batteryChargePercent', 0) / 100.0,
                n.get('currentPacketCount', 0) / n_cap,
                n.get('resourceUtilization', 0.0),
                n.get('packetLossRate', 0.0),
                min(n.get('nodeProcessingDelayMs', 0) / MAX_PROCESSING_DELAY_MS, 1.0),
                1.0 if n.get('isOperational') else 0.0,
                vec_to_n[0] / MAX_DIST_KM,
                vec_to_n[1] / MAX_DIST_KM,
                vec_to_n[2] / MAX_DIST_KM,
                1.0 if 'SATELLITE' in n_type else 0.0
            )
            candidates.append((dist_n_to_dest, feat))

        # Sorting: Gần đích nhất lên đầu
        candidates.sort(key=lambda x: x[0])

        # Pre-allocate mảng neighbor state với giá trị mặc định là PAD_VEC
        # Cách này nhanh hơn extend list rất nhiều
        neighbor_matrix = np.tile(self.PAD_VEC, (MAX_NEIGHBORS, 1)) # Shape: (MAX_NEIGHBORS, 14)

        # Fill dữ liệu thật vào matrix
        num_fill = min(len(candidates), MAX_NEIGHBORS)
        for i in range(num_fill):
            neighbor_matrix[i] = candidates[i][1] # Lấy phần feature vector

        # Flatten
        neighbor_flat = neighbor_matrix.flatten()

        # Final Concatenation (dùng np.concatenate trên list các array nhanh hơn append list)
        full_state = np.concatenate([self_state, dest_state, neighbor_flat])
        
        return full_state.astype(np.float32)