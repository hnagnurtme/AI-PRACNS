import numpy as np
from .constants import (
    MAX_DIST_KM, MAX_BW_MHZ, MIN_SNR_DB, MAX_SNR_DB,
    MAX_PROCESSING_DELAY_MS, MAX_NEIGHBORS, DEFAULT_BUFFER_CAPACITY,
    NEIGHBOR_FEAT_SIZE
)
from .math_utils import to_cartesian_ecef, calculate_link_budget_snr
from simulation.core.packet import Packet

class StateBuilder:
    def __init__(self, db_connector):
        self.db = db_connector
        self.MAX_NEIGHBORS = MAX_NEIGHBORS
        self.PAD_VEC = np.array(
            [0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )

        assert len(self.PAD_VEC) == NEIGHBOR_FEAT_SIZE, \
            f"PAD_VEC size {len(self.PAD_VEC)} != NEIGHBOR_FEAT_SIZE {NEIGHBOR_FEAT_SIZE}"

        self.TOTAL_STATE_SIZE = 14 + 8 + (MAX_NEIGHBORS * NEIGHBOR_FEAT_SIZE)

    from typing import Optional

    def get_state_vector(self, packet: Packet, dynamic_neighbors: Optional[list] = None) -> np.ndarray:
        return self.build(packet, dynamic_neighbors)

    def build(self, packet: Packet, dynamic_neighbors: list = None) -> np.ndarray:
        curr_node_id = packet.current_holding_node_id
        dest_node_id = packet.station_dest
        
        curr_node = self.db.get_node(curr_node_id)
        dest_node = self.db.get_node(dest_node_id)
        
        if not curr_node or not dest_node:
            return np.zeros(self.TOTAL_STATE_SIZE, dtype=np.float32)

        try:
            curr_pos = to_cartesian_ecef(curr_node.get('position', {}))
            dest_pos = to_cartesian_ecef(dest_node.get('position', {}))
        except Exception:
            return np.zeros(self.TOTAL_STATE_SIZE, dtype=np.float32)
        
        vec_to_dest = dest_pos - curr_pos
        dist_to_dest = np.linalg.norm(vec_to_dest)
        dir_to_dest = vec_to_dest / (dist_to_dest + 1e-6)

        comm = curr_node.get('communication', {})
        cap = max(curr_node.get('packetBufferCapacity', DEFAULT_BUFFER_CAPACITY) or DEFAULT_BUFFER_CAPACITY, 1.0)
        
        norm_ttl = packet.ttl / 50.0
        max_lat = packet.service_qos.max_latency_ms
        norm_delay = min(packet.accumulated_delay_ms / max_lat, 1.0)

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

        if dynamic_neighbors is not None:
            neighbor_ids = dynamic_neighbors
        else:
            neighbor_ids = curr_node.get('neighbors', [])
        neighbors_raw = self.db.get_nodes(neighbor_ids)
        
        candidates = []

        for n in neighbors_raw:
            if not n: continue

            n_pos = to_cartesian_ecef(n.get('position', {}))
            vec_to_n = n_pos - curr_pos
            dist_to_n = np.linalg.norm(vec_to_n)
            
            n_to_dest_pos = dest_pos - n_pos
            dist_n_to_dest = np.linalg.norm(n_to_dest_pos)
            
            vec_to_n_unit = vec_to_n / (dist_to_n + 1e-6)
            cosine_sim = np.dot(vec_to_n_unit, dir_to_dest)

            weather = curr_node.get('weather', 'CLEAR')
            snr = calculate_link_budget_snr(curr_node, n, float(dist_to_n), weather)
            norm_snr = np.clip((snr - MIN_SNR_DB) / (MAX_SNR_DB - MIN_SNR_DB), 0.0, 1.0)

            n_cap = max(n.get('packetBufferCapacity', DEFAULT_BUFFER_CAPACITY) or DEFAULT_BUFFER_CAPACITY, 1.0)
            n_type = n.get('nodeType', 'UNKNOWN')

            feat = (
                1.0,
                min(dist_n_to_dest / MAX_DIST_KM, 1.0),
                cosine_sim,
                norm_snr,
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

        candidates.sort(key=lambda x: x[0])

        neighbor_matrix = np.tile(self.PAD_VEC, (MAX_NEIGHBORS, 1))

        num_fill = min(len(candidates), MAX_NEIGHBORS)
        for i in range(num_fill):
            neighbor_matrix[i] = candidates[i][1]

        neighbor_flat = neighbor_matrix.flatten()

        full_state = np.concatenate([self_state, dest_state, neighbor_flat])
        
        return full_state.astype(np.float32)