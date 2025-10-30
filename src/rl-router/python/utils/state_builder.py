import numpy as np
import math
import time
from typing import Dict, List, Any, Tuple
from .db_connector import MongoConnector
from datetime import datetime

# --- I. HẰNG SỐ & CHUẨN HÓA ---

SPEED_OF_LIGHT_KM_PER_S = 299792.458
MAX_SYSTEM_DISTANCE_KM = 50000.0
MAX_SYSTEM_LATENCY_MS = 2000.0
MAX_LINK_BANDWIDTH_MBPS = 500.0
MAX_PROCESSING_DELAY_MS = 250.0
MAX_STALENESS_SEC = 30.0

MAX_NEIGHBORS = 4
NUM_SERVICE_TYPES = 5
NUM_NODE_TYPES = 4
NEIGHBOR_SLOT_SIZE = 7

SERVICE_TYPE_MAP = {
    "VIDEO_STREAM": 0, "AUDIO_CALL": 1, "IMAGE_TRANSFER": 2, 
    "FILE_TRANSFER": 3, "TEXT_MESSAGE": 4
}

NODE_TYPE_MAP = {
    "GROUND_STATION": 0, "LEO_SATELLITE": 1, 
    "MEO_SATELLITE": 2, "GEO_SATELLITE": 3
}


# --- II. HELPER ---

def convert_to_ecef(pos: Dict[str, float]) -> Tuple[float, float, float]:
    """Placeholder: chuyển Lat/Lon/Alt sang ECEF."""
    return pos.get('longitude', 0.0), pos.get('latitude', 0.0), pos.get('altitude', 0.0)

def calculate_distance_km(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def calculate_direction_vector(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> List[float]:
    dist = calculate_distance_km(p1, p2)
    if dist == 0: return [0.0, 0.0, 0.0]
    return [(p2[i]-p1[i])/dist for i in range(3)]

def calculate_fspl_attenuation_db(distance_km: float, freq_ghz: float) -> float:
    if distance_km <= 0 or freq_ghz <= 0: return 0.0
    return 20*math.log10(distance_km) + 20*math.log10(freq_ghz) + 92.45

def get_ohe_vector(value: str, mapping: Dict[str,int], size: int) -> List[float]:
    vec = [0.0]*size
    idx = mapping.get(value)
    if idx is not None and 0 <= idx < size:
        vec[idx] = 1.0
    return vec


# --- III. STATE BUILDER ---

class StateBuilder:
    """Xây dựng vector trạng thái chuẩn hóa cho RL."""

    def __init__(self, mongo_connector: MongoConnector):
        self.db = mongo_connector

    def _staleness_ratio(self, last_updated: Any) -> float:
        now_ms = time.time()*1000
        last_ms = now_ms
        if isinstance(last_updated, datetime):
            last_ms = last_updated.timestamp()*1000
        staleness = (now_ms - last_ms)/1000.0
        return min(staleness/MAX_STALENESS_SEC, 1.0)

    def get_state_vector(self, packet_data: Dict[str, Any]) -> np.ndarray:
        S = []

        current_id = packet_data.get('currentHoldingNodeId')
        dest_id = packet_data.get('stationDest')
        if not current_id or not dest_id:
            raise ValueError("Node hiện tại hoặc đích đến không hợp lệ.")

        current_node = self.db.get_node(current_id)
        dest_node = self.db.get_node(dest_id, projection={"position":1})
        if not current_node or not dest_node:
            raise ValueError("Node hiện tại hoặc đích đến không tồn tại.")

        neighbor_ids = current_node.get('neighbors', [])
        neighbors_data = self.db.get_neighbor_status_batch(neighbor_ids)

        cur_pos = convert_to_ecef(current_node.get('position', {}))
        dest_pos = convert_to_ecef(dest_node.get('position', {}))

        qos = packet_data.get('serviceQoS', {})
        acc_delay = packet_data.get('accumulatedDelayMs', 0.0)
        max_lat = qos.get('maxLatencyMs', MAX_SYSTEM_LATENCY_MS)
        cur_proc_delay = current_node.get('nodeProcessingDelayMs', 0.0)
        curr_count = current_node.get('currentPacketCount', 0)
        capacity = max(current_node.get('packetBufferCapacity',1),1)

        # --- A. QoS + Progress (V_G) ---
        S.extend(get_ohe_vector(qos.get('serviceType','UNKNOWN'), SERVICE_TYPE_MAP, NUM_SERVICE_TYPES))
        S.extend([
            min(qos.get('maxLatencyMs',0.0)/MAX_SYSTEM_LATENCY_MS,1.0),
            min(qos.get('minBandwidthMbps',0.0)/MAX_LINK_BANDWIDTH_MBPS,1.0),
            min(qos.get('maxLossRate',0.0),1.0),
            min(acc_delay/max_lat,1.0) if max_lat>0 else 0.0,
            min(packet_data.get('ttl',10)/20.0,1.0)
        ])

        # --- B. Position + Direction (V_P) ---
        dist_to_dest = calculate_distance_km(cur_pos, dest_pos)
        dir_vec = calculate_direction_vector(cur_pos, dest_pos)

        S.extend(get_ohe_vector(current_node.get('nodeType','UNKNOWN'), NODE_TYPE_MAP, NUM_NODE_TYPES))
        S.extend([
            min(dist_to_dest/MAX_SYSTEM_DISTANCE_KM,1.0),
            *dir_vec
        ])

        # --- C. Current Node Status (V_C) ---
        S.extend([
            min(current_node.get('resourceUtilization',0.0),1.0),
            min(curr_count/capacity,1.0),
            min(current_node.get('packetLossRate',0.0),1.0),
            min(cur_proc_delay/MAX_PROCESSING_DELAY_MS,1.0),
            self._staleness_ratio(current_node.get('lastUpdated')),
            1.0 if current_node.get('operational',False) else 0.0
        ])

        # --- D. Neighbor Slots (V_N) ---
        k_slots = []
        for i in range(MAX_NEIGHBORS):
            if i < len(neighbor_ids):
                n_data = neighbors_data.get(neighbor_ids[i],{})
                if not n_data:
                    k_slots.extend([0.0]*NEIGHBOR_SLOT_SIZE)
                    continue

                n_pos = convert_to_ecef(n_data.get('position', {}))
                prop_dist = calculate_distance_km(cur_pos, n_pos)
                prop_latency = prop_dist/SPEED_OF_LIGHT_KM_PER_S*1000
                total_latency = prop_latency + cur_proc_delay + n_data.get('nodeProcessingDelayMs',0.0)
                max_bw = n_data.get('communication',{}).get('bandwidthMHz',MAX_LINK_BANDWIDTH_MBPS)
                utilization = n_data.get('resourceUtilization',0.0)
                avail_bw = max_bw*(1.0-utilization)
                freq = n_data.get('communication',{}).get('frequencyGHz',10.0)
                fspl = calculate_fspl_attenuation_db(prop_dist,freq)

                k_slots.extend([
                    1.0 if n_data.get('operational',False) else 0.0,
                    min(total_latency/MAX_SYSTEM_LATENCY_MS,1.0),
                    min(avail_bw/MAX_LINK_BANDWIDTH_MBPS,1.0),
                    min(utilization,1.0),
                    min(n_data.get('packetLossRate',0.0),1.0),
                    min(calculate_distance_km(n_pos,dest_pos)/MAX_SYSTEM_DISTANCE_KM,1.0),
                    min(fspl/300.0,1.0)
                ])
            else:
                k_slots.extend([0.0]*NEIGHBOR_SLOT_SIZE)

        S.extend(k_slots)
        return np.clip(np.array(S,dtype=np.float32),0.0,1.0)
