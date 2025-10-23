# env/state_processor.py
from data.mongo_manager import MongoManager  # Fixed import
from env.packet import Packet
from typing import List, Dict
import numpy as np
import logging
logger = logging.getLogger(__name__)

class StateProcessor:
    """Dùng để xử lý trạng thái môi trường cho agent DRL"""
    def __init__(self, mongo_manager: MongoManager, max_neighbors: int = 20):
        self.mongo_manager = mongo_manager
        self.max_neighbors = max_neighbors
        self.state_size = 18 + 6 + (self.max_neighbors * 10) 
        
    def get_state(self, current_node: Dict, packet: Packet, next_hops: List[Dict]) -> np.ndarray:
        try:
            # Build state vector from node and packet features
            state_features = []
            
            # Current node position (3 features)
            pos = current_node.get('position', {})
            state_features.extend([
                pos.get('latitude', 0) / 90.0,
                pos.get('longitude', 0) / 180.0,
                pos.get('altitude', 0) / 1000.0
            ])
            
            # Current node velocity (3 features)
            vel = current_node.get('velocity', {})
            state_features.extend([
                vel.get('velocityX', 0) / 10.0,
                vel.get('velocityY', 0) / 10.0,
                vel.get('velocityZ', 0) / 10.0
            ])
            
            # Communication params (3 features)
            comm = current_node.get('communication', {})
            state_features.extend([
                comm.get('bandwidthMHz', 0) / 500.0,
                comm.get('maxRangeKm', 0) / 2000.0,
                comm.get('minElevationDeg', 0) / 90.0
            ])
            
            # Packet QoS (3 features)
            qos = packet.service_qos
            state_features.extend([
                qos.get('minBandwidthMbps', 0) / 100.0,
                qos.get('maxLatencyMs', 0) / 1000.0,
                packet.accumulated_delay_ms / 1000.0
            ])
            
            # Destination info (3 features)
            # Get destination GS position
            dest_gs_id = packet.station_dest
            dest_gs = self.mongo_manager.get_node(dest_gs_id)  # ← Đây là nơi gọi get_node
            if dest_gs:
                dest_pos = dest_gs.get('position', {})
                dist_to_dest = self.mongo_manager.calculate_distance(pos, dest_pos)
                state_features.extend([
                    dest_pos.get('latitude', 0) / 90.0,
                    dest_pos.get('longitude', 0) / 180.0,
                    dist_to_dest / 20000.0  # Normalize by max earth distance
                ])
            else:
                state_features.extend([0, 0, 0])
            
            # Next hops stats (4 features)
            if next_hops:
                distances = [
                    self.mongo_manager.calculate_distance(pos, nh.get('position', {}))
                    for nh in next_hops
                ]
                state_features.extend([
                    min(distances) / 2000.0,
                    max(distances) / 2000.0,
                    sum(distances) / len(distances) / 2000.0,
                    len(next_hops) / 10.0
                ])
            else:
                state_features.extend([0, 0, 0, 0])
            
            # Padding to fixed size
            target_size = 20
            while len(state_features) < target_size:
                state_features.append(0.0)
            
            return np.array(state_features[:target_size], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error processing state for node {current_node.get('nodeId', 'unknown')}: {e}")
            return np.zeros(self.state_size, dtype=np.float32)