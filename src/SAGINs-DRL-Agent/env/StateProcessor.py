# env/StateProcessor.py
import numpy as np
from typing import Dict, Any

class StateProcessor:
    
    def __init__(self, max_neighbors: int = 10):
        self.MAX_NEIGHBORS = max_neighbors
        self.BASE_FEATURES_SIZE = 6 
        self.LINK_FEATURES_PER_NEIGHBOR = 4
        self.STATE_SIZE = self.BASE_FEATURES_SIZE + (self.MAX_NEIGHBORS * self.LINK_FEATURES_PER_NEIGHBOR)

    def json_to_state_vector(self, data: Dict[str, Any]) -> np.ndarray:
        
        # 1. QoS + 2. Source Load
        qos = data.get('targetQoS', {}); src_info = data.get('sourceNodeInfo', {})
        qos_vector = [qos.get('maxLatencyMs', 100), qos.get('minBandwidthMbps', 100), qos.get('maxLossRate', 0.1)]
        
        current_packet_count = src_info.get('currentPacketCount', 0)
        packet_buffer_capacity = src_info.get('packetBufferCapacity', 1)
        buffer_load_ratio = current_packet_count / packet_buffer_capacity if packet_buffer_capacity > 0 else 0.0

        source_vector = [src_info.get('resourceUtilization', 0.1), src_info.get('batteryChargePercent', 100), buffer_load_ratio]
        
        # 3. Link Metrics
        link_vector = []
        neighbor_links = data.get('neighborLinkMetrics', {})
        sorted_neighbor_ids = sorted(neighbor_links.keys())
        
        for i, neighbor_id in enumerate(sorted_neighbor_ids):
            if i >= self.MAX_NEIGHBORS: break
            link = neighbor_links[neighbor_id]
            link_vector.extend([link.get('latencyMs', 1000.0), link.get('currentAvailableBandwidthMbps', 0.0), link.get('packetLossRate', 1.0), link.get('linkScore', 0.0)])

        # Padding
        expected_link_vector_size = self.MAX_NEIGHBORS * self.LINK_FEATURES_PER_NEIGHBOR
        padding_needed = expected_link_vector_size - len(link_vector)
        if padding_needed > 0: link_vector.extend([0.0] * padding_needed)
            
        state_vector = np.array(qos_vector + source_vector + link_vector, dtype=np.float32)
        
        return state_vector[:self.STATE_SIZE]