# env/packet.py
import base64
import time
from typing import List, Dict
from data.mongo_manager import MongoManager
import logging

logger = logging.getLogger(__name__)

class Packet:
    def __init__(self, source_user: str, dest_user: str, service_type: str, payload: str, qos: Dict, client_a_pos: Dict, client_b_pos: Dict, mongo_manager: MongoManager):
        self.packet_id = f"PKT-{int(time.time() * 1000)}"
        self.source_user_id = source_user
        self.destination_user_id = dest_user
        # Assume get_closest_gs returns node ID (str), but we'll fetch dict when needed
        self.station_source = mongo_manager.get_closest_gs(client_a_pos)  # str
        self.station_dest = mongo_manager.get_closest_gs(client_b_pos)  # str
        self.type = "DATA"
        self.acknowledged_packet_id = None
        self.time_sent_ms = int(time.time() * 1000)
        self.payload_base64 = base64.b64encode(payload.encode()).decode()
        self.payload_size = len(payload)
        self.service_type = service_type
        self.service_qos = qos
        self.ttl = 10
        self.current_holding_node_id = self.station_source
        self.next_hop_node_id = None
        self.path_history: List[str] = [source_user, self.station_source]
        self.hop_records: List[Dict] = []
        self.accumulated_delay_ms = 0.0
        self.accumulated_loss_rate = 0.0  # Added: to accumulate loss (1 - prod(1 - loss_i))
        self.min_bandwidth_mbps = float('inf')  # Added: min bandwidth along path
        self.priority_level = qos.get('defaultPriority', 1)
        self.is_use_rl = True
        self.max_acceptable_latency_ms = qos.get('maxLatencyMs', 150.0)
        self.max_acceptable_loss_rate = qos.get('maxLossRate', 0.01)
        self.dropped = False
        self.drop_reason = None
    
    def update_hop(self, from_node: str, to_node: str, latency_ms: float, bandwidth_mbps: float, loss_rate: float, timestamp_ms: int):
        if self.ttl <= 0:
            self.dropped = True
            self.drop_reason = "TTL expired"
            return 
        self.hop_records.append({
            "fromNodeId": from_node,
            "toNodeId": to_node,
            "latencyMs": latency_ms,
            "bandwidthMbps": bandwidth_mbps,  # Added
            "lossRate": loss_rate,  # Added
            "timestampMs": timestamp_ms
        })
        self.accumulated_delay_ms += latency_ms
        self.min_bandwidth_mbps = min(self.min_bandwidth_mbps, bandwidth_mbps)  # Update min bandwidth
        self.accumulated_loss_rate = 1 - (1 - self.accumulated_loss_rate) * (1 - loss_rate)  # Cumulative loss
        self.path_history.append(to_node)
        self.current_holding_node_id = to_node
        self.ttl -= 1
        
    def is_at_dest(self) -> bool:
        return self.current_holding_node_id == self.station_dest
    
    def to_dict(self) -> Dict:
        data = vars(self)
        data['path'] = self.path_history  # Added for output
        data['total_latency_ms'] = self.accumulated_delay_ms
        data['path_bandwidth_mbps'] = self.min_bandwidth_mbps if self.min_bandwidth_mbps != float('inf') else 0
        data['path_loss_rate'] = self.accumulated_loss_rate
        return data
    
    @classmethod
    def from_dict(cls, data: Dict, mongo_manager: MongoManager):
        qos = data.get('service_qos', {})
        client_a_pos = data.get('client_a_pos', {})  # Fixed typo: clint -> client
        client_b_pos = data.get('client_b_pos', {})
        payload = base64.b64decode(data.get('payload_base64', '')).decode()
        packet = cls(
            data.get('source_user_id', 'UNKNOWN'),
            data.get('destination_user_id', 'UNKNOWN'),
            data.get('service_type', 'UNKNOWN'),
            payload,
            qos,
            client_a_pos,
            client_b_pos,
            mongo_manager
        )
        packet.packet_id = data.get('packet_id', packet.packet_id)
        packet.station_source = data.get('station_source', packet.station_source)
        packet.station_dest = data.get('station_dest', packet.station_dest)
        packet.current_holding_node_id = data.get('current_holding_node_id', packet.station_source)
        packet.path_history = data.get('path_history', packet.path_history)
        packet.hop_records = data.get('hop_records', packet.hop_records)
        packet.accumulated_delay_ms = data.get('accumulated_delay_ms', packet.accumulated_delay_ms)
        packet.ttl = data.get('ttl', packet.ttl)
        packet.dropped = data.get('dropped', packet.dropped)
        packet.drop_reason = data.get('drop_reason', packet.drop_reason)
        packet.min_bandwidth_mbps = data.get('min_bandwidth_mbps', float('inf'))  # Added
        packet.accumulated_loss_rate = data.get('accumulated_loss_rate', 0.0)  # Added
        return packet