# env/link_metrics_calculator.py
import data.mongo_manager as MongoManager  # Assume this is correct
from typing import Dict
import random
import math  # Added for potential calculations

class LinkMetricsCalculator:
    def __init__(self, mongo_manager: MongoManager.MongoManager):  # Fixed type hint
        self.mongo_manager = mongo_manager
        
    def calculate_link_metrics(self, from_node: Dict, to_node: Dict, packet_qos: Dict) -> Dict:
        """Tính toán chỉ số link metric giữa hai node dựa trên QoS yêu cầu"""
        distance_km = self.mongo_manager.calculate_distance(from_node['position'], to_node['position'])
        latency_ms = (distance_km / 300) * 1000  # Giả sử tốc độ ánh sáng trong chân không ~300,000 km/s
        bandwidth_mbps = min(from_node['communication']['bandwidthMHz'], to_node['communication']['bandwidthMHz']) * 0.8
        loss_rate = distance_km / 20000 + random.uniform(0, 0.05) + (1 - from_node['status']['batteryChargePercent']/100) * 0.1
        active = (distance_km <= from_node['communication']['maxRangeKm'] and from_node['status']['active'] and to_node['status']['active'])
        
        metrics = {
            "distanceKm": distance_km,
            "latencyMs": latency_ms,
            "availableBandwidthMbps": bandwidth_mbps,
            "packetLossRate": min(loss_rate, 1.0),  # Cap loss_rate at 1.0
            "isActive": active
        }
        return metrics