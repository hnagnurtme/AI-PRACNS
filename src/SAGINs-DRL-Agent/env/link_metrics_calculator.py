# env/link_metrics_calculator.py
import data.mongo_manager as MongoManager
from typing import Dict
import random
import math

# Hằng số (Constants)
SPEED_OF_LIGHT_KM_PER_S = 299792.458  # km/s
SPEED_OF_LIGHT_KM_PER_MS = SPEED_OF_LIGHT_KM_PER_S / 1000 # ~300 km/ms
BANDWIDTH_EFFICIENCY = 0.8 # Hệ số hiệu suất cho băng thông


class LinkMetricsCalculator:
    def __init__(self, mongo_manager: MongoManager.MongoManager):
        self.mongo_manager = mongo_manager
        
    def calculate_link_metrics(self, from_node: Dict, to_node: Dict, packet_qos: Dict) -> Dict:
        """Tính toán chỉ số link metric giữa hai node dựa trên QoS yêu cầu"""
        
        # 1. TÍNH KHOẢNG CÁCH VÀ ĐỘ TRỄ
        distance_km = self.mongo_manager.calculate_distance(from_node['position'], to_node['position'])
        # Độ trễ (Latency) = Khoảng cách / Tốc độ truyền (ms)
        # 300 là tốc độ ánh sáng xấp xỉ 300 km/ms
        latency_ms = (distance_km / SPEED_OF_LIGHT_KM_PER_MS) 
        
        # 2. TÍNH BĂNG THÔNG
        # Giả định: Băng thông có sẵn = min(Băng thông của 2 node) * Hiệu suất
        # Đơn vị: bandwidthMHz (MHz) -> availableBandwidthMbps (Mbps)
        # Giả sử 1 MHz ~ 1 Mbps trong mô hình đơn giản này
        min_bandwidth_mhz = min(
            from_node.get('communication', {}).get('bandwidthMHz', 0), 
            to_node.get('communication', {}).get('bandwidthMHz', 0)
        )
        bandwidth_mbps = min_bandwidth_mhz * BANDWIDTH_EFFICIENCY

        # 3. TÍNH TỶ LỆ MẤT GÓI (SỬA LỖI KEY ACCESS)
        # Giả định: Mất gói tăng theo khoảng cách và giảm theo pin
        battery_percent = from_node.get('batteryChargePercent', 100) / 100 
        
        loss_rate = (distance_km / 20000) \
                  + random.uniform(0, 0.05) \
                  + (1 - battery_percent) * 0.1
        
        # 4. TRẠNG THÁI HOẠT ĐỘNG (SỬA LỖI KEY ACCESS)
        # Key chính xác là 'operational', không phải 'status'
        is_operational_from = from_node.get('operational', False)
        is_operational_to = to_node.get('operational', False)
        max_range_km = from_node.get('communication', {}).get('maxRangeKm', 0)
        
        active = (distance_km <= max_range_km) \
                 and is_operational_from \
                 and is_operational_to
        
        metrics = {
            "distanceKm": distance_km,
            "latencyMs": latency_ms,
            "availableBandwidthMbps": bandwidth_mbps,
            "packetLossRate": min(loss_rate, 1.0),
            "isActive": active
        }
        return metrics