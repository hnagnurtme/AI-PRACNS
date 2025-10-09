# utils/static_data.py
import copy
from typing import Dict, Any, List
import random
import math

# --- HÀM TIỆN ÍCH TÍNH TOÁN (Tái tạo LinkScore) ---
def calculate_link_score_simplified(link: Dict[str, Any]) -> float:
    """Tính toán LinkScore giả định dựa trên các thông số chính."""
    if not link.get('isLinkActive', False) or link.get('currentAvailableBandwidthMbps', 0) < 10.0:
        return 0.0
    latency_cost = 1.0 + math.log(1.0 + link.get('latencyMs', 1.0))
    loss_factor = 1.0 - link.get('packetLossRate', 0.0)
    attenuation_factor = 1.0 / (1.0 + 0.05 * link.get('linkAttenuationDb', 0.0))
    score = (link['currentAvailableBandwidthMbps'] / 100.0) / latency_cost * loss_factor * attenuation_factor
    return max(0.001, score)

# --- 1. DỮ LIỆU NODE CƠ SỞ (BASELINES) ---
BASE_NODE_INFO: Dict[str, Any] = {
    # Dữ liệu Node cơ sở giữ nguyên (đã có trong file trước)
    "NodeA": { "nodeId": "NodeA", "nodeType": "GROUND_STATION", "isOperational": True, "host": "10.0.0.1", "port": 8081, "lastUpdated": 0, "position": {"latitude": 10.0, "longitude": 10.0, "altitude": 0.0}, "velocity": {"velocityX": 0.0, "velocityY": 0.0, "velocityZ": 0.0}, "nodeProcessingDelayMs": 1.5, "packetLossRate": 0.001, "weather": "CLEAR", "resourceUtilization": 0.2, "batteryChargePercent": 95, "currentPacketCount": 50, "packetBufferCapacity": 1000, },
    "NodeB": { "nodeId": "NodeB", "nodeType": "LEO_SATELLITE", "isOperational": True, "host": "10.0.0.2", "port": 8081, "lastUpdated": 0, "position": {"latitude": 20.0, "longitude": 20.0, "altitude": 500.0}, "orbit": {"semiMajorAxisKm": 6800.0, "eccentricity": 0.001, "inclinationDeg": 55.0, "raanDeg": 10.0, "argumentOfPerigeeDeg": 90.0, "trueAnomalyDeg": 180.0}, "velocity": {"velocityX": 7.5, "velocityY": 0.5, "velocityZ": 0.1}, "nodeProcessingDelayMs": 0.1, "packetLossRate": 0.0, "weather": "CLEAR", "resourceUtilization": 0.1, "batteryChargePercent": 90, "currentPacketCount": 30, "packetBufferCapacity": 1000, },
    "NodeC": { "nodeId": "NodeC", "nodeType": "LEO_SATELLITE", "isOperational": True, "host": "10.0.0.3", "port": 8081, "lastUpdated": 0, "position": {"latitude": 30.0, "longitude": 10.0, "altitude": 500.0}, "orbit": {"semiMajorAxisKm": 6800.0, "eccentricity": 0.001, "inclinationDeg": 55.0, "raanDeg": 10.0, "argumentOfPerigeeDeg": 270.0, "trueAnomalyDeg": 0.0}, "velocity": {"velocityX": -7.0, "velocityY": 0.1, "velocityZ": -0.2}, "nodeProcessingDelayMs": 0.15, "packetLossRate": 0.001, "weather": "CLEAR", "resourceUtilization": 0.15, "batteryChargePercent": 85, "currentPacketCount": 40, "packetBufferCapacity": 1000, },
    "NodeD": { "nodeId": "NodeD", "nodeType": "GROUND_STATION", "isOperational": True, "host": "10.0.0.4", "port": 8081, "lastUpdated": 0, "position": {"latitude": 40.0, "longitude": 40.0, "altitude": 0.0}, "velocity": {"velocityX": 0.0, "velocityY": 0.0, "velocityZ": 0.0}, "nodeProcessingDelayMs": 2.0, "packetLossRate": 0.002, "weather": "CLEAR", "resourceUtilization": 0.05, "batteryChargePercent": 99, "currentPacketCount": 20, "packetBufferCapacity": 1000, },
}

BASE_LINK_METRICS: List[Dict[str, Any]] = [
    {"sourceNodeId": "NodeA", "destinationNodeId": "NodeB", "distanceKm": 500.0, "maxBandwidthMbps": 1000.0, "currentAvailableBandwidthMbps": 900.0, "latencyMs": 5.0, "packetLossRate": 0.005, "linkAttenuationDb": 1.0, "isLinkActive": True},
    {"sourceNodeId": "NodeA", "destinationNodeId": "NodeC", "distanceKm": 1500.0, "maxBandwidthMbps": 800.0, "currentAvailableBandwidthMbps": 600.0, "latencyMs": 15.0, "packetLossRate": 0.01, "linkAttenuationDb": 2.5, "isLinkActive": True},
    {"sourceNodeId": "NodeC", "destinationNodeId": "NodeD", "distanceKm": 550.0, "maxBandwidthMbps": 800.0, "currentAvailableBandwidthMbps": 780.0, "latencyMs": 8.0, "packetLossRate": 0.008, "linkAttenuationDb": 1.5, "isLinkActive": True},
    {"sourceNodeId": "NodeB", "destinationNodeId": "NodeD", "distanceKm": 1000.0, "maxBandwidthMbps": 500.0, "currentAvailableBandwidthMbps": 400.0, "latencyMs": 12.0, "packetLossRate": 0.03, "linkAttenuationDb": 3.0, "isLinkActive": True},
    {"sourceNodeId": "NodeB", "destinationNodeId": "NodeA", "distanceKm": 500.0, "maxBandwidthMbps": 1000.0, "currentAvailableBandwidthMbps": 900.0, "latencyMs": 5.0, "packetLossRate": 0.005, "linkAttenuationDb": 1.0, "isLinkActive": True},
    {"sourceNodeId": "NodeD", "destinationNodeId": "NodeC", "distanceKm": 550.0, "maxBandwidthMbps": 800.0, "currentAvailableBandwidthMbps": 780.0, "latencyMs": 8.0, "packetLossRate": 0.008, "linkAttenuationDb": 1.5, "isLinkActive": True},
]

for link in BASE_LINK_METRICS:
    link['linkScore'] = calculate_link_score_simplified(link)

# --- 2. ĐỊNH NGHĨA SỐ LƯỢNG BIẾN THỂ ---
NUM_VARIANTS = 100

def create_static_link_variants(base_links: List[Dict[str, Any]], num_variants: int) -> List[List[Dict[str, Any]]]:
    """Tạo 100 biến thể trạng thái LinkMetrics bằng cách thêm nhiễu ngẫu nhiên và phân bố đều."""
    variants = []
    
    for i in range(num_variants):
        variant = copy.deepcopy(base_links)
        
        # 1. Áp dụng nhiễu trên Link A-B (tuyến thường được chọn)
        for link in variant:
            if link['sourceNodeId'] == 'NodeA' and link['destinationNodeId'] == 'NodeB':
                # Phân bố tải tuyến tính từ 0 đến 100%
                load_factor = i / (num_variants - 1) 
                
                # Biến đổi tuyến tính/ngẫu nhiên:
                # BW giảm dần từ 900 Mbps đến 100 Mbps
                link['currentAvailableBandwidthMbps'] = max(100.0, 900.0 - (800.0 * load_factor))
                # Latency tăng từ 5 ms lên 50 ms
                link['latencyMs'] = 5.0 + 45.0 * load_factor
                # Loss tăng từ 0.5% lên 10%
                link['packetLossRate'] = 0.005 + 0.095 * load_factor
                
                # Biến đổi Node Info theo tải (ví dụ: Node B)
                BASE_NODE_INFO['NodeB']['resourceUtilization'] = 0.1 + 0.8 * load_factor
            
            # 2. Áp dụng nhiễu nhỏ ngẫu nhiên cho tất cả các link khác (để tăng đa dạng)
            else:
                 link['currentAvailableBandwidthMbps'] *= (1.0 + random.uniform(-0.05, 0.05)) # Thay đổi BW ±5%
                 link['latencyMs'] *= (1.0 + random.uniform(-0.02, 0.02)) # Thay đổi Latency ±2%


            link['linkScore'] = calculate_link_score_simplified(link) # Tính lại LinkScore
            
        variants.append(variant)
        
    return variants

# Export các biến thể cuối cùng
STATIC_LINK_VARIANTS = create_static_link_variants(BASE_LINK_METRICS, NUM_VARIANTS)