
import math
import random
from itertools import combinations
from datetime import datetime
import sys
from typing import List, Dict, Any, Optional

sys.path.insert(0, '.')

try:
    from config.mongo_config import get_collection
    from models.link_metric_model import LinkMetric
except ImportError as e:
    print(f"Lỗi import: {e}. Hãy đảm bảo bạn đang chạy script từ thư mục gốc của dự án.")
    sys.exit(1)

# --- Các hàm tính toán con (giữ nguyên) ---
SPEED_OF_LIGHT_KPS = 299792.458
EARTH_RADIUS_KM = 6371.0

def calculate_haversine_distance(lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float) -> float:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    surface_distance = EARTH_RADIUS_KM * c
    d_alt = abs(alt1 - alt2)
    distance = math.sqrt(surface_distance**2 + d_alt**2)
    return distance

def simulate_link_properties(node1: Dict[str, Any], node2: Dict[str, Any], distance: float) -> tuple:
    latency = (distance / SPEED_OF_LIGHT_KPS) * 1000 + random.uniform(5, 20)
    pair_type = tuple(sorted((node1.get('type', ''), node2.get('type', ''))))
    if "GROUND_STATION" in pair_type: max_bw = random.uniform(500, 2000)
    elif pair_type == ("LEO_SATELLITE", "LEO_SATELLITE"): max_bw = random.uniform(2000, 10000)
    else: max_bw = random.uniform(1000, 3000)
    available_bw = max_bw * random.uniform(0.65, 0.95)
    packet_loss = 0.0005 + (distance / 40000) * 0.01
    attenuation = 20 * math.log10(distance) + random.uniform(10, 25)
    return max_bw, available_bw, latency, packet_loss, attenuation

def calculate_link_score(bw: float, latency: float, loss: float) -> float:
    bw_norm = min(bw / 5000, 1.0)
    lat_norm = 1 - min(latency / 250, 1.0)
    loss_norm = 1 - min(loss / 0.05, 1.0)
    score = (bw_norm * 50) + (lat_norm * 30) + (loss_norm * 20)
    return max(0, min(100, score))

# --- HÀM MỚI: Tính toán cho một cặp Node cụ thể ---
def calculate_single_link_metric(source_node_id: str, dest_node_id: str) -> Optional[Dict[str, Any]]:
    """
    Tính toán chỉ số liên kết cho một cặp node cụ thể và trả về kết quả.
    Không lưu vào database.
    Trả về một dictionary chứa thông tin LinkMetric, hoặc None nếu có lỗi.
    """
    nodes_collection = get_collection("nodes")
    
    # Lấy thông tin của hai node từ DB
    node1 = nodes_collection.find_one({"nodeId": source_node_id})
    node2 = nodes_collection.find_one({"nodeId": dest_node_id})

    if not node1 or not node2:
        print(f"Lỗi: Không tìm thấy một trong hai node: {source_node_id}, {dest_node_id}")
        return None

    # Kiểm tra xem cả hai node có hoạt động không
    if not node1.get("status", {}).get("active") or not node2.get("status", {}).get("active"):
        return {
            "sourceNodeId": source_node_id,
            "destinationNodeId": dest_node_id,
            "isLinkActive": False,
            "linkScore": 0,
            "reason": "Một hoặc cả hai node không hoạt động"
        }

    pos1 = node1.get('position', {})
    pos2 = node2.get('position', {})
    
    distance_km = calculate_haversine_distance(
        pos1.get('latitude', 0), pos1.get('longitude', 0), pos1.get('altitude', 0),
        pos2.get('latitude', 0), pos2.get('longitude', 0), pos2.get('altitude', 0)
    )
    
    max_bw, avail_bw, lat, loss, atten = simulate_link_properties(node1, node2, distance_km)
    score = calculate_link_score(avail_bw, lat, loss)
    
    metric = LinkMetric(
        sourceNodeId=node1['nodeId'],
        destinationNodeId=node2['nodeId'],
        distanceKm=round(distance_km, 2),
        maxBandwidthMbps=round(max_bw, 2),
        currentAvailableBandwidthMbps=round(avail_bw, 2),
        latencyMs=round(lat, 2),
        packetLossRate=round(loss, 5),
        linkAttenuationDb=round(atten, 2),
        linkScore=round(score, 2)
    )
    # Trả về dưới dạng dictionary để trang view có thể sử dụng
    return metric.to_dict()

# --- HÀM CŨ: Đổi tên để rõ mục đích (vẫn hữu ích để chạy thủ công) ---
def update_all_link_metrics_in_db():
    """
    Tạo chỉ số cho tất cả các cặp node hoạt động và lưu vào collection 'link_metrics'.
    """
    # ... (Nội dung của hàm update_all_link_metrics cũ giữ nguyên ở đây) ...
    nodes_collection = get_collection("nodes")
    links_collection = get_collection("link_metrics")
    try:
        active_nodes = list(nodes_collection.find({"status.active": True}))
    except Exception as e:
        print(f"Lỗi khi truy vấn database: {e}")
        return
    if len(active_nodes) < 2:
        print("Không đủ node hoạt động để tạo liên kết.")
        return
    print(f"Tìm thấy {len(active_nodes)} node đang hoạt động. Bắt đầu tạo chỉ số liên kết...")
    all_link_metrics: List[Dict] = []
    for node1, node2 in combinations(active_nodes, 2):
        # ... (Toàn bộ logic tính toán và tạo metric object)
        pos1, pos2 = node1.get('position', {}), node2.get('position', {})
        distance_km = calculate_haversine_distance(pos1.get('latitude',0), pos1.get('longitude',0), pos1.get('altitude',0), pos2.get('latitude',0), pos2.get('longitude',0), pos2.get('altitude',0))
        max_bw, avail_bw, lat, loss, atten = simulate_link_properties(node1, node2, distance_km)
        score = calculate_link_score(avail_bw, lat, loss)
        metric = LinkMetric(sourceNodeId=node1['nodeId'], destinationNodeId=node2['nodeId'], distanceKm=round(distance_km, 2), maxBandwidthMbps=round(max_bw, 2), currentAvailableBandwidthMbps=round(avail_bw, 2), latencyMs=round(lat, 2), packetLossRate=round(loss, 5), linkAttenuationDb=round(atten, 2), linkScore=round(score, 2))
        all_link_metrics.append(metric.to_dict())
    if all_link_metrics:
        print(f"Đã tạo {len(all_link_metrics)} chỉ số liên kết. Đang cập nhật database...")
        try:
            links_collection.delete_many({})
            links_collection.insert_many(all_link_metrics)
            print("Cập nhật collection 'link_metrics' thành công.")
        except Exception as e:
            print(f"Lỗi khi cập nhật database: {e}")

if __name__ == "__main__":
    print("--- Chạy cập nhật thủ công TOÀN BỘ Link Metrics vào DB ---")
    update_all_link_metrics_in_db()
    print("--- Hoàn thành ---")