# scripts/scenario_manager.py
from typing import Dict, Any, List
from utils.static_data import (
    BASE_NODE_INFO, 
    BASE_LINK_METRICS, 
    STATIC_LINK_VARIANTS, 
    calculate_link_score_simplified, 
    NUM_VARIANTS,
    create_static_link_variants
) 
import copy
import random

# --- HÀM ỨNG DỤNG CẬP NHẬT ---

def apply_link_updates(link_key: str, updates: Dict[str, Any]):
    """
    Áp dụng cập nhật cho LinkMetrics CƠ SỞ và TẤT CẢ các biến thể.
    """
    try:
        source_id, dest_id = link_key.split('-')
    except ValueError:
        print(f"LỖI: Link key '{link_key}' không đúng định dạng 'Source-Dest'.")
        return

    # 1. Cập nhật BASE_LINK_METRICS
    for link in BASE_LINK_METRICS:
        if link.get('sourceNodeId') == source_id and link.get('destinationNodeId') == dest_id:
            link.update(updates)
            link['linkScore'] = calculate_link_score_simplified(link)
            break
            
    # 2. Cập nhật TẤT CẢ các biến thể trong STATIC_LINK_VARIANTS
    for variant in STATIC_LINK_VARIANTS:
        for link in variant:
            if link.get('sourceNodeId') == source_id and link.get('destinationNodeId') == dest_id:
                link.update(updates)
                link['linkScore'] = calculate_link_score_simplified(link)
    
    print(f"   [Link Cập nhật] Đã áp dụng {link_key}. Score mới: {link.get('linkScore', 'N/A'):.2f}")


def apply_node_updates(node_id: str, updates: Dict[str, Any]):
    """Áp dụng cập nhật cho NodeInfo CƠ SỞ (BASE_NODE_INFO)."""
    if node_id in BASE_NODE_INFO:
        current_data = BASE_NODE_INFO[node_id]
        current_data.update(updates) 
        
        util = current_data.get('resourceUtilization', 'N/A')
        bat = current_data.get('batteryChargePercent', 'N/A')
        print(f"   [Node Cập nhật] Node {node_id}: Tải={util}, Pin={bat}%")


# --- ĐỊNH NGHĨA VÀ THỰC THI KỊCH BẢN ---

def run_scenario(scenario_type: str):
    """Thực hiện các kịch bản mạng khác nhau bằng cách ghi đè dữ liệu tĩnh."""
    
    if scenario_type != 'BASELINE':
        run_scenario('BASELINE')
        
    print(f"\n--- ÁP DỤNG KỊCH BẢN: {scenario_type} ---")
    
    if scenario_type == 'CONGESTION_LINK_AB':
        # 🚨 TĂNG CƯỜNG ĐỘ LỖI: BW chỉ còn 1 Mbps
        apply_link_updates("NodeA-NodeB", {"latencyMs": 500.0, "currentAvailableBandwidthMbps": 1.0, "packetLossRate": 0.99}) 
        apply_node_updates("NodeB", {"resourceUtilization": 0.95, "currentPacketCount": 900, "nodeProcessingDelayMs": 50.0})
        print("CẢNH BÁO: Kịch bản QoS FAILS (Tắc nghẽn KHẮC NGHIỆT). DRL Agent phải chọn A->C.")

    elif scenario_type == 'SEVERE_WEATHER_AC':
        # Kịch bản 2: Ảnh hưởng thời tiết (suy hao) lên Link A->C
        apply_link_updates("NodeA-NodeC", {"linkAttenuationDb": 35.0, "packetLossRate": 0.90, "currentAvailableBandwidthMbps": 10.0})
        apply_node_updates("NodeA", {"weather": "SEVERE_STORM"})
        print("CẢNH BÁO: Link NodeA->NodeC gần như không khả dụng do thời tiết xấu.")
        
    elif scenario_type == 'NODE_C_OVERLOAD':
        # Kịch bản 3: Node xử lý trung gian (Node C) bị quá tải và sắp hết pin
        apply_node_updates("NodeC", {"resourceUtilization": 0.99, "currentPacketCount": 990, "batteryChargePercent": 15.0, "nodeProcessingDelayMs": 20.0})
        print("CẢNH BÁO: NodeC đang quá tải và pin yếu. Agent phải tránh NodeC.")
        
    elif scenario_type == 'LINK_FAIL_AB':
        # Kịch bản 4: Mất Link Hoàn toàn (Hard Fail)
        apply_link_updates("NodeA-NodeB", {"isLinkActive": False, "currentAvailableBandwidthMbps": 0.0, "packetLossRate": 1.0})
        apply_link_updates("NodeB-NodeA", {"isLinkActive": False, "currentAvailableBandwidthMbps": 0.0, "packetLossRate": 1.0})
        print("CẢNH BÁO: Link NodeA<->NodeB ĐÃ HỎNG. Agent bắt buộc phải chọn A->C.")

    elif scenario_type == 'EXTREME_ATTENUATION_AC':
        # Kịch bản 5: Suy hao cực độ (Môi trường rất xấu tại trạm mặt đất A)
        apply_link_updates("NodeA-NodeC", {"linkAttenuationDb": 35.0, "packetLossRate": 0.95, "currentAvailableBandwidthMbps": 5.0})
        print("CẢNH BÁO: Link A->C gần như không thể dùng được do suy hao.")

    elif scenario_type == 'BASELINE':
        # Kịch bản 6: Khôi phục về trạng thái ban đầu (Trạng thái mạng tối ưu)
        print("Khôi phục trạng thái mạng về Baseline.")
        
        global STATIC_LINK_VARIANTS
        base_links_copy = copy.deepcopy(BASE_LINK_METRICS)
        STATIC_LINK_VARIANTS.clear() 
        STATIC_LINK_VARIANTS.extend(create_static_link_variants(base_links_copy, NUM_VARIANTS))
        
        for node_id in BASE_NODE_INFO.keys():
            BASE_NODE_INFO[node_id]['resourceUtilization'] = 0.15
            BASE_NODE_INFO[node_id]['currentPacketCount'] = 40
            BASE_NODE_INFO[node_id]['batteryChargePercent'] = 90.0
            BASE_NODE_INFO[node_id]['isOperational'] = True
            BASE_NODE_INFO[node_id]['weather'] = 'CLEAR'