"""
Routing API Blueprint
Provides endpoints for packet routing and path calculation
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Optional, Dict, Any
from models.database import db
import logging
import math
from bson import ObjectId
import json
from environment.constants import (
    EARTH_RADIUS_M,
    SPEED_OF_LIGHT_MPS,
    MS_PER_SECOND,
    UTILIZATION_MAX_PERCENT,
    UTILIZATION_HIGH_PERCENT,
    UTILIZATION_MEDIUM_PERCENT,
    UTILIZATION_LOW_PERCENT,
    UTILIZATION_CRITICAL_PERCENT,
    TERMINAL_UTILIZATION_IMPACT,
    BATTERY_MAX_PERCENT,
    BATTERY_LOW_PERCENT,
    PACKET_LOSS_HIGH,
    TRAP_PACKET_LOSS_HIGH,
    NORM_PACKET_BUFFER,
    GS_CONNECTION_OVERLOADED,
    GS_DIRECT_CONNECTION_THRESHOLD_KM,
    TERMINAL_TO_GS_MAX_RANGE_KM,
    DEFAULT_MAX_RANGE_KM,
    GS_TO_LEO_MAX_RANGE_KM,
    GS_TO_MEO_MAX_RANGE_KM,
    GS_TO_GEO_MAX_RANGE_KM,
    LEO_MAX_RANGE_KM,
    LEO_TO_MEO_MAX_RANGE_KM,
    LEO_TO_GEO_MAX_RANGE_KM,
    MEO_MAX_RANGE_KM,
    MEO_TO_GEO_MAX_RANGE_KM,
    GEO_MAX_RANGE_KM,
    RESOURCE_FACTOR_LOW_THRESHOLD,
    RESOURCE_FACTOR_MEDIUM_THRESHOLD,
    RESOURCE_FACTOR_HIGH_THRESHOLD,
    RESOURCE_FACTOR_MAX_PERCENT,
    RESOURCE_FACTOR_LOW_BONUS,
    RESOURCE_FACTOR_MEDIUM_PENALTY_MAX,
    RESOURCE_FACTOR_MEDIUM_PENALTY_RANGE,
    RESOURCE_FACTOR_HIGH_PENALTY_MAX,
    RESOURCE_FACTOR_HIGH_PENALTY_RANGE,
    SATELLITE_RANGE_MARGIN,
    DIJKSTRA_DROP_THRESHOLD,
    TRAP_BATTERY_MODERATE,
    M_TO_KM
)

logger = logging.getLogger(__name__)

def emit_node_update_via_websocket(node_data: dict):
    """
    Emit node update via WebSocket để frontend có thể nhận real-time updates.
    Sử dụng Flask-SocketIO nếu available, hoặc HTTP request tới WebSocket gateway.
    """
    try:
        # Try Flask-SocketIO first
        try:
            from flask import current_app
            socketio = current_app.extensions.get('socketio')
            if socketio:
                # Emit to all clients via SocketIO on /topic/node-status
                socketio.emit('node-status', node_data, namespace='/', broadcast=True)
                logger.info(f"📡 Emitted node update via SocketIO: {node_data.get('nodeId')} - Util: {node_data.get('resourceUtilization')}%")
                return
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"SocketIO not available: {str(e)}")
            pass
        
        # Try STOMP/Spring WebSocket gateway via HTTP (if exists)
        try:
            import requests
            import os
            ws_gateway_url = os.getenv('WS_GATEWAY_URL', 'http://localhost:8080/api/websocket/broadcast')
            response = requests.post(
                f"{ws_gateway_url}/node-status",
                json=node_data,
                timeout=1
            )
            if response.status_code == 200:
                logger.debug(f"📡 Emitted node update via HTTP gateway: {node_data.get('nodeId')}")
                return
        except (ImportError, Exception) as e:
            logger.debug(f"HTTP gateway not available: {str(e)}")
        
        # Fallback: Log và để frontend auto-refresh nhận update
        logger.debug(f"No WebSocket available, node update logged: {node_data.get('nodeId')}")
        
    except Exception as e:
        logger.warning(f"Error emitting node update via WebSocket: {str(e)}")
        # Frontend đã có auto-refresh mỗi 5s và WebSocket subscription
        logger.debug(f"📡 Node update ready (frontend will receive via auto-refresh/WebSocket): {node_data.get('nodeId')}")
        
    except Exception as e:
        logger.warning(f"Could not emit WebSocket message for node {node_data.get('nodeId', 'unknown')}: {str(e)}")


def clean_for_json(obj: Any) -> Any:
    """
    Recursively clean object for JSON serialization.
    Converts ObjectId to string and handles other non-serializable types.
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items() if key != '_id'}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    else:
        return obj

routing_bp = Blueprint('routing', __name__, url_prefix='/api/v1/routing')

def calculate_distance(pos1: dict, pos2: dict) -> float:
    """Calculate distance between two positions in meters"""
    # Haversine formula for great circle distance
    lat1 = math.radians(pos1['latitude'])
    lon1 = math.radians(pos1['longitude'])
    lat2 = math.radians(pos2['latitude'])
    lon2 = math.radians(pos2['longitude'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    R = EARTH_RADIUS_M
    
    # Calculate 3D distance including altitude
    horizontal_dist = R * c
    vertical_dist = abs((pos1.get('altitude', 0) - pos2.get('altitude', 0)))
    
    return math.sqrt(horizontal_dist**2 + vertical_dist**2)

def find_nearest_node(terminal: dict, nodes: list) -> dict:
    """Find the nearest node to a terminal (deprecated - use find_best_ground_station)"""
    if not nodes:
        return None
    
    terminal_pos = terminal['position']
    min_distance = float('inf')
    nearest_node = None
    
    for node in nodes:
        if not node.get('position'):
            continue
        
        distance = calculate_distance(terminal_pos, node['position'])
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    
    return nearest_node

def find_nearest_ground_station(terminal: dict, nodes: list) -> dict:
    """
    Tìm Ground Station GẦN NHẤT cho terminal - chỉ xét khoảng cách.
    
    Dùng cho Dijkstra routing để làm baseline "ngu" - không tối ưu tài nguyên.
    RL routing sẽ dùng find_best_ground_station() để tối ưu resource.
    
    Args:
        terminal: Terminal dictionary with position
        nodes: List of all nodes
    
    Returns:
        Nearest Ground Station or None if not found
    """
    if not nodes:
        return None
    
    terminal_pos = terminal['position']
    ground_stations = [
        n for n in nodes
        if n.get('nodeType') == 'GROUND_STATION' 
        and n.get('isOperational', True)
        and n.get('position')
    ]
    
    if not ground_stations:
        logger.error(f"❌ No operational Ground Stations found for terminal {terminal.get('terminalId')}")
        return None
    
    # Chỉ xét khoảng cách - KHÔNG quan tâm utilization, battery, load...
    nearest_gs = None
    min_distance = float('inf')
    
    for gs in ground_stations:
        distance = calculate_distance(terminal_pos, gs['position'])
        if distance < min_distance:
            min_distance = distance
            nearest_gs = gs
    
    if nearest_gs:
        logger.debug(
            f"Dijkstra: Terminal {terminal.get('terminalId')} → GS {nearest_gs['nodeId']} "
            f"at {min_distance/1000:.1f}km (nearest only, no resource optimization)"
        )
    
    return nearest_gs

def get_node_connection_count(node_id: str) -> int:
    """Get the number of terminals currently connected to a node"""
    try:
        terminals_collection = db.get_collection('terminals')
        count = terminals_collection.count_documents({
            'connectedNodeId': node_id,
            'status': {'$in': ['connected', 'transmitting']}
        })
        return count
    except Exception as e:
        logger.warning(f"Error counting connections for node {node_id}: {str(e)}")
        return 0

def update_node_resource_utilization(node_id: str):
    """
    Cập nhật resource utilization, packet loss, queue, và battery của node 
    dựa trên số lượng terminals đang kết nối để mô phỏng thực tế hơn.
    
    Mỗi terminal kết nối sẽ:
    - Tăng utilization (10% per terminal) - mô phỏng tác động lớn hơn
    - Tăng packet loss rate (0.002 per terminal, base 0.001)
    - Tăng queue packets (20 packets per terminal) - tăng gấp đôi
    - Giảm battery (1% per terminal per update, minimum 10%) - tiêu thụ nhiều hơn
    """
    try:
        nodes_collection = db.get_collection('nodes')
        terminals_collection = db.get_collection('terminals')
        
        # Đếm số terminals connected
        connection_count = terminals_collection.count_documents({
            'connectedNodeId': node_id,
            'status': {'$in': ['connected', 'transmitting']}
        })
        
        # Lấy node hiện tại
        node = nodes_collection.find_one({'nodeId': node_id})
        if not node:
            logger.warning(f"Node {node_id} not found for utilization update.")
            return
        
        # ========== 1. UTILIZATION (%) ==========
        # Base utilization từ node (có thể có từ trước)
        # Nhưng để tính toán chính xác, ta sẽ tính lại dựa trên số terminals
        # Giả sử base utilization là 0 khi không có terminals
        base_utilization = 0  # Base utilization khi không có terminals
        terminal_utilization_impact = connection_count * TERMINAL_UTILIZATION_IMPACT
        new_utilization = min(UTILIZATION_MAX_PERCENT, base_utilization + terminal_utilization_impact)
        
        # ========== 2. PACKET LOSS RATE ==========
        # Reset base packet loss về 0.001 (0.1%) nếu không có terminals
        base_packet_loss = 0.001  # Base: 0.1%
        # Mỗi terminal tăng packet loss (do congestion) - tác động lớn hơn
        # Formula: base + (terminals * 0.002) + (terminals^2 * 0.0002) để mô phỏng exponential growth
        terminal_packet_loss_impact = connection_count * 0.002 + (connection_count ** 2) * 0.0002
        new_packet_loss = min(0.15, base_packet_loss + terminal_packet_loss_impact)  # Cap at 15%
        
        # ========== 3. QUEUE (currentPacketCount) ==========
        # Queue chỉ tính từ terminals (không giữ packets cũ)
        base_packet_count = 0  # Base queue khi không có terminals
        # Mỗi terminal tạo ~20 packets trong queue (tăng gấp đôi để mô phỏng tác động lớn hơn)
        packets_per_terminal = 20
        new_packet_count = base_packet_count + (connection_count * packets_per_terminal)
        buffer_capacity = node.get('packetBufferCapacity', NORM_PACKET_BUFFER)
        new_packet_count = min(buffer_capacity, new_packet_count)
        
        current_battery = node.get('batteryChargePercent', BATTERY_MAX_PERCENT)
        battery_drain_per_terminal = 1.0
        battery_drain = connection_count * battery_drain_per_terminal
        new_battery = max(BATTERY_LOW_PERCENT, current_battery - battery_drain)
        
        # ========== Cập nhật node ==========
        nodes_collection.update_one(
            {'nodeId': node_id},
            {
                '$set': {
                    'resourceUtilization': round(new_utilization, 2),
                    'packetLossRate': round(new_packet_loss, 4),
                    'currentPacketCount': int(new_packet_count),
                    'batteryChargePercent': round(new_battery, 2),
                    'lastUpdated': datetime.now().isoformat()
                }
            }
        )
        
        # Lấy node đã cập nhật để gửi qua WebSocket
        updated_node = nodes_collection.find_one({'nodeId': node_id}, {'_id': 0})
        
        # Emit WebSocket update để frontend nhận real-time
        if updated_node:
            emit_node_update_via_websocket(updated_node)
        
        logger.info(
            f"📊 Updated {node_id} resources ({connection_count} terminals): "
            f"Utilization: {base_utilization:.1f}% → {new_utilization:.1f}% | "
            f"Packet Loss: {base_packet_loss:.4f} → {new_packet_loss:.4f} | "
            f"Queue: {base_packet_count} → {new_packet_count} | "
            f"Battery: {current_battery:.1f}% → {new_battery:.1f}%"
        )
        
    except Exception as e:
        logger.warning(f"Error updating resource utilization for node {node_id}: {str(e)}")

def find_best_ground_station(terminal: dict, nodes: list, 
                             distance_weight: float = 0.25,
                             utilization_weight: float = 0.25,
                             connection_weight: float = 0.15,
                             battery_weight: float = 0.15,
                             packet_loss_weight: float = 0.20) -> dict:
    """
    Tìm Ground Station tối ưu nhất cho terminal dựa trên tối ưu tài nguyên.
    
    Optimization factors (tối ưu tài nguyên):
    - Distance: Gần hơn = tốt hơn (latency thấp) - 25%
    - Resource Utilization: Thấp hơn = tốt hơn (nhiều capacity) - 25%
    - Connection Count: Ít hơn = tốt hơn (load balancing) - 15%
    - Battery Level: Cao hơn = tốt hơn (năng lượng) - 15%
    - Packet Loss Rate: Thấp hơn = tốt hơn (chất lượng) - 20%
    
    Args:
        terminal: Terminal dictionary with position
        nodes: List of all nodes
        distance_weight: Weight for distance factor (default: 0.25)
        utilization_weight: Weight for resource utilization (default: 0.25)
        connection_weight: Weight for connection count (default: 0.15)
        battery_weight: Weight for battery level (default: 0.15)
        packet_loss_weight: Weight for packet loss rate (default: 0.20)
    
    Returns:
        Best Ground Station node or None if no suitable node found
    """
    if not nodes:
        return None
    
    # Terminal tìm Ground Station trong phạm vi mở rộng (để hỗ trợ kết nối xa)
    # Mở rộng range để terminal ở Đà Nẵng có thể chọn Hà Nội (~750km) hoặc Hồ Chí Minh (~900km)
    # Range mở rộng: 1500km (đủ để bao phủ toàn bộ Việt Nam)
    ground_stations = []
    terminal_pos = terminal['position']
    
    terminal_max_range_km = terminal.get('communication', {}).get('maxRangeKm', TERMINAL_TO_GS_MAX_RANGE_KM)
    terminal_max_range_km = min(terminal_max_range_km, TERMINAL_TO_GS_MAX_RANGE_KM)
    
    for node in nodes:
        if (node.get('nodeType') == 'GROUND_STATION' 
            and node.get('isOperational', True)
            and node.get('position')):
            
            # CHỈ tìm trong phạm vi THỰC TẾ của terminal
            distance_km = calculate_distance(terminal_pos, node['position']) / 1000.0
            
            # Terminal chỉ kết nối được trong phạm vi nhỏ (50-100km)
            # Không dùng node's maxRangeKm (có thể là 2000km) vì đó là range của node với satellites
            if distance_km <= terminal_max_range_km:
                ground_stations.append(node)
                logger.debug(f"Terminal {terminal.get('terminalId')} found GS {node['nodeId']} at {distance_km:.1f}km (within {terminal_max_range_km}km range)")
    
    if not ground_stations:
        # Fallback: Tìm GS gần nhất (không giới hạn range) để đảm bảo routing vẫn hoạt động
        logger.warning(
            f"⚠️ No Ground Stations within {terminal_max_range_km}km range for terminal {terminal.get('terminalId')} "
            f"at ({terminal_pos.get('latitude', 0):.2f}, {terminal_pos.get('longitude', 0):.2f}), "
            f"searching for nearest GS..."
        )
        
        # Tìm tất cả GS và chọn gần nhất
        all_gs = [
            n for n in nodes
            if n.get('nodeType') == 'GROUND_STATION' 
            and n.get('isOperational', True)
            and n.get('position')
        ]
        
        if not all_gs:
            logger.error(f"❌ No operational Ground Stations available at all!")
            return None
        
        # Tìm GS gần nhất
        nearest_gs = None
        min_distance = float('inf')
        for gs in all_gs:
            distance_km = calculate_distance(terminal_pos, gs['position']) / 1000.0
            if distance_km < min_distance:
                min_distance = distance_km
                nearest_gs = gs
        
        if nearest_gs:
            logger.info(
                f"✅ Found nearest Ground Station {nearest_gs['nodeId']} at {min_distance:.1f}km "
                f"(beyond {terminal_max_range_km}km range, but using as fallback)"
            )
            # Trả về GS gần nhất (dù xa hơn range)
            return nearest_gs
        
        return None
    
    best_node = None
    best_score = -1
    
    max_distance = 0
    max_utilization = UTILIZATION_MAX_PERCENT
    max_connections = 0
    min_battery = BATTERY_MAX_PERCENT
    max_packet_loss = 0.0
    
    # First pass: calculate normalization factors
    for node in ground_stations:
        distance = calculate_distance(terminal_pos, node['position'])
        max_distance = max(max_distance, distance)
        
        connection_count = get_node_connection_count(node['nodeId'])
        max_connections = max(max_connections, connection_count)
        
        battery = node.get('batteryChargePercent', BATTERY_MAX_PERCENT)
        min_battery = min(min_battery, battery)
        
        packet_loss = node.get('packetLossRate', 0)
        max_packet_loss = max(max_packet_loss, packet_loss)
    
    # If all distances are 0, set a default
    if max_distance == 0:
        max_distance = 1
    if max_connections == 0:
        max_connections = 1
    if max_packet_loss == 0:
        max_packet_loss = TRAP_PACKET_LOSS_HIGH
    
    # Second pass: calculate scores and find best node (tối ưu tài nguyên)
    for node in ground_stations:
        distance = calculate_distance(terminal_pos, node['position'])
        utilization = node.get('resourceUtilization', 0)
        connection_count = get_node_connection_count(node['nodeId'])
        battery = node.get('batteryChargePercent', BATTERY_MAX_PERCENT)
        packet_loss = node.get('packetLossRate', 0)
        packet_count = node.get('currentPacketCount', 0)
        packet_capacity = node.get('packetBufferCapacity', NORM_PACKET_BUFFER)
        
        normalized_distance = 1.0 - (distance / max_distance) if max_distance > 0 else 0.5
        normalized_utilization = 1.0 - (utilization / max_utilization)
        normalized_connections = 1.0 - (connection_count / max(max_connections, 1))
        normalized_battery = battery / BATTERY_MAX_PERCENT
        normalized_packet_loss = 1.0 - (packet_loss / max_packet_loss) if max_packet_loss > 0 else 1.0
        buffer_factor = 1.0 - min(packet_count / max(packet_capacity, 1), 1.0)
        
        # Combined score với weights tối ưu tài nguyên
        # Higher score = better choice
        score = (
            normalized_distance * distance_weight +
            normalized_utilization * utilization_weight +
            normalized_connections * connection_weight +
            normalized_battery * battery_weight +
            normalized_packet_loss * packet_loss_weight +
            buffer_factor * 0.05  # 5% weight for buffer capacity
        )
        
        penalty = 0.0
        if utilization > UTILIZATION_CRITICAL_PERCENT:
            penalty += 0.3
        if battery < BATTERY_LOW_PERCENT:
            penalty += 0.3
        if packet_loss > PACKET_LOSS_HIGH:
            penalty += 0.2
        if connection_count > GS_CONNECTION_OVERLOADED:
            penalty += 0.15
        
        score = max(0, score - penalty)
        
        if utilization < UTILIZATION_LOW_PERCENT and battery > UTILIZATION_HIGH_PERCENT and packet_loss < TRAP_PACKET_LOSS_HIGH:
            score += 0.1
        
        if score > best_score:
            best_score = score
            best_node = node
    
    if best_node:
        distance_km = calculate_distance(terminal_pos, best_node['position']) / 1000.0
        logger.info(
            f"✅ Selected optimal Ground Station {best_node['nodeId']} for terminal {terminal.get('terminalId')} "
            f"(score: {best_score:.3f}, distance: {distance_km:.1f}km, "
            f"utilization: {best_node.get('resourceUtilization', 0):.1f}%, "
            f"battery: {best_node.get('batteryChargePercent', BATTERY_MAX_PERCENT):.1f}%, "
            f"packet_loss: {best_node.get('packetLossRate', 0)*100:.2f}%, "
            f"connections: {get_node_connection_count(best_node['nodeId'])})"
        )
    
    return best_node

def calculate_path(source_terminal: dict, dest_terminal: dict, nodes: list) -> dict:
    """
    Calculate path from source terminal to destination terminal
    Returns path with intermediate nodes
    """
    path = {
        'source': {
            'terminalId': source_terminal['terminalId'],
            'position': source_terminal['position']
        },
        'destination': {
            'terminalId': dest_terminal['terminalId'],
            'position': dest_terminal['position']
        },
        'path': [],
        'totalDistance': 0,
        'estimatedLatency': 0,
        'hops': 0
    }
    
    # Find best Ground Stations for source and destination (load-balanced)
    source_node = find_best_ground_station(source_terminal, nodes)
    dest_node = find_best_ground_station(dest_terminal, nodes)
    
    if not source_node or not dest_node:
        logger.warning(f"⚠️ Cannot find ground stations: source_node={source_node is not None}, dest_node={dest_node is not None}")
        path['success'] = False
        return path
    
    # Add source terminal
    path['path'].append({
        'type': 'terminal',
        'id': source_terminal['terminalId'],
        'name': source_terminal.get('terminalName', source_terminal['terminalId']),
        'position': source_terminal['position']
    })
    
    # Add source node
    path['path'].append({
        'type': 'node',
        'id': source_node['nodeId'],
        'name': source_node.get('nodeName', source_node['nodeId']),
        'position': source_node['position']
    })
    
    # Simple routing: if source and dest nodes are different, add intermediate nodes
    if source_node['nodeId'] != dest_node['nodeId']:
        # Check if ground stations can communicate directly (strict range check)
        source_pos = source_node['position']
        dest_pos = dest_node['position']
        distance = calculate_distance(source_pos, dest_pos)
        
        source_max_range = source_node.get('communication', {}).get('maxRangeKm', DEFAULT_MAX_RANGE_KM) * M_TO_KM
        dest_max_range = dest_node.get('communication', {}).get('maxRangeKm', DEFAULT_MAX_RANGE_KM) * M_TO_KM
        max_range = min(source_max_range, dest_max_range)  # Use minimum range
        
        # BẮT BUỘC: Ground stations quá xa (>80% range) PHẢI đi qua satellites
        # Chỉ cho phép direct connection nếu rất gần (<50% range)
        distance_ratio = distance / max_range if max_range > 0 else 1.0
        
        # BẮT BUỘC: Luôn đi qua satellites, không cho phép direct ground-to-ground connection
        # (trừ khi cùng một ground station)
        logger.info(f"🛰️ Routing through satellites (distance: {distance/1000:.1f}km, max_range: {max_range/1000:.1f}km)")
        
        # Find best satellite path between source and dest ground stations
        # Priority: LEO > MEO > GEO (for lower latency)
        satellites = [
            n for n in nodes 
            if n.get('nodeType') in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']
            and n.get('isOperational', True)
            and n.get('position')
        ]
        
        if not satellites:
            logger.warning("⚠️ No satellites available, falling back to direct connection")
        else:
            # Find satellite that can connect to both ground stations
            best_satellite = None
            best_score = float('inf')
            
            for sat in satellites:
                dist_to_source = calculate_distance(source_node['position'], sat['position'])
                dist_to_dest = calculate_distance(sat['position'], dest_node['position'])
                
                sat_max_range = sat.get('communication', {}).get('maxRangeKm', DEFAULT_MAX_RANGE_KM) * M_TO_KM
                
                # Check if satellite can reach both ground stations
                if dist_to_source <= sat_max_range and dist_to_dest <= sat_max_range:
                    # Score: prefer LEO (lower altitude = lower latency), then by total distance
                    priority = 0
                    if sat.get('nodeType') == 'LEO_SATELLITE':
                        priority = 0  # Best
                    elif sat.get('nodeType') == 'MEO_SATELLITE':
                        priority = 1000000  # Medium
                    else:  # GEO
                        priority = 2000000  # Worst
                    
                    total_dist = dist_to_source + dist_to_dest
                    score = priority + total_dist
                    
                    if score < best_score:
                        best_score = score
                        best_satellite = sat
            
            if best_satellite:
                path['path'].append({
                    'type': 'node',
                    'id': best_satellite['nodeId'],
                    'name': best_satellite.get('nodeName', best_satellite['nodeId']),
                    'position': best_satellite['position']
                })
                logger.info(f"✅ Using {best_satellite.get('nodeType')} {best_satellite['nodeId']} as intermediate (altitude: {best_satellite.get('position', {}).get('altitude', 0)/1000:.0f}km)")
            else:
                # If no single satellite can reach both, try multi-hop through satellites
                logger.warning("No single satellite can reach both ground stations, attempting multi-hop")
                # Try to find a path through multiple satellites (simplified: find closest to source, then to dest)
                closest_to_source = None
                closest_to_dest = None
                min_dist_source = float('inf')
                min_dist_dest = float('inf')
                
                for sat in satellites:
                    sat_max_range = sat.get('communication', {}).get('maxRangeKm', DEFAULT_MAX_RANGE_KM) * M_TO_KM
                    
                    dist_to_source = calculate_distance(source_node['position'], sat['position'])
                    if dist_to_source <= sat_max_range and dist_to_source < min_dist_source:
                        min_dist_source = dist_to_source
                        closest_to_source = sat
                    
                    dist_to_dest = calculate_distance(sat['position'], dest_node['position'])
                    if dist_to_dest <= sat_max_range and dist_to_dest < min_dist_dest:
                        min_dist_dest = dist_to_dest
                        closest_to_dest = sat
                
                # If we found satellites that can connect, add them
                if closest_to_source:
                    path['path'].append({
                        'type': 'node',
                        'id': closest_to_source['nodeId'],
                        'name': closest_to_source.get('nodeName', closest_to_source['nodeId']),
                        'position': closest_to_source['position']
                    })
                    logger.info(f"✅ Added {closest_to_source.get('nodeType')} {closest_to_source['nodeId']} near source (altitude: {closest_to_source.get('position', {}).get('altitude', 0)/1000:.0f}km)")
                    if closest_to_dest and closest_to_dest['nodeId'] != closest_to_source['nodeId']:
                        path['path'].append({
                            'type': 'node',
                            'id': closest_to_dest['nodeId'],
                            'name': closest_to_dest.get('nodeName', closest_to_dest['nodeId']),
                            'position': closest_to_dest['position']
                        })
                        logger.info(f"✅ Added {closest_to_dest.get('nodeType')} {closest_to_dest['nodeId']} near destination (altitude: {closest_to_dest.get('position', {}).get('altitude', 0)/1000:.0f}km)")
    
    # Add destination node (only if not already added)
    if not path['path'] or path['path'][-1]['id'] != dest_node['nodeId']:
        path['path'].append({
            'type': 'node',
            'id': dest_node['nodeId'],
            'name': dest_node.get('nodeName', dest_node['nodeId']),
            'position': dest_node['position']
        })
    
    # Add destination terminal (always add at the end)
    path['path'].append({
        'type': 'terminal',
        'id': dest_terminal['terminalId'],
        'name': dest_terminal.get('terminalName', dest_terminal['terminalId']),
        'position': dest_terminal['position']
    })
    
    # Validate path has at least source terminal, source node, dest node, dest terminal
    if len(path['path']) < 4:
        logger.warning(f"⚠️ Path has only {len(path['path'])} segments, expected at least 4 (source_terminal, source_node, dest_node, dest_terminal)")
    
    # Calculate total distance and estimated latency
    total_distance = 0
    for i in range(len(path['path']) - 1):
        pos1 = path['path'][i]['position']
        pos2 = path['path'][i + 1]['position']
        segment_dist = calculate_distance(pos1, pos2)
        total_distance += segment_dist
        
        speed_of_light = SPEED_OF_LIGHT_MPS
        propagation_delay = (segment_dist / speed_of_light) * MS_PER_SECOND
        processing_delay = 5
        path['estimatedLatency'] += propagation_delay + processing_delay
    
    path['totalDistance'] = round(total_distance / M_TO_KM, 2)
    path['estimatedLatency'] = round(path['estimatedLatency'], 2)
    path['hops'] = len(path['path']) - 1
    path['success'] = len(path['path']) >= 4  # At least: source_terminal, source_node, dest_node, dest_terminal
    
    # Debug: Log final path structure
    logger.info(f"📊 Final path structure: {path['hops']} hops, {len(path['path'])} segments, success={path['success']}")
    for i, seg in enumerate(path['path']):
        seg_alt = seg['position'].get('altitude', 0) / 1000.0
        logger.debug(f"  Segment {i}: {seg['type']} {seg['id']} at altitude {seg_alt:.0f}km")
    
    if not path['success']:
        logger.warning(f"⚠️ Path validation failed: only {len(path['path'])} segments, expected at least 4")
    
    return path

def calculate_edge_latency(node1: dict, node2: dict, distance_m: float) -> float:
    """
    Tính latency chính xác cho edge giữa 2 nodes (ms)
    QUAN TRỌNG: Resource quality TÁC ĐỘNG đến processing delay!
    - High utilization → processing chậm hơn
    - High packet loss → retransmission overhead
    - Low battery → power-saving mode, processing chậm
    """
    speed_of_light = SPEED_OF_LIGHT_MPS
    propagation_delay_ms = (distance_m / speed_of_light) * MS_PER_SECOND
    
    # Processing delay khác nhau cho từng loại node
    node1_type = node1.get('nodeType', '')
    node2_type = node2.get('nodeType', '')
    
    # Base processing delay cho từng loại node (ms)
    base_processing_delays = {
        'GROUND_STATION': 2.0,      # Ground station: processing nhanh nhất
        'LEO_SATELLITE': 5.0,      # LEO: processing nhanh
        'MEO_SATELLITE': 8.0,       # MEO: processing trung bình
        'GEO_SATELLITE': 10.0      # GEO: processing chậm hơn
    }
    
    # Lấy base processing delay từ node config hoặc dùng default
    node1_base_delay = node1.get('nodeProcessingDelayMs', 
                                base_processing_delays.get(node1_type, 5.0))
    node2_base_delay = node2.get('nodeProcessingDelayMs', 
                                base_processing_delays.get(node2_type, 5.0))
    
    node1_util = node1.get('resourceUtilization', 50) / UTILIZATION_MAX_PERCENT
    node2_util = node2.get('resourceUtilization', 50) / UTILIZATION_MAX_PERCENT
    
    util_threshold = UTILIZATION_MEDIUM_PERCENT / UTILIZATION_MAX_PERCENT
    node1_util_penalty = 1.0 + (node1_util - util_threshold) * 2.0 if node1_util > util_threshold else 1.0
    node2_util_penalty = 1.0 + (node2_util - util_threshold) * 2.0 if node2_util > util_threshold else 1.0
    
    node1_processing = node1_base_delay * node1_util_penalty
    node2_processing = node2_base_delay * node2_util_penalty
    
    processing_delay_ms = (node1_processing + node2_processing) / 2.0
    
    # 🎯 PACKET LOSS IMPACT: Loss rate cao → retransmission overhead
    node1_loss = node1.get('packetLossRate', 0.001)  # 0-1
    node2_loss = node2.get('packetLossRate', 0.001)
    
    node1_loss_penalty = node1_loss * 1000 if node1_loss > TRAP_PACKET_LOSS_HIGH else 0
    node2_loss_penalty = node2_loss * 1000 if node2_loss > TRAP_PACKET_LOSS_HIGH else 0
    loss_delay_ms = (node1_loss_penalty + node2_loss_penalty) / 2.0
    
    node1_battery = node1.get('batteryChargePercent', BATTERY_MAX_PERCENT) / BATTERY_MAX_PERCENT
    node2_battery = node2.get('batteryChargePercent', BATTERY_MAX_PERCENT) / BATTERY_MAX_PERCENT
    
    battery_threshold = TRAP_BATTERY_MODERATE / BATTERY_MAX_PERCENT
    node1_battery_penalty = (battery_threshold - node1_battery) * 20 if node1_battery < battery_threshold else 0
    node2_battery_penalty = (battery_threshold - node2_battery) * 20 if node2_battery < battery_threshold else 0
    battery_delay_ms = (node1_battery_penalty + node2_battery_penalty) / 2.0
    
    node1_queue = node1.get('currentPacketCount', 0) / max(node1.get('packetBufferCapacity', NORM_PACKET_BUFFER), 1) * 10.0
    node2_queue = node2.get('currentPacketCount', 0) / max(node2.get('packetBufferCapacity', NORM_PACKET_BUFFER), 1) * 10.0
    queue_delay_ms = (node1_queue + node2_queue) / 2.0
    
    # Total latency = propagation + processing + resource penalties + queue
    total_latency = (propagation_delay_ms + processing_delay_ms + 
                    loss_delay_ms + battery_delay_ms + queue_delay_ms)
    
    return total_latency

COMM_RANGES = {
    'GS_TO_LEO': GS_TO_LEO_MAX_RANGE_KM,
    'GS_TO_MEO': GS_TO_MEO_MAX_RANGE_KM,
    'GS_TO_GEO': GS_TO_GEO_MAX_RANGE_KM,
    'LEO_TO_LEO': LEO_MAX_RANGE_KM,
    'LEO_TO_MEO': LEO_TO_MEO_MAX_RANGE_KM,
    'LEO_TO_GEO': LEO_TO_GEO_MAX_RANGE_KM,
    'MEO_TO_MEO': MEO_MAX_RANGE_KM,
    'MEO_TO_GEO': MEO_TO_GEO_MAX_RANGE_KM,
    'GEO_TO_GEO': GEO_MAX_RANGE_KM,
    'TERMINAL_TO_GS': TERMINAL_TO_GS_MAX_RANGE_KM
}

def get_max_comm_range(node1_type: str, node2_type: str) -> float:
    """
    Get maximum communication range between two node types (in km)
    Returns realistic range based on satellite orbital characteristics
    """
    # Normalize types
    type1 = node1_type.replace('_SATELLITE', '').replace('GROUND_STATION', 'GS')
    type2 = node2_type.replace('_SATELLITE', '').replace('GROUND_STATION', 'GS')
    
    # Sort alphabetically for consistent lookup
    if type1 > type2:
        type1, type2 = type2, type1
    
    key = f"{type1}_TO_{type2}"
    
    return COMM_RANGES.get(key, DEFAULT_MAX_RANGE_KM)

def calculate_path_dijkstra(source_terminal: dict, dest_terminal: dict, nodes: list, 
                           resource_aware: bool = False, drop_threshold: float = DIJKSTRA_DROP_THRESHOLD,
                           penalty_threshold: float = 80.0, penalty_multiplier: float = 3.0,
                           source_gs: Optional[dict] = None, dest_gs: Optional[dict] = None) -> dict:
    """
    Calculate path using Dijkstra's algorithm - Pure Distance Optimization
    
    Always uses find_nearest_ground_station (distance only, no resource optimization).
    Edge weights are pure distance only (no resource penalties).
    
    Args:
        source_terminal: Source terminal dict
        dest_terminal: Destination terminal dict
        nodes: List of available nodes
        resource_aware: DEPRECATED - Always False (pure distance only)
        drop_threshold: Resource utilization % above which nodes are DROPPED
        penalty_threshold: DEPRECATED - Not used
        penalty_multiplier: DEPRECATED - Not used
        source_gs: DEPRECATED - Not used (always uses nearest GS)
        dest_gs: DEPRECATED - Not used (always uses nearest GS)
    """
    graph = {}
    node_map = {node['nodeId']: node for node in nodes}
    
    def get_node_utilization(node: dict) -> float:
        """Get max resource utilization across CPU, Memory, Bandwidth"""
        cpu = node.get('cpu', {}).get('utilization', 0)
        mem = node.get('memory', {}).get('utilization', 0)
        bw = node.get('bandwidth', {}).get('utilization', 0)
        return max(cpu, mem, bw)
    
    def should_drop_node(node: dict) -> bool:
        """Drop nodes with resource usage > drop_threshold"""
        if not resource_aware:
            return False
        util = get_node_utilization(node)
        return util >= drop_threshold
    
    def calculate_edge_weight(node: dict, other_node: dict, base_distance_km: float) -> float:
        """Calculate edge weight - pure distance only (no resource penalties)"""
        return base_distance_km
    
    source_node = find_nearest_ground_station(source_terminal, nodes)
    dest_node = find_nearest_ground_station(dest_terminal, nodes)
    
    if source_node:
        source_distance_km = calculate_distance(source_terminal.get('position'), source_node.get('position')) / 1000.0
        logger.info(
            f"📐 Dijkstra (BASELINE): Selected NEAREST Ground Station {source_node['nodeId']} "
            f"for terminal {source_terminal.get('terminalId')} "
            f"(distance: {source_distance_km:.1f}km, NO resource optimization)"
        )
    
    if dest_node:
        dest_distance_km = calculate_distance(dest_terminal.get('position'), dest_node.get('position')) / 1000.0
        logger.info(
            f"📐 Dijkstra (BASELINE): Selected NEAREST Ground Station {dest_node['nodeId']} "
            f"for terminal {dest_terminal.get('terminalId')} "
            f"(distance: {dest_distance_km:.1f}km, NO resource optimization)"
        )
    
    if not source_node or not dest_node:
        logger.error(
            f"❌ Dijkstra failed: Cannot find ground stations - "
            f"source_node={'found' if source_node else 'NOT FOUND'}, "
            f"dest_node={'found' if dest_node else 'NOT FOUND'}"
        )
        return calculate_path(source_terminal, dest_terminal, nodes)
    
    available_nodes = [source_node, dest_node]
    dropped_count = 0
    
    for n in nodes:
        if n['nodeId'] not in [source_node['nodeId'], dest_node['nodeId']]:
            if should_drop_node(n):
                dropped_count += 1
                logger.debug(f"🚫 Dropped node {n['nodeId']} due to high resource usage: {get_node_utilization(n):.1f}%")
            else:
                available_nodes.append(n)
    
    if dropped_count > 0:
        logger.info(f"⚠️ Dijkstra: Dropped {dropped_count} congested nodes (resource > {drop_threshold}%)")
    
    for node in available_nodes:
        graph[node['nodeId']] = []
        for other_node in available_nodes:
            if node['nodeId'] != other_node['nodeId']:
                distance = calculate_distance(node['position'], other_node['position'])
                node_max_range = node.get('communication', {}).get('maxRangeKm', 2000) * 1000
                other_max_range = other_node.get('communication', {}).get('maxRangeKm', 2000) * 1000
                max_range = min(node_max_range, other_max_range)
                
                node_type = node.get('nodeType', '')
                other_type = other_node.get('nodeType', '')
                distance_km = distance / 1000.0
                
                if node_type == 'GROUND_STATION' and other_type == 'GROUND_STATION':
                    if distance_km <= GS_DIRECT_CONNECTION_THRESHOLD_KM:
                        base_weight = distance_km
                        edge_weight = calculate_edge_weight(node, other_node, base_weight)
                        graph[node['nodeId']].append((other_node['nodeId'], edge_weight))
                        logger.debug(
                            f"✅ Direct GS-GS connection: {node['nodeId']} → {other_node['nodeId']} "
                            f"({distance_km:.1f}km < {GS_DIRECT_CONNECTION_THRESHOLD_KM}km threshold)"
                        )
                    continue
                
                realistic_max_range = get_max_comm_range(node_type, other_type) * M_TO_KM
                
                if distance <= realistic_max_range * SATELLITE_RANGE_MARGIN:
                    # BASELINE: Pure distance only (no resource penalties)
                    base_weight = distance_km
                    edge_weight = calculate_edge_weight(node, other_node, base_weight)
                    
                    logger.debug(
                        f"📐 Dijkstra edge {node['nodeId']} → {other_node['nodeId']}: "
                        f"weight={edge_weight:.2f}km (pure distance, baseline)"
                    )
                    
                    graph[node['nodeId']].append((other_node['nodeId'], edge_weight))

    import heapq
    distances = {node_id: float('inf') for node_id in graph}
    previous = {node_id: None for node_id in graph}
    distances[source_node['nodeId']] = 0
    pq = [(0, source_node['nodeId'])]
    
    while pq:
        current_distance, current_id = heapq.heappop(pq)
        if current_distance > distances[current_id]:
            continue
        
        if current_id == dest_node['nodeId']:
            break
        
        for neighbor_id, edge_distance in graph.get(current_id, []):
            new_distance = distances[current_id] + edge_distance
            if new_distance < distances[neighbor_id]:
                distances[neighbor_id] = new_distance
                previous[neighbor_id] = current_id
                heapq.heappush(pq, (new_distance, neighbor_id))
    
    # Reconstruct path
    path_nodes = []
    current = dest_node['nodeId']
    while current:
        path_nodes.insert(0, current)
        current = previous.get(current)
    
    if not path_nodes or path_nodes[0] != source_node['nodeId']:
        logger.warning(f"⚠️ Dijkstra: No valid path found from {source_node['nodeId']} to {dest_node['nodeId']}, using fallback")
        return calculate_path(source_terminal, dest_terminal, nodes)
    
    result_path = {
        'source': {
            'terminalId': source_terminal['terminalId'],
            'position': source_terminal['position']
        },
        'destination': {
            'terminalId': dest_terminal['terminalId'],
            'position': dest_terminal['position']
        },
        'path': [],
        'totalDistance': 0,
        'estimatedLatency': 0,
        'hops': 0
    }
    
    result_path['path'].append({
        'type': 'terminal',
        'id': source_terminal['terminalId'],
        'name': source_terminal.get('terminalName', source_terminal['terminalId']),
        'position': source_terminal['position']
    })
    
    for node_id in path_nodes:
        if node_id in node_map:
            node = node_map[node_id]
            result_path['path'].append({
                'type': 'node',
                'id': node['nodeId'],
                'name': node.get('nodeName', node['nodeId']),
                'position': node['position']
            })
    
    result_path['path'].append({
        'type': 'terminal',
        'id': dest_terminal['terminalId'],
        'name': dest_terminal.get('terminalName', dest_terminal['terminalId']),
        'position': dest_terminal['position']
    })
    
    if len(result_path['path']) < 4:
        logger.warning(f"⚠️ Dijkstra path has only {len(result_path['path'])} segments, expected at least 4")
    
    total_distance = 0
    total_latency = 0.0
    
    for i in range(len(result_path['path']) - 1):
        seg1 = result_path['path'][i]
        seg2 = result_path['path'][i + 1]
        
        segment_dist = calculate_distance(seg1['position'], seg2['position'])
        total_distance += segment_dist
        
        if seg1['type'] == 'node' and seg1['id'] in node_map:
            node1 = node_map[seg1['id']]
        else:
            node1 = {'nodeType': 'GROUND_STATION', 'nodeProcessingDelayMs': 2.0}
        
        if seg2['type'] == 'node' and seg2['id'] in node_map:
            node2 = node_map[seg2['id']]
        else:
            node2 = {'nodeType': 'GROUND_STATION', 'nodeProcessingDelayMs': 2.0}
        
        segment_latency = calculate_edge_latency(node1, node2, segment_dist)
        total_latency += segment_latency
    
    result_path['totalDistance'] = round(total_distance / 1000, 2)
    result_path['estimatedLatency'] = round(total_latency, 2)
    result_path['hops'] = len(result_path['path']) - 1
    
    logger.info(
        f"📐 Dijkstra (BASELINE - Pure Distance): {result_path['hops']} hops, "
        f"{result_path['totalDistance']:.1f}km (shortest distance), "
        f"{result_path['estimatedLatency']:.2f}ms latency"
    )
    for i, seg in enumerate(result_path['path']):
        if seg['type'] == 'node' and seg['id'] in node_map:
            node = node_map[seg['id']]
            logger.debug(f"  {i+1}. {node.get('nodeType')} {seg['id']}")
    
    result_path['success'] = len(path_nodes) > 0 and path_nodes[0] == source_node['nodeId']
    
    return result_path

def calculate_path_rl(
    source_terminal: dict,
    dest_terminal: dict,
    nodes: list,
    service_qos: Optional[dict] = None
) -> dict:
    """
    Calculate path using RL agent (with fallback to heuristic)
    
    ⚠️ LƯU Ý: RL Routing hiện tại còn YẾU KÉM so với Dijkstra.
    Xem chi tiết trong docs/RL_LIMITATIONS.md để biết lý do.
    
    Args:
        source_terminal: Source terminal
        dest_terminal: Destination terminal
        nodes: List of nodes
        service_qos: Service QoS requirements (CRITICAL for drop prevention)
    """
    try:
        # Try to use RL agent
        from services.rl_routing_service import get_rl_routing_service
        from config import Config
        
        rl_service = get_rl_routing_service(Config.get_yaml_config())
        
        # Get topology if available
        topology = None
        try:
            from api.topology_bp import get_topology
            topology_response = get_topology()
            if topology_response:
                topology = topology_response.get_json()
        except:
            pass
        
        # Get scenario if available
        scenario = None
        try:
            from api.simulation_bp import get_current_scenario
            scenario_response = get_current_scenario()
            if scenario_response:
                scenario = scenario_response.get_json()
        except:
            pass
        
        # Use RL service with QoS
        path = rl_service.calculate_path_rl(
            source_terminal=source_terminal,
            dest_terminal=dest_terminal,
            nodes=nodes,
            service_qos=service_qos,
            topology=topology,
            scenario=scenario
        )
        
        return path
    
    except Exception as e:
        logger.error(f"❌ RL routing failed: {e}")
        # Re-raise error để endpoint xử lý, KHÔNG silent fallback
        raise RuntimeError(f"RL routing error: {e}") from e

def _calculate_path_rl_heuristic(source_terminal: dict, dest_terminal: dict, nodes: list) -> dict:
    """
    Calculate path using RL heuristic: Avoid congested nodes (high resource utilization)
    Prefers nodes with lower utilization and better connectivity
    """
    source_node = find_best_ground_station(source_terminal, nodes)
    dest_node = find_best_ground_station(dest_terminal, nodes)
    
    if not source_node or not dest_node:
        return calculate_path(source_terminal, dest_terminal, nodes)
    
    # FIXED: Don't filter out nodes completely - use all nodes but with congestion penalty
    # This ensures we can still find paths even in extreme congestion
    # Instead of filtering, we'll penalize congested nodes heavily in weights
    available_nodes = [n for n in nodes if n.get('isOperational', True)]
    
    # If no operational nodes, fall back to all nodes
    if not available_nodes:
        available_nodes = nodes
    
    # Penalty should be strong enough to avoid congested nodes
    def get_node_weight(node):
        utilization = node.get('resourceUtilization', 0)
        packet_count = node.get('currentPacketCount', 0)
        capacity = node.get('packetBufferCapacity', NORM_PACKET_BUFFER)
        packet_ratio = packet_count / capacity if capacity > 0 else 0
        
        if utilization > UTILIZATION_HIGH_PERCENT:
            congestion_penalty = 2.0 + (utilization - UTILIZATION_HIGH_PERCENT) / 20 * 3.0
        elif utilization > UTILIZATION_LOW_PERCENT:
            congestion_penalty = 1.0 + (utilization - UTILIZATION_LOW_PERCENT) / 20 * 1.0
        else:
            congestion_penalty = utilization / UTILIZATION_LOW_PERCENT * 1.0
        
        # Add packet buffer penalty
        buffer_penalty = packet_ratio * 0.5  # 0 to 0.5
        
        # Add weather penalty if available
        weather_penalty = 0
        weather = node.get('weather', 'CLEAR')
        if weather == 'STORM':
            weather_penalty = 1.0
        elif weather == 'RAIN':
            weather_penalty = 0.5
        elif weather == 'LIGHT_RAIN':
            weather_penalty = 0.2
        
        total_penalty = congestion_penalty + buffer_penalty + weather_penalty
        return 1 + total_penalty
    
    # Use Dijkstra-like approach but with congestion-aware weights
    import heapq
    graph = {}
    node_map = {node['nodeId']: node for node in available_nodes}
    
    # Add terminals to available nodes
    all_nodes = [source_node, dest_node] + [n for n in available_nodes 
                                             if n['nodeId'] not in [source_node['nodeId'], dest_node['nodeId']]]
    
    for node in all_nodes:
        graph[node['nodeId']] = []
        for other_node in all_nodes:
            if node['nodeId'] != other_node['nodeId']:
                distance = calculate_distance(node['position'], other_node['position'])
                
                # Get communication ranges (strict check)
                node_max_range = node.get('communication', {}).get('maxRangeKm', 2000) * 1000
                other_max_range = other_node.get('communication', {}).get('maxRangeKm', 2000) * 1000
                max_range = min(node_max_range, other_max_range)  # Use minimum range
                
                # Special rule: Ground stations can only connect directly if very close
                # Otherwise must go through satellites
                node_type = node.get('nodeType', '')
                other_type = other_node.get('nodeType', '')
                
                if node_type == 'GROUND_STATION' and other_type == 'GROUND_STATION':
                    # BẮT BUỘC: Ground stations KHÔNG được kết nối trực tiếp
                    # Phải đi qua satellites
                    pass  # Không thêm edge giữa 2 ground stations
                else:
                    # Satellite connections: allow small margin for orbital movement
                    if distance <= max_range * 1.1:  # 10% margin for satellites
                        # Weight includes congestion penalty
                        weight = distance * get_node_weight(other_node)
                        graph[node['nodeId']].append((other_node['nodeId'], weight))
    
    # Dijkstra with congestion-aware weights
    distances = {node_id: float('inf') for node_id in graph}
    previous = {node_id: None for node_id in graph}
    distances[source_node['nodeId']] = 0
    pq = [(0, source_node['nodeId'])]
    
    while pq:
        current_dist, current_id = heapq.heappop(pq)
        if current_dist > distances[current_id]:
            continue
        
        if current_id == dest_node['nodeId']:
            break
        
        for neighbor_id, edge_weight in graph.get(current_id, []):
            new_dist = distances[current_id] + edge_weight
            if new_dist < distances[neighbor_id]:
                distances[neighbor_id] = new_dist
                previous[neighbor_id] = current_id
                heapq.heappush(pq, (new_dist, neighbor_id))
    
    # Reconstruct path
    path_nodes = []
    current = dest_node['nodeId']
    while current:
        path_nodes.insert(0, current)
        current = previous.get(current)
    
    if not path_nodes or path_nodes[0] != source_node['nodeId']:
        # Fallback to simple path
        return calculate_path(source_terminal, dest_terminal, available_nodes)
    
    # Build path result
    result_path = {
        'source': {
            'terminalId': source_terminal['terminalId'],
            'position': source_terminal['position']
        },
        'destination': {
            'terminalId': dest_terminal['terminalId'],
            'position': dest_terminal['position']
        },
        'path': [],
        'totalDistance': 0,
        'estimatedLatency': 0,
        'hops': 0,
        'algorithm': 'rl',
        'optimization': 'avoid_congestion'
    }
    
    result_path['path'].append({
        'type': 'terminal',
        'id': source_terminal['terminalId'],
        'name': source_terminal.get('terminalName', source_terminal['terminalId']),
        'position': source_terminal['position']
    })
    
    for node_id in path_nodes:
        if node_id in node_map:
            node = node_map[node_id]
            result_path['path'].append({
                'type': 'node',
                'id': node['nodeId'],
                'name': node.get('nodeName', node['nodeId']),
                'position': node['position']
            })
    
    result_path['path'].append({
        'type': 'terminal',
        'id': dest_terminal['terminalId'],
        'name': dest_terminal.get('terminalName', dest_terminal['terminalId']),
        'position': dest_terminal['position']
    })
    
    # Calculate metrics
    total_distance = 0
    for i in range(len(result_path['path']) - 1):
        pos1 = result_path['path'][i]['position']
        pos2 = result_path['path'][i + 1]['position']
        segment_dist = calculate_distance(pos1, pos2)
        total_distance += segment_dist
        speed_of_light = 299792458
        propagation_delay = (segment_dist / speed_of_light) * 1000
        processing_delay = 5
        result_path['estimatedLatency'] += propagation_delay + processing_delay
    
    result_path['totalDistance'] = round(total_distance / 1000, 2)
    result_path['estimatedLatency'] = round(result_path['estimatedLatency'], 2)
    result_path['hops'] = len(result_path['path']) - 1
    
    return result_path

@routing_bp.route('/calculate-path', methods=['POST'])
def calculate_path_endpoint():
    """Calculate path from source terminal to destination terminal"""
    try:
        data = request.get_json() or {}
        source_terminal_id = data.get('sourceTerminalId')
        dest_terminal_id = data.get('destinationTerminalId')
        algorithm = data.get('algorithm', 'rl')  # 🆕 DEFAULT: RL (was 'simple')
        service_qos = data.get('serviceQos')  # Optional QoS requirements (CRITICAL for drop prevention)
        
        if not source_terminal_id or not dest_terminal_id:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'Missing sourceTerminalId or destinationTerminalId'
            }), 400
        
        if source_terminal_id == dest_terminal_id:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'Source and destination cannot be the same'
            }), 400
        
        # Get terminals
        terminals_collection = db.get_collection('terminals')
        source_terminal = terminals_collection.find_one({'terminalId': source_terminal_id}, {'_id': 0})
        dest_terminal = terminals_collection.find_one({'terminalId': dest_terminal_id}, {'_id': 0})
        
        if not source_terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Source terminal {source_terminal_id} not found'
            }), 404
        
        if not dest_terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Destination terminal {dest_terminal_id} not found'
            }), 404
        
        # Get all nodes
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Calculate path based on algorithm
        if algorithm == 'dijkstra':
            path = calculate_path_dijkstra(source_terminal, dest_terminal, nodes)
        elif algorithm == 'rl':
            path = calculate_path_rl(source_terminal, dest_terminal, nodes, service_qos=service_qos)
        else:
            path = calculate_path(source_terminal, dest_terminal, nodes)
        
        path['algorithm'] = algorithm
        
        # Clean path for JSON serialization (remove ObjectId, handle datetime, etc.)
        cleaned_path = clean_for_json(path)
        
        logger.info(f"Calculated path from {source_terminal_id} to {dest_terminal_id} using {algorithm}: {path['hops']} hops, {len(path.get('path', []))} segments")
        return jsonify(cleaned_path), 200
        
    except Exception as e:
        logger.error(f"Error calculating path: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@routing_bp.route('/send-packet', methods=['POST'])
def send_packet():
    """Send a packet from source terminal to destination terminal"""
    try:
        data = request.get_json() or {}
        source_terminal_id = data.get('sourceTerminalId')
        dest_terminal_id = data.get('destinationTerminalId')
        packet_size = data.get('packetSize', 1024)  # bytes
        priority = data.get('priority', 5)
        service_qos = data.get('serviceQos')  # Optional QoS requirements
        
        if not source_terminal_id or not dest_terminal_id:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'Missing sourceTerminalId or destinationTerminalId'
            }), 400
        
        # Get terminals
        terminals_collection = db.get_collection('terminals')
        source_terminal = terminals_collection.find_one({'terminalId': source_terminal_id}, {'_id': 0})
        dest_terminal = terminals_collection.find_one({'terminalId': dest_terminal_id}, {'_id': 0})
        
        if not source_terminal or not dest_terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': 'Source or destination terminal not found'
            }), 404
        
        # Get all nodes
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Get routing algorithm - 🆕 DEFAULT: RL
        algorithm = data.get('algorithm', 'rl')
        
        # Calculate path based on algorithm (pass QoS to RL for drop prevention)
        if algorithm == 'dijkstra':
            path = calculate_path_dijkstra(source_terminal, dest_terminal, nodes)
        elif algorithm == 'rl':
            path = calculate_path_rl(source_terminal, dest_terminal, nodes, service_qos=service_qos)
        else:
            path = calculate_path(source_terminal, dest_terminal, nodes)
        
        # Validate path meets QoS requirements if provided (RL already validates, but double-check)
        if service_qos:
            max_latency = service_qos.get('maxLatencyMs', float('inf'))
            if path.get('estimatedLatency', 0) > max_latency:
                logger.warning(
                    f"Path latency {path['estimatedLatency']}ms exceeds QoS requirement {max_latency}ms"
                )
                # Path may have drop probability calculated by RL service
        
        # IMPORTANT: Add algorithm to path for frontend color differentiation
        path['algorithm'] = algorithm
        
        # Create packet record
        packet = {
            'packetId': f"PKT-{int(datetime.now().timestamp() * 1000)}",
            'sourceTerminalId': source_terminal_id,
            'destinationTerminalId': dest_terminal_id,
            'packetSize': packet_size,
            'priority': priority,
            'path': path,
            'status': 'sent',
            'sentAt': datetime.now().isoformat(),
            'estimatedArrival': None,
            'actualArrival': None
        }
        
        # Add serviceQos if provided
        if service_qos:
            packet['serviceQos'] = service_qos
        
        # Estimate arrival time
        if path['estimatedLatency'] > 0:
            from datetime import timedelta
            arrival_time = datetime.now() + timedelta(milliseconds=path['estimatedLatency'])
            packet['estimatedArrival'] = arrival_time.isoformat()
        
        # Store packet vào database
        try:
            packets_collection = db.get_collection('packets')
            result = packets_collection.insert_one(packet)
            logger.debug(f"Packet {packet['packetId']} stored in database with _id: {result.inserted_id}")
        except Exception as e:
            logger.warning(f"Could not store packet in database: {e}")
        
        # Lưu traffic demand summary vào traffic_demand collection
        try:
            _update_traffic_demand(db, source_terminal, dest_terminal, path, packet_size)
        except Exception as e:
            logger.warning(f"Could not update traffic demand: {e}")
        
        # Clean packet for JSON serialization (remove _id if present)
        clean_packet = clean_for_json(packet)
        
        logger.info(f"Packet sent from {source_terminal_id} to {dest_terminal_id}")
        return jsonify(clean_packet), 200
        
    except Exception as e:
        logger.error(f"Error sending packet: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


def _update_traffic_demand(db, source_terminal: dict, dest_terminal: dict, path: dict, packet_size: int):
    """
    Cập nhật traffic_demand collection với thông tin routing
    Tổng hợp traffic demand theo node và terminal
    """
    from datetime import datetime
    from collections import defaultdict
    
    try:
        traffic_demand_collection = db.get_collection('traffic_demand')
        
        # Extract nodes từ path
        path_segments = path.get('path', [])
        nodes_in_path = [seg.get('id') for seg in path_segments if seg.get('type') == 'node']
        
        # Update traffic demand cho từng node trong path
        for node_id in nodes_in_path:
            # Tìm hoặc tạo traffic demand record cho node này
            today = datetime.now().date().isoformat()
            query = {
                'nodeId': node_id,
                'date': today
            }
            
            update = {
                '$inc': {
                    'totalPackets': 1,
                    'totalBytes': packet_size,
                    'incomingTraffic': packet_size,
                    'outgoingTraffic': packet_size
                },
                '$set': {
                    'lastUpdated': datetime.now().isoformat(),
                    'nodeId': node_id,
                    'date': today
                },
                '$addToSet': {
                    'sourceTerminals': source_terminal.get('terminalId'),
                    'destTerminals': dest_terminal.get('terminalId')
                }
            }
            
            traffic_demand_collection.update_one(query, update, upsert=True)
        
        # Update terminal traffic demand
        source_terminal_id = source_terminal.get('terminalId')
        dest_terminal_id = dest_terminal.get('terminalId')
        
        today = datetime.now().date().isoformat()
        
        # Source terminal
        traffic_demand_collection.update_one(
            {'terminalId': source_terminal_id, 'date': today},
            {
                '$inc': {
                    'outgoingPackets': 1,
                    'outgoingBytes': packet_size
                },
                '$set': {
                    'lastUpdated': datetime.now().isoformat(),
                    'terminalId': source_terminal_id,
                    'date': today
                }
            },
            upsert=True
        )
        
        # Destination terminal
        traffic_demand_collection.update_one(
            {'terminalId': dest_terminal_id, 'date': today},
            {
                '$inc': {
                    'incomingPackets': 1,
                    'incomingBytes': packet_size
                },
                '$set': {
                    'lastUpdated': datetime.now().isoformat(),
                    'terminalId': dest_terminal_id,
                    'date': today
                }
            },
            upsert=True
        )
        
    except Exception as e:
        logger.error(f"Error updating traffic demand: {e}")


@routing_bp.route('/compare-algorithms', methods=['POST'])
def compare_algorithms():
    """Compare two routing algorithms for a given source and destination terminal"""
    try:
        data = request.get_json() or {}
        source_terminal_id = data.get('sourceTerminalId')
        dest_terminal_id = data.get('destinationTerminalId')
        service_qos = data.get('serviceQos')
        algorithm1 = data.get('algorithm1', 'dijkstra')  # First algorithm
        algorithm2 = data.get('algorithm2', 'rl')  # Second algorithm
        scenario = data.get('scenario', 'NORMAL')  # Simulation scenario
        
        if not source_terminal_id or not dest_terminal_id:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'Missing sourceTerminalId or destinationTerminalId'
            }), 400
        
        if source_terminal_id == dest_terminal_id:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'Source and destination cannot be the same'
            }), 400
        
        if algorithm1 not in ['simple', 'dijkstra', 'rl'] or algorithm2 not in ['simple', 'dijkstra', 'rl']:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'Invalid algorithm. Must be one of: simple, dijkstra, rl'
            }), 400
        
        # Get terminals
        terminals_collection = db.get_collection('terminals')
        source_terminal = terminals_collection.find_one({'terminalId': source_terminal_id}, {'_id': 0})
        dest_terminal = terminals_collection.find_one({'terminalId': dest_terminal_id}, {'_id': 0})
        
        if not source_terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Source terminal {source_terminal_id} not found'
            }), 404
        
        if not dest_terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Destination terminal {dest_terminal_id} not found'
            }), 404
        
        # Get all nodes
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Calculate path for algorithm 1
        if algorithm1 == 'dijkstra':
            path1 = calculate_path_dijkstra(source_terminal, dest_terminal, nodes)
        elif algorithm1 == 'rl':
            path1 = calculate_path_rl(source_terminal, dest_terminal, nodes, service_qos=service_qos)
        else:
            path1 = calculate_path(source_terminal, dest_terminal, nodes)
        path1['algorithm'] = algorithm1
        
        # Calculate path for algorithm 2
        if algorithm2 == 'dijkstra':
            path2 = calculate_path_dijkstra(source_terminal, dest_terminal, nodes)
        elif algorithm2 == 'rl':
            path2 = calculate_path_rl(source_terminal, dest_terminal, nodes, service_qos=service_qos)
        else:
            path2 = calculate_path(source_terminal, dest_terminal, nodes)
        path2['algorithm'] = algorithm2
        
        # Check QoS requirements if provided
        qos_met_1 = True
        qos_met_2 = True
        qos_warnings = []
        
        if service_qos:
            max_latency = service_qos.get('maxLatencyMs', float('inf'))
            if path1['estimatedLatency'] > max_latency:
                qos_met_1 = False
                qos_warnings.append(f'{algorithm1} path latency {path1["estimatedLatency"]}ms exceeds QoS requirement {max_latency}ms')
            if path2['estimatedLatency'] > max_latency:
                qos_met_2 = False
                qos_warnings.append(f'{algorithm2} path latency {path2["estimatedLatency"]}ms exceeds QoS requirement {max_latency}ms')
        
        # Extract node IDs from paths for resource lookup
        node_ids_path1 = [seg['id'] for seg in path1['path'] if seg['type'] == 'node']
        node_ids_path2 = [seg['id'] for seg in path2['path'] if seg['type'] == 'node']
        all_node_ids = list(set(node_ids_path1 + node_ids_path2))
        
        # Get node resources
        node_resources = {}
        for node_id in all_node_ids:
            node = next((n for n in nodes if n['nodeId'] == node_id), None)
            if node:
                node_resources[node_id] = {
                    'nodeId': node_id,
                    'nodeName': node.get('nodeName', node_id),
                    'nodeType': node.get('nodeType', 'UNKNOWN'),
                    'isOperational': node.get('isOperational', False),
                    'resourceUtilization': node.get('resourceUtilization', 0),
                    'currentPacketCount': node.get('currentPacketCount', 0),
                    'packetBufferCapacity': node.get('packetBufferCapacity', 0),
                    'nodeProcessingDelayMs': node.get('nodeProcessingDelayMs', 0),
                    'packetLossRate': node.get('packetLossRate', 0),
                    'batteryChargePercent': node.get('batteryChargePercent', 100)
                }
        
        # Build comparison result
        timestamp = datetime.now()
        comparison_result = {
            'sourceTerminalId': source_terminal_id,
            'destinationTerminalId': dest_terminal_id,
            'serviceQos': service_qos,
            'scenario': scenario,
            'algorithm1': {
                'name': algorithm1,
                'path': path1,
                'qosMet': qos_met_1
            },
            'algorithm2': {
                'name': algorithm2,
                'path': path2,
                'qosMet': qos_met_2
            },
            'comparison': {
                'distanceDifference': round(path1['totalDistance'] - path2['totalDistance'], 2),
                'latencyDifference': round(path1['estimatedLatency'] - path2['estimatedLatency'], 2),
                'hopsDifference': path1['hops'] - path2['hops'],
                'bestDistance': algorithm1 if path1['totalDistance'] < path2['totalDistance'] else algorithm2,
                'bestLatency': algorithm1 if path1['estimatedLatency'] < path2['estimatedLatency'] else algorithm2,
                'bestHops': algorithm1 if path1['hops'] < path2['hops'] else algorithm2
            },
            'nodeResources': node_resources,
            'qosWarnings': qos_warnings,
            'timestamp': timestamp.isoformat()
        }
        
        # Save to database
        try:
            comparisons_collection = db.get_collection('algorithm_comparisons')
            comparison_doc = {
                **comparison_result,
                'createdAt': timestamp
            }
            comparisons_collection.insert_one(comparison_doc)
            logger.info(f"Saved comparison to database: {algorithm1} vs {algorithm2}")
        except Exception as save_error:
            logger.error(f"Failed to save comparison to database: {save_error}")
            # Don't fail the request if save fails
        
        logger.info(f"Compared {algorithm1} vs {algorithm2} from {source_terminal_id} to {dest_terminal_id}")
        return jsonify(comparison_result), 200
        
    except Exception as e:
        logger.error(f"Error comparing algorithms: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

