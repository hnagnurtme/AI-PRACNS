"""
Simulation Scenario API Blueprint
Provides endpoints for managing simulation scenarios with active resource modification
"""
from flask import Blueprint, jsonify, request
from datetime import datetime
import logging
import random
from models.database import Database

logger = logging.getLogger(__name__)

simulation_bp = Blueprint('simulation', __name__, url_prefix='/api/v1/simulation')

# Database instance
db = Database()
# Ensure database is connected
if not db.is_connected():
    db.connect()

# Available scenario types - Real-world network conditions
SCENARIO_TYPES = [
    'NORMAL',              # Điều kiện bình thường
    'PEAK_HOURS',          # Giờ cao điểm - tải cao (GLOBAL)
    'STORM_WEATHER',       # Thời tiết xấu - ảnh hưởng tín hiệu (LOCAL - khu vực)
    'HEAVY_TRAFFIC',       # Lưu lượng lớn - băng thông thấp (GLOBAL)
    'REMOTE_AREA',         # Vùng xa - tín hiệu yếu (LOCAL - khu vực cụ thể)
    'EQUIPMENT_AGING',     # Thiết bị cũ - hiệu suất giảm (LOCAL - một số thiết bị)
    'MAINTENANCE_MODE',    # Bảo trì - một số node offline (LOCAL - đã có logic)
    'EMERGENCY_LOAD',      # Tải khẩn cấp - quá tải hệ thống (GLOBAL)
]

# Scenario classification
GLOBAL_SCENARIOS = ['PEAK_HOURS', 'HEAVY_TRAFFIC', 'EMERGENCY_LOAD']
LOCAL_SCENARIOS = ['STORM_WEATHER', 'REMOTE_AREA', 'EQUIPMENT_AGING', 'MAINTENANCE_MODE']

# Current active scenario (in-memory, could be stored in DB or Redis)
_current_scenario = {
    'scenario': 'NORMAL',
    'displayName': 'Normal',
    'description': 'Normal network conditions',
    'isActive': True,
    'startedAt': datetime.now().isoformat(),
    'parameters': {}
}

@simulation_bp.route('/scenarios', methods=['GET'])
def get_scenarios():
    """Get all available simulation scenarios"""
    try:
        scenarios = []
        for scenario_name in SCENARIO_TYPES:
            scenarios.append({
                'name': scenario_name,
                'displayName': scenario_name.replace('_', ' ').title(),
                'description': _get_scenario_description(scenario_name),
            })
        
        logger.info(f"Scenarios requested: {len(scenarios)} scenarios")
        return jsonify(scenarios), 200
    except Exception as e:
        logger.error(f"Error getting scenarios: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@simulation_bp.route('/scenario/current', methods=['GET'])
def get_current_scenario():
    """Get current active scenario"""
    try:
        return jsonify(_current_scenario), 200
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting current scenario: {error_msg}", exc_info=True)
        try:
            return jsonify({
                'status': 500,
                'error': 'Internal server error',
                'message': error_msg
            }), 500
        except Exception as json_error:
            logger.error(f"Error creating JSON response: {json_error}")
            return {'status': 500, 'error': 'Internal server error', 'message': error_msg}, 500

@simulation_bp.route('/scenario/<scenario_name>', methods=['POST'])
def set_scenario(scenario_name: str):
    """Set active simulation scenario and apply resource modifications"""
    try:
        if scenario_name not in SCENARIO_TYPES:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': f'Invalid scenario: {scenario_name}'
            }), 400
        
        data = request.get_json() or {}
        parameters = data.get('parameters', {})
        
        # Apply scenario effects to nodes
        success = _apply_scenario_to_nodes(scenario_name, parameters)
        
        if not success:
            return jsonify({
                'status': 500,
                'error': 'Internal server error',
                'message': 'Failed to apply scenario modifications'
            }), 500
        
        # Update current scenario
        global _current_scenario
        _current_scenario = {
            'scenario': scenario_name,
            'displayName': scenario_name.replace('_', ' ').title(),
            'description': _get_scenario_description(scenario_name),
            'isActive': True,
            'startedAt': datetime.now().isoformat(),
            'parameters': parameters,
            'resourceModificationsApplied': True
        }
        
        logger.info(f"Scenario changed to: {scenario_name} with resource modifications")
        return jsonify({
            'success': True,
            'scenario': _current_scenario
        }), 200
    except Exception as e:
        logger.error(f"Error setting scenario: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@simulation_bp.route('/scenario/reset', methods=['POST'])
def reset_scenario():
    """Reset scenario to NORMAL and restore baseline node resources"""
    try:
        # Restore baseline nodes
        _restore_baseline_nodes()
        
        global _current_scenario
        _current_scenario = {
            'scenario': 'NORMAL',
            'displayName': 'Normal',
            'description': 'Normal network conditions',
            'isActive': True,
            'startedAt': datetime.now().isoformat(),
            'parameters': {},
            'resourceModificationsApplied': True
        }
        
        logger.info("Scenario reset to NORMAL with baseline restoration")
        return jsonify({
            'success': True,
            'scenario': _current_scenario
        }), 200
    except Exception as e:
        logger.error(f"Error resetting scenario: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@simulation_bp.route('/start', methods=['POST'])
def start_simulation():
    """Start simulation with a scenario and apply resource modifications"""
    try:
        data = request.get_json() or {}
        scenario_name = data.get('scenario', 'NORMAL')
        parameters = data.get('parameters', {})
        
        if scenario_name not in SCENARIO_TYPES:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': f'Invalid scenario: {scenario_name}'
            }), 400
        
        # Apply scenario effects
        success = _apply_scenario_to_nodes(scenario_name, parameters)
        
        if not success:
            return jsonify({
                'status': 500,
                'error': 'Internal server error',
                'message': 'Failed to apply scenario modifications'
            }), 500
        
        global _current_scenario
        _current_scenario = {
            'scenario': scenario_name,
            'displayName': scenario_name.replace('_', ' ').title(),
            'description': _get_scenario_description(scenario_name),
            'isActive': True,
            'startedAt': datetime.now().isoformat(),
            'parameters': parameters,
            'resourceModificationsApplied': True
        }
        
        logger.info(f"Simulation started with scenario: {scenario_name} (resources modified)")
        return jsonify(_current_scenario), 200
    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500
        return jsonify(_current_scenario), 200
    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@simulation_bp.route('/stop', methods=['POST'])
def stop_simulation():
    """Stop simulation, reset to NORMAL and restore baseline resources"""
    try:
        # Restore baseline nodes
        _restore_baseline_nodes()
        
        global _current_scenario
        _current_scenario = {
            'scenario': 'NORMAL',
            'displayName': 'Normal',
            'description': 'Normal network conditions',
            'isActive': False,
            'startedAt': datetime.now().isoformat(),
            'parameters': {},
            'resourceModificationsApplied': True
        }
        
        logger.info("Simulation stopped and resources restored to baseline")
        return jsonify({
            'success': True,
            'scenario': _current_scenario
        }), 200
    except Exception as e:
        logger.error(f"Error stopping simulation: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@simulation_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get simulation metrics"""
    try:
        # TODO: Calculate actual metrics from packet data
        # For now, return placeholder metrics
        metrics = {
            'totalPackets': 0,
            'successfulPackets': 0,
            'failedPackets': 0,
            'averageLatency': 0,
            'averageDistance': 0,
            'averageHops': 0,
            'algorithmPerformance': {
                'dijkstra': {
                    'avgLatency': 0,
                    'successRate': 0,
                    'avgHops': 0
                },
                'rl': {
                    'avgLatency': 0,
                    'successRate': 0,
                    'avgHops': 0
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

def _get_scenario_description(scenario_name: str) -> str:
    """Get description for a scenario"""
    descriptions = {
        'NORMAL': 'Điều kiện mạng bình thường, tất cả hoạt động ổn định',
        'PEAK_HOURS': 'Giờ cao điểm - nhiều người dùng cùng lúc, tải cao',
        'STORM_WEATHER': 'Thời tiết xấu - ảnh hưởng tín hiệu vệ tinh và trạm mặt đất',
        'HEAVY_TRAFFIC': 'Lưu lượng truy cập lớn - băng thông bị chia sẻ nhiều',
        'REMOTE_AREA': 'Khu vực xa xôi - tín hiệu yếu, độ trễ cao',
        'EQUIPMENT_AGING': 'Thiết bị cũ - hiệu suất giảm, mất gói tăng',
        'MAINTENANCE_MODE': 'Chế độ bảo trì - một số node tạm ngưng hoạt động',
        'EMERGENCY_LOAD': 'Tải khẩn cấp - hệ thống quá tải do nhu cầu đột biến',
    }
    return descriptions.get(scenario_name, f'{scenario_name.replace("_", " ").lower()} scenario')


def _get_neighbor_nodes(node, all_nodes, max_distance_km=2000):
    """Get neighboring nodes within a certain distance"""
    import math
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km"""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    neighbors = []
    node_lat = node.get('latitude', 0)
    node_lon = node.get('longitude', 0)
    
    for other_node in all_nodes:
        if other_node['_id'] == node['_id']:
            continue
        
        other_lat = other_node.get('latitude', 0)
        other_lon = other_node.get('longitude', 0)
        
        distance = haversine_distance(node_lat, node_lon, other_lat, other_lon)
        if distance <= max_distance_km:
            neighbors.append(other_node)
    
    return neighbors


def _store_baseline_nodes():
    """Store baseline node states before modification"""
    try:
        nodes_collection = db.get_collection('nodes')
        baseline_collection = db.get_collection('nodes_baseline')
        
        # Check if baseline already exists
        if baseline_collection.count_documents({}) > 0:
            logger.info("Baseline already exists, skipping storage")
            return
        
        # Copy all nodes to baseline
        nodes = list(nodes_collection.find({}))
        if nodes:
            baseline_collection.insert_many(nodes)
            logger.info(f"Stored {len(nodes)} nodes as baseline")
    except Exception as e:
        logger.error(f"Error storing baseline nodes: {e}")


def _restore_baseline_nodes():
    """Restore nodes from baseline"""
    try:
        nodes_collection = db.get_collection('nodes')
        baseline_collection = db.get_collection('nodes_baseline')
        
        baseline_nodes = list(baseline_collection.find({}))
        if not baseline_nodes:
            logger.warning("No baseline nodes found")
            return
        
        # Update each node from baseline
        for node in baseline_nodes:
            node_id = node['_id']
            nodes_collection.update_one(
                {'_id': node_id},
                {'$set': {
                    'resourceUtilization': node.get('resourceUtilization', 0),
                    'packetLossRate': node.get('packetLossRate', 0),
                    'bandwidthMbps': node.get('bandwidthMbps', 100),
                    'queueLength': node.get('queueLength', 0),
                    'batteryChargePercent': node.get('batteryChargePercent', 100),
                    'signalStrengthDbm': node.get('signalStrengthDbm', -50),
                    'status': node.get('status', 'active')
                }}
            )
        
        logger.info(f"Restored {len(baseline_nodes)} nodes from baseline")
    except Exception as e:
        logger.error(f"Error restoring baseline nodes: {e}")


def _apply_scenario_to_nodes(scenario_name: str, parameters = None):
    """Apply scenario effects to nodes in database"""
    try:
        # Store baseline before first modification
        _store_baseline_nodes()
        
        nodes_collection = db.get_collection('nodes')
        parameters = parameters or {}
        
        if scenario_name == 'NORMAL':
            # Reset all nodes to healthy state with realistic random values
            nodes = list(nodes_collection.find({}))
            for node in nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(5, 25),  # 5-25% utilization
                        'packetLossRate': random.uniform(0, 0.005),    # 0-0.5% packet loss
                        'bandwidthMbps': random.uniform(90, 100),      # 90-100 Mbps
                        'queueLength': random.randint(0, 10),          # 0-10 packets in queue
                        'batteryChargePercent': random.uniform(85, 100), # 85-100% battery
                        'signalStrengthDbm': random.uniform(-55, -45), # Good signal strength
                        'status': 'active'
                    }}
                )
            logger.info(f"Reset {len(nodes)} nodes to healthy NORMAL state with realistic values")
            
        elif scenario_name == 'PEAK_HOURS':
            # GLOBAL: Giờ cao điểm - tải cao đồng đều trên toàn hệ thống
            nodes = list(nodes_collection.find({}))
            for node in nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(60, 85),  # Tải cao đồng đều
                        'packetLossRate': random.uniform(0.01, 0.03),   # Mất gói tăng nhẹ
                        'queueLength': random.randint(30, 80),          # Hàng đợi dài
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.7, 0.85), # Băng thông chia sẻ
                        'status': 'active'
                    }}
                )
            logger.info(f"Applied PEAK_HOURS (GLOBAL) to {len(nodes)} nodes")
            
        elif scenario_name == 'STORM_WEATHER':
            # LOCAL: Thời tiết xấu - ảnh hưởng khu vực (30-50% nodes)
            nodes = list(nodes_collection.find({}))
            num_affected = int(len(nodes) * random.uniform(0.3, 0.5))  # 30-50% nodes bị ảnh hưởng
            affected_nodes = set()
            
            # Chọn các cặp nodes gần nhau
            available_nodes = nodes.copy()
            while len(affected_nodes) < num_affected and available_nodes:
                # Chọn node ngẫu nhiên
                base_node = random.choice(available_nodes)
                available_nodes.remove(base_node)
                
                # Lấy neighbor gần nhất
                neighbors = _get_neighbor_nodes(base_node, nodes, max_distance_km=1500)
                
                if neighbors:
                    neighbor = random.choice(neighbors)
                    # 1 node tệ, 1 node ổn hơn
                    bad_node = random.choice([base_node, neighbor])
                    good_node = neighbor if bad_node == base_node else base_node
                    
                    # Node tệ - thời tiết xấu
                    nodes_collection.update_one(
                        {'_id': bad_node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-95, -75),  # Tín hiệu rất yếu
                            'packetLossRate': random.uniform(0.1, 0.2),     # Mất gói cao
                            'resourceUtilization': random.uniform(50, 70),   # Tải cao do retry
                            'queueLength': random.randint(40, 80),
                            'status': 'active'
                        }}
                    )
                    
                    # Node ổn hơn - ảnh hưởng nhẹ
                    nodes_collection.update_one(
                        {'_id': good_node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-70, -60),  # Tín hiệu yếu nhưng còn dùng được
                            'packetLossRate': random.uniform(0.02, 0.05),   # Mất gói thấp
                            'resourceUtilization': random.uniform(30, 50),
                            'queueLength': random.randint(10, 30),
                            'status': 'active'
                        }}
                    )
                    
                    affected_nodes.add(bad_node['_id'])
                    affected_nodes.add(good_node['_id'])
                else:
                    # Không có neighbor, áp dụng cho node này thôi
                    nodes_collection.update_one(
                        {'_id': base_node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-85, -65),
                            'packetLossRate': random.uniform(0.05, 0.15),
                            'resourceUtilization': random.uniform(40, 60),
                            'queueLength': random.randint(20, 50),
                            'status': 'active'
                        }}
                    )
                    affected_nodes.add(base_node['_id'])
            
            # Các node còn lại giữ trạng thái bình thường
            for node in nodes:
                if node['_id'] not in affected_nodes:
                    nodes_collection.update_one(
                        {'_id': node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-55, -45),
                            'packetLossRate': random.uniform(0, 0.01),
                            'resourceUtilization': random.uniform(10, 30),
                            'queueLength': random.randint(0, 15),
                            'status': 'active'
                        }}
                    )
            
            logger.info(f"Applied STORM_WEATHER (LOCAL) to {len(affected_nodes)} nodes, {len(nodes) - len(affected_nodes)} normal")
            
        elif scenario_name == 'HEAVY_TRAFFIC':
            # GLOBAL: Lưu lượng lớn đồng đều - toàn hệ thống bị tắc nghẽn
            nodes = list(nodes_collection.find({}))
            for node in nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.3, 0.6), # Băng thông rất thấp
                        'resourceUtilization': random.uniform(70, 90),   # Tải cao
                        'packetLossRate': random.uniform(0.02, 0.06),    # Mất gói do tắc nghẽn
                        'queueLength': random.randint(50, 120),          # Hàng đợi rất dài
                        'status': 'active'
                    }}
                )
            logger.info(f"Applied HEAVY_TRAFFIC (GLOBAL) to {len(nodes)} nodes")
            
        elif scenario_name == 'REMOTE_AREA':
            # LOCAL: Vùng xa cụ thể - chỉ một số khu vực (20-35% nodes)
            nodes = list(nodes_collection.find({}))
            num_affected = int(len(nodes) * random.uniform(0.2, 0.35))  # 20-35% nodes
            affected_nodes = set()
            
            available_nodes = nodes.copy()
            while len(affected_nodes) < num_affected and available_nodes:
                base_node = random.choice(available_nodes)
                available_nodes.remove(base_node)
                
                neighbors = _get_neighbor_nodes(base_node, nodes, max_distance_km=2000)
                
                if neighbors:
                    neighbor = random.choice(neighbors)
                    bad_node = random.choice([base_node, neighbor])
                    good_node = neighbor if bad_node == base_node else base_node
                    
                    # Node tệ - vùng xa, tín hiệu rất yếu
                    nodes_collection.update_one(
                        {'_id': bad_node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-100, -85),  # Tín hiệu cực yếu
                            'packetLossRate': random.uniform(0.15, 0.3),     # Mất gói rất cao
                            'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.15, 0.35),
                            'resourceUtilization': random.uniform(40, 60),
                            'queueLength': random.randint(30, 70),
                            'status': 'active'
                        }}
                    )
                    
                    # Node ổn hơn - biên vùng xa
                    nodes_collection.update_one(
                        {'_id': good_node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-75, -60),   # Tín hiệu yếu nhưng dùng được
                            'packetLossRate': random.uniform(0.03, 0.08),
                            'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.5, 0.7),
                            'resourceUtilization': random.uniform(25, 45),
                            'queueLength': random.randint(10, 30),
                            'status': 'active'
                        }}
                    )
                    
                    affected_nodes.add(bad_node['_id'])
                    affected_nodes.add(good_node['_id'])
                else:
                    nodes_collection.update_one(
                        {'_id': base_node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-95, -75),
                            'packetLossRate': random.uniform(0.08, 0.2),
                            'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.2, 0.5),
                            'resourceUtilization': random.uniform(30, 50),
                            'status': 'active'
                        }}
                    )
                    affected_nodes.add(base_node['_id'])
            
            # Các node còn lại bình thường
            for node in nodes:
                if node['_id'] not in affected_nodes:
                    nodes_collection.update_one(
                        {'_id': node['_id']},
                        {'$set': {
                            'signalStrengthDbm': random.uniform(-55, -45),
                            'packetLossRate': random.uniform(0, 0.01),
                            'bandwidthMbps': random.uniform(85, 100),
                            'resourceUtilization': random.uniform(10, 30),
                            'queueLength': random.randint(0, 15),
                            'status': 'active'
                        }}
                    )
            
            logger.info(f"Applied REMOTE_AREA (LOCAL) to {len(affected_nodes)} nodes, {len(nodes) - len(affected_nodes)} normal")
            
        elif scenario_name == 'EQUIPMENT_AGING':
            # LOCAL: Thiết bị cũ - chỉ một số thiết bị cụ thể (25-40% nodes)
            nodes = list(nodes_collection.find({}))
            num_affected = int(len(nodes) * random.uniform(0.25, 0.4))  # 25-40% nodes
            affected_nodes = set()
            
            available_nodes = nodes.copy()
            while len(affected_nodes) < num_affected and available_nodes:
                base_node = random.choice(available_nodes)
                available_nodes.remove(base_node)
                
                neighbors = _get_neighbor_nodes(base_node, nodes, max_distance_km=1800)
                
                if neighbors:
                    neighbor = random.choice(neighbors)
                    bad_node = random.choice([base_node, neighbor])
                    good_node = neighbor if bad_node == base_node else base_node
                    
                    # Node tệ - thiết bị cũ
                    nodes_collection.update_one(
                        {'_id': bad_node['_id']},
                        {'$set': {
                            'resourceUtilization': random.uniform(65, 85),   # Tải cao do xử lý chậm
                            'packetLossRate': random.uniform(0.06, 0.12),    # Mất gói do lỗi phần cứng
                            'batteryChargePercent': random.uniform(30, 55),  # Pin yếu
                            'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.5, 0.7),
                            'queueLength': random.randint(25, 60),
                            'signalStrengthDbm': random.uniform(-70, -60),
                            'status': 'active'
                        }}
                    )
                    
                    # Node ổn hơn - thiết bị mới hơn
                    nodes_collection.update_one(
                        {'_id': good_node['_id']},
                        {'$set': {
                            'resourceUtilization': random.uniform(20, 40),
                            'packetLossRate': random.uniform(0.01, 0.03),
                            'batteryChargePercent': random.uniform(70, 90),
                            'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.8, 0.95),
                            'queueLength': random.randint(5, 20),
                            'signalStrengthDbm': random.uniform(-55, -48),
                            'status': 'active'
                        }}
                    )
                    
                    affected_nodes.add(bad_node['_id'])
                    affected_nodes.add(good_node['_id'])
                else:
                    nodes_collection.update_one(
                        {'_id': base_node['_id']},
                        {'$set': {
                            'resourceUtilization': random.uniform(50, 75),
                            'packetLossRate': random.uniform(0.03, 0.08),
                            'batteryChargePercent': random.uniform(40, 70),
                            'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.6, 0.8),
                            'queueLength': random.randint(15, 40),
                            'status': 'active'
                        }}
                    )
                    affected_nodes.add(base_node['_id'])
            
            # Các node còn lại bình thường
            for node in nodes:
                if node['_id'] not in affected_nodes:
                    nodes_collection.update_one(
                        {'_id': node['_id']},
                        {'$set': {
                            'resourceUtilization': random.uniform(10, 30),
                            'packetLossRate': random.uniform(0, 0.01),
                            'batteryChargePercent': random.uniform(80, 100),
                            'bandwidthMbps': random.uniform(85, 100),
                            'queueLength': random.randint(0, 15),
                            'signalStrengthDbm': random.uniform(-55, -45),
                            'status': 'active'
                        }}
                    )
            
            logger.info(f"Applied EQUIPMENT_AGING (LOCAL) to {len(affected_nodes)} nodes, {len(nodes) - len(affected_nodes)} normal")
            
        elif scenario_name == 'MAINTENANCE_MODE':
            # LOCAL: Bảo trì - một số node offline, neighbors chịu tải
            nodes = list(nodes_collection.find({}))
            num_maintenance = int(len(nodes) * random.uniform(0.15, 0.3))  # 15-30% nodes bảo trì
            maintenance_nodes = random.sample(nodes, num_maintenance)
            high_load_neighbors = set()
            
            # Các node bảo trì
            for node in maintenance_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'status': 'maintenance',
                        'resourceUtilization': 0,
                        'packetLossRate': 1.0,  # Không hoạt động
                        'bandwidthMbps': 0,
                        'queueLength': 0
                    }}
                )
                
                # Tìm neighbors của node bảo trì - họ sẽ chịu tải cao hơn
                neighbors = _get_neighbor_nodes(node, nodes, max_distance_km=1500)
                for neighbor in neighbors:
                    if neighbor not in maintenance_nodes:
                        high_load_neighbors.add(neighbor['_id'])
            
            # Các node neighbors của nodes bảo trì - chịu tải cao
            for node in nodes:
                if node['_id'] in high_load_neighbors:
                    nodes_collection.update_one(
                        {'_id': node['_id']},
                        {'$set': {
                            'resourceUtilization': random.uniform(70, 90),  # Tải rất cao
                            'packetLossRate': random.uniform(0.03, 0.07),
                            'queueLength': random.randint(40, 80),
                            'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.6, 0.8),
                            'status': 'active'
                        }}
                    )
                elif node not in maintenance_nodes:
                    # Các node khác - tải tăng nhẹ
                    nodes_collection.update_one(
                        {'_id': node['_id']},
                        {'$set': {
                            'resourceUtilization': random.uniform(35, 55),
                            'packetLossRate': random.uniform(0.01, 0.03),
                            'queueLength': random.randint(15, 35),
                            'status': 'active'
                        }}
                    )
            
            logger.info(f"Applied MAINTENANCE_MODE (LOCAL): {num_maintenance} maintenance, {len(high_load_neighbors)} high-load neighbors, others moderate")
            
        elif scenario_name == 'EMERGENCY_LOAD':
            # GLOBAL: Tải khẩn cấp - quá tải toàn hệ thống
            nodes = list(nodes_collection.find({}))
            for node in nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(85, 98),   # Quá tải đồng đều
                        'packetLossRate': random.uniform(0.1, 0.25),     # Mất gói rất cao
                        'queueLength': random.randint(100, 200),         # Hàng đợi cực dài
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.4, 0.6), # Băng thông khan hiếm
                        'signalStrengthDbm': random.uniform(-70, -55),   # Tín hiệu giảm do nhiễu
                        'batteryChargePercent': random.uniform(30, 60),  # Pin hao nhanh
                        'status': 'active'
                    }}
                )
            logger.info(f"Applied EMERGENCY_LOAD (GLOBAL) to {len(nodes)} nodes")
        
        return True
        
    except Exception as e:
        logger.error(f"Error applying scenario {scenario_name}: {e}")
        return False
