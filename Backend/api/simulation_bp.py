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
    """
    Apply scenario effects ONLY to trap nodes in database.
    Non-trap nodes remain at healthy baseline state.
    This makes scenarios more realistic - trap nodes represent problematic nodes that RL should learn to avoid.
    """
    try:
        # Store baseline before first modification
        _store_baseline_nodes()
        
        nodes_collection = db.get_collection('nodes')
        parameters = parameters or {}
        
        # Separate trap nodes from normal nodes
        trap_nodes = list(nodes_collection.find({'isTrapNode': True}))
        normal_nodes = list(nodes_collection.find({'isTrapNode': {'$ne': True}}))
        
        logger.info(f"Applying scenario {scenario_name}: {len(trap_nodes)} trap nodes, {len(normal_nodes)} normal nodes")
        
        # Helper function to set healthy baseline for normal nodes
        def set_healthy_baseline(node):
            nodes_collection.update_one(
                {'_id': node['_id']},
                {'$set': {
                    'resourceUtilization': random.uniform(5, 25),    # 5-25% utilization
                    'packetLossRate': random.uniform(0, 0.005),      # 0-0.5% packet loss
                    'bandwidthMbps': random.uniform(90, 100),        # 90-100 Mbps
                    'queueLength': random.randint(0, 10),            # 0-10 packets
                    'batteryChargePercent': random.uniform(85, 100), # 85-100% battery
                    'signalStrengthDbm': random.uniform(-55, -45),   # Good signal
                    'status': 'active'
                }}
            )
        
        if scenario_name == 'NORMAL':
            # Reset ALL nodes to healthy state
            all_nodes = list(nodes_collection.find({}))
            for node in all_nodes:
                set_healthy_baseline(node)
            logger.info(f"Reset {len(all_nodes)} nodes to healthy NORMAL state")
            
        elif scenario_name == 'PEAK_HOURS':
            # TRAP NODES: High load during peak hours
            for node in trap_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(75, 95),   # Very high load
                        'packetLossRate': random.uniform(0.03, 0.08),    # Higher packet loss
                        'queueLength': random.randint(50, 100),          # Long queue
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.5, 0.7),
                        'status': 'active'
                    }}
                )
            # NORMAL NODES: Slightly elevated but still good
            for node in normal_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(20, 40),   # Moderate load
                        'packetLossRate': random.uniform(0.005, 0.015),  # Low packet loss
                        'queueLength': random.randint(5, 20),
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.85, 0.95),
                        'status': 'active'
                    }}
                )
            logger.info(f"Applied PEAK_HOURS: {len(trap_nodes)} trap nodes affected, {len(normal_nodes)} normal")
            
        elif scenario_name == 'STORM_WEATHER':
            # TRAP NODES: Severely affected by weather
            for node in trap_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'signalStrengthDbm': random.uniform(-95, -75),   # Very weak signal
                        'packetLossRate': random.uniform(0.15, 0.3),     # High packet loss
                        'resourceUtilization': random.uniform(60, 80),   # High load due to retry
                        'queueLength': random.randint(40, 80),
                        'status': 'active'
                    }}
                )
            # NORMAL NODES: Healthy baseline
            for node in normal_nodes:
                set_healthy_baseline(node)
            logger.info(f"Applied STORM_WEATHER: {len(trap_nodes)} trap nodes affected")
            
        elif scenario_name == 'HEAVY_TRAFFIC':
            # TRAP NODES: Congested
            for node in trap_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.2, 0.4),
                        'resourceUtilization': random.uniform(80, 98),   # Almost full
                        'packetLossRate': random.uniform(0.05, 0.12),    # High loss
                        'queueLength': random.randint(80, 150),          # Very long queue
                        'status': 'active'
                    }}
                )
            # NORMAL NODES: Healthy baseline
            for node in normal_nodes:
                set_healthy_baseline(node)
            logger.info(f"Applied HEAVY_TRAFFIC: {len(trap_nodes)} trap nodes affected")
            
        elif scenario_name == 'REMOTE_AREA':
            # TRAP NODES: Poor connectivity
            for node in trap_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'signalStrengthDbm': random.uniform(-100, -85),  # Extremely weak signal
                        'packetLossRate': random.uniform(0.2, 0.35),     # Very high packet loss
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.1, 0.3),
                        'resourceUtilization': random.uniform(50, 70),
                        'queueLength': random.randint(30, 60),
                        'status': 'active'
                    }}
                )
            # NORMAL NODES: Healthy baseline
            for node in normal_nodes:
                set_healthy_baseline(node)
            logger.info(f"Applied REMOTE_AREA: {len(trap_nodes)} trap nodes affected")
            
        elif scenario_name == 'EQUIPMENT_AGING':
            # TRAP NODES: Degraded equipment
            for node in trap_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(70, 90),    # High load due to slow processing
                        'packetLossRate': random.uniform(0.08, 0.15),     # Hardware errors
                        'batteryChargePercent': random.uniform(20, 45),   # Low battery
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.4, 0.6),
                        'queueLength': random.randint(35, 70),
                        'signalStrengthDbm': random.uniform(-75, -60),
                        'status': 'active'
                    }}
                )
            # NORMAL NODES: Healthy baseline
            for node in normal_nodes:
                set_healthy_baseline(node)
            logger.info(f"Applied EQUIPMENT_AGING: {len(trap_nodes)} trap nodes affected")
            
        elif scenario_name == 'MAINTENANCE_MODE':
            # TRAP NODES: Offline for maintenance
            for node in trap_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'status': 'maintenance',
                        'resourceUtilization': 0,
                        'packetLossRate': 1.0,  # Cannot route through
                        'bandwidthMbps': 0,
                        'queueLength': 0
                    }}
                )
            # NORMAL NODES: Slightly elevated load due to rerouting
            for node in normal_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(25, 45),   # Moderate load increase
                        'packetLossRate': random.uniform(0.01, 0.02),
                        'queueLength': random.randint(10, 30),
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.8, 0.92),
                        'status': 'active'
                    }}
                )
            logger.info(f"Applied MAINTENANCE_MODE: {len(trap_nodes)} trap nodes offline")
            
        elif scenario_name == 'EMERGENCY_LOAD':
            # TRAP NODES: Completely overloaded
            for node in trap_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(92, 99),    # Near 100%
                        'packetLossRate': random.uniform(0.2, 0.4),       # Extreme packet loss
                        'queueLength': random.randint(150, 250),          # Extremely long queue
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.1, 0.25),
                        'signalStrengthDbm': random.uniform(-80, -65),
                        'batteryChargePercent': random.uniform(15, 35),
                        'status': 'active'
                    }}
                )
            # NORMAL NODES: Elevated but manageable
            for node in normal_nodes:
                nodes_collection.update_one(
                    {'_id': node['_id']},
                    {'$set': {
                        'resourceUtilization': random.uniform(35, 55),
                        'packetLossRate': random.uniform(0.015, 0.03),
                        'queueLength': random.randint(20, 40),
                        'bandwidthMbps': node.get('bandwidthMbps', 100) * random.uniform(0.7, 0.85),
                        'status': 'active'
                    }}
                )
            logger.info(f"Applied EMERGENCY_LOAD: {len(trap_nodes)} trap nodes severely affected")
        
        return True
        
    except Exception as e:
        logger.error(f"Error applying scenario {scenario_name}: {e}")
        return False

