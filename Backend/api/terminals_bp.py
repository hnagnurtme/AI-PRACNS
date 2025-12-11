"""
Terminals API Blueprint
Provides endpoints for managing user terminals
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from bson import ObjectId
from models.database import db
import random
import logging

logger = logging.getLogger(__name__)

terminals_bp = Blueprint('terminals', __name__, url_prefix='/api/v1/terminals')

def generate_terminal_id(index: int) -> str:
    """Generate a unique terminal ID"""
    timestamp = int(datetime.now().timestamp() * 1000)
    return f"TERM-{timestamp}-{index:04d}"

def generate_random_qos() -> dict:
    """Generate random QoS requirements"""
    service_types = ['VIDEO_STREAM', 'AUDIO_CALL', 'IMAGE_TRANSFER', 'TEXT_MESSAGE', 'FILE_TRANSFER']
    return {
        'maxLatencyMs': random.uniform(50, 500),
        'minBandwidthMbps': random.uniform(1, 100),
        'maxLossRate': random.uniform(0.001, 0.05),
        'priority': random.randint(1, 10),
        'serviceType': random.choice(service_types)
    }

def generate_random_position(bounds: dict) -> dict:
    """Generate random position within bounds"""
    lat = random.uniform(bounds.get('minLat', -90), bounds.get('maxLat', 90))
    lon = random.uniform(bounds.get('minLon', -180), bounds.get('maxLon', 180))
    # Ground terminals: 0-100m, Aircraft: 5000-12000m
    altitude = random.uniform(5000, 12000) if random.random() > 0.8 else random.uniform(0, 100)
    return {
        'latitude': lat,
        'longitude': lon,
        'altitude': altitude
    }

@terminals_bp.route('/generate', methods=['POST'])
def generate_terminals():
    """Generate user terminals"""
    try:
        data = request.get_json() or {}
        count = data.get('count', 10)
        bounds = data.get('bounds', {
            'minLat': -90,
            'maxLat': 90,
            'minLon': -180,
            'maxLon': 180
        })
        terminal_type = data.get('terminalType', random.choice(['MOBILE', 'FIXED', 'VEHICLE', 'AIRCRAFT']))
        
        collection = db.get_collection('terminals')
        new_terminals = []
        
        # Get all nodes for auto-connection
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        for i in range(count):
            terminal = {
                'id': generate_terminal_id(i),
                'terminalId': generate_terminal_id(i),
                'terminalName': f'Terminal {i + 1}',
                'terminalType': terminal_type,
                'position': generate_random_position(bounds),
                'status': 'idle',
                'connectedNodeId': None,
                'qosRequirements': generate_random_qos(),
                'metadata': {
                    'description': f'Generated terminal {i + 1}',
                    'region': 'auto-generated'
                },
                'lastUpdated': datetime.now().isoformat()
            }
            
            # Insert into database
            result = collection.insert_one(terminal)
            terminal['_id'] = str(result.inserted_id)
            
            # Tự động connect tới ground station tối ưu nhất
            try:
                from api.routing_bp import find_best_ground_station
                best_node = find_best_ground_station(terminal, nodes)
                
                if best_node:
                    # Auto-connect với 100% success rate (vì đã được tối ưu)
                    connection_metrics = {
                        'latencyMs': random.uniform(20, 80),  # Lower latency for optimal connection
                        'bandwidthMbps': random.uniform(20, 60),
                        'packetLossRate': random.uniform(0, 0.005),  # Lower packet loss
                        'signalStrength': random.uniform(-70, -50)  # Better signal
                    }
                    
                    collection.update_one(
                        {'terminalId': terminal['terminalId']},
                        {
                            '$set': {
                                'status': 'connected',
                                'connectedNodeId': best_node['nodeId'],
                                'connectionMetrics': connection_metrics,
                                'lastUpdated': datetime.now().isoformat()
                            }
                        }
                    )
                    
                    terminal['status'] = 'connected'
                    terminal['connectedNodeId'] = best_node['nodeId']
                    terminal['connectionMetrics'] = connection_metrics
                    logger.info(f"✅ Auto-connected terminal {terminal['terminalId']} to optimal ground station {best_node['nodeId']}")
                else:
                    logger.warning(f"⚠️ No optimal ground station found for terminal {terminal['terminalId']}")
            except Exception as e:
                logger.warning(f"Could not auto-connect terminal {terminal['terminalId']}: {e}")
            
            new_terminals.append(terminal)
        
        logger.info(f"Generated {count} terminals (auto-connected to optimal ground stations)")
        return jsonify(new_terminals), 200
        
    except Exception as e:
        logger.error(f"Error generating terminals: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@terminals_bp.route('', methods=['GET'])
def get_all_terminals():
    """Get all user terminals"""
    try:
        collection = db.get_collection('terminals')
        terminals = list(collection.find({}, {'_id': 0}))
        
        return jsonify(terminals), 200
        
    except Exception as e:
        logger.error(f"Error fetching terminals: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@terminals_bp.route('/<terminal_id>', methods=['GET'])
def get_terminal_by_id(terminal_id: str):
    """Get terminal by ID"""
    try:
        collection = db.get_collection('terminals')
        terminal = collection.find_one({'terminalId': terminal_id}, {'_id': 0})
        
        if not terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Terminal with ID {terminal_id} not found'
            }), 404
        
        return jsonify(terminal), 200
        
    except Exception as e:
        logger.error(f"Error fetching terminal: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@terminals_bp.route('/create', methods=['POST'])
def create_terminal():
    """Tạo terminal mới từ vị trí trên map (double-click)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        position = data.get('position')
        if not position or 'latitude' not in position or 'longitude' not in position:
            return jsonify({'error': 'Invalid position data'}), 400
        
        terminal_type = data.get('terminalType', 'MOBILE')
        terminal_name = data.get('terminalName', f'Terminal at ({position["latitude"]:.2f}, {position["longitude"]:.2f})')
        
        collection = db.get_collection('terminals')
        nodes_collection = db.get_collection('nodes')
        
        # Get all nodes for connection
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Tạo terminal mới
        terminal = {
            'id': generate_terminal_id(0),
            'terminalId': generate_terminal_id(0),
            'terminalName': terminal_name,
            'terminalType': terminal_type,
            'position': {
                'latitude': float(position['latitude']),
                'longitude': float(position['longitude']),
                'altitude': float(position.get('altitude', 0))
            },
            'status': 'idle',
            'connectedNodeId': None,
            'qosRequirements': generate_random_qos(),
            'metadata': {
                'description': f'Terminal created from map at ({position["latitude"]:.2f}, {position["longitude"]:.2f})',
                'region': 'user-created',
                'createdFrom': 'map-double-click'
            },
            'lastUpdated': datetime.now().isoformat()
        }
        
        # Insert vào database
        result = collection.insert_one(terminal)
        terminal['_id'] = str(result.inserted_id)
        
        # Tự động connect tới ground station GẦN NHẤT (theo Dijkstra - shortest path)
        # Không dùng find_best_ground_station vì nó tối ưu tài nguyên, ta cần gần nhất
        try:
            from api.routing_bp import calculate_distance
            
            # Tìm tất cả ground stations
            ground_stations = [
                n for n in nodes 
                if n.get('nodeType') == 'GROUND_STATION' 
                and n.get('isOperational', True)
                and n.get('position')
            ]
            
            if ground_stations:
                # Tìm ground station GẦN NHẤT (theo distance, không phải tối ưu tài nguyên)
                nearest_gs = None
                min_distance = float('inf')
                
                for gs in ground_stations:
                    distance = calculate_distance(terminal['position'], gs['position'])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_gs = gs
                
                if nearest_gs:
                    distance_km = min_distance / 1000.0
                    
                    # Tính connection metrics dựa trên distance
                    # Gần hơn = tốt hơn
                    if distance_km <= 50:
                        latency = random.uniform(10, 30)
                        bandwidth = random.uniform(50, 100)
                        signal = random.uniform(-60, -40)
                    elif distance_km <= 100:
                        latency = random.uniform(20, 50)
                        bandwidth = random.uniform(30, 70)
                        signal = random.uniform(-70, -50)
                    else:
                        latency = random.uniform(30, 80)
                        bandwidth = random.uniform(20, 50)
                        signal = random.uniform(-80, -60)
                    
                    connection_metrics = {
                        'latencyMs': round(latency, 2),
                        'bandwidthMbps': round(bandwidth, 2),
                        'packetLossRate': round(random.uniform(0, 0.01), 4),
                        'signalStrength': round(signal, 1),
                        'snrDb': round(random.uniform(10, 25), 1),
                        'jitterMs': round(random.uniform(1, 10), 2)
                    }
                    
                    # Update terminal với connection
                    collection.update_one(
                        {'_id': result.inserted_id},
                        {
                            '$set': {
                                'status': 'connected',
                                'connectedNodeId': nearest_gs['nodeId'],
                                'connectionMetrics': connection_metrics,
                                'lastUpdated': datetime.now().isoformat()
                            }
                        }
                    )
                    
                    terminal['status'] = 'connected'
                    terminal['connectedNodeId'] = nearest_gs['nodeId']
                    terminal['connectionMetrics'] = connection_metrics
                    
                    # Cập nhật resource utilization của GS (nhiều terminals → tăng utilization)
                    try:
                        from api.routing_bp import update_node_resource_utilization
                        update_node_resource_utilization(nearest_gs['nodeId'])
                    except Exception as e:
                        logger.warning(f"Error updating GS resource utilization: {e}")
                    
                    logger.info(
                        f"✅ Created terminal {terminal['terminalId']} and connected to nearest GS {nearest_gs['nodeId']} "
                        f"at {distance_km:.1f}km"
                    )
            else:
                logger.warning(f"⚠️ No ground stations available for terminal {terminal['terminalId']}")
        
        except Exception as e:
            logger.error(f"Error connecting terminal to ground station: {e}", exc_info=True)
        
        # Clean for JSON
        terminal.pop('_id', None)
        
        return jsonify({
            'success': True,
            'terminal': terminal,
            'message': f'Terminal created and connected to nearest ground station'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating terminal: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@terminals_bp.route('/<terminal_id>/connect', methods=['POST'])
def connect_terminal(terminal_id: str):
    """Connect terminal to a Ground Station (auto-selects best station if nodeId not provided)"""
    try:
        data = request.get_json() or {}
        node_id = data.get('nodeId')
        
        collection = db.get_collection('terminals')
        terminal = collection.find_one({'terminalId': terminal_id})
        
        if not terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Terminal with ID {terminal_id} not found'
            }), 404
        
        # If nodeId not provided, auto-select best Ground Station
        if not node_id:
            from api.routing_bp import find_best_ground_station
            nodes_collection = db.get_collection('nodes')
            nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
            
            best_node = find_best_ground_station(terminal, nodes)
            if not best_node:
                return jsonify({
                    'status': 404,
                    'error': 'Not found',
                    'message': 'No available Ground Station found'
                }), 404
            
            node_id = best_node['nodeId']
            logger.info(f"Auto-selected Ground Station {node_id} for terminal {terminal_id}")
        else:
            # Validate that the provided node is a Ground Station
            nodes_collection = db.get_collection('nodes')
            node = nodes_collection.find_one({'nodeId': node_id}, {'_id': 0})
            
            if not node:
                return jsonify({
                    'status': 404,
                    'error': 'Not found',
                    'message': f'Node with ID {node_id} not found'
                }), 404
            
            if node.get('nodeType') != 'GROUND_STATION':
                return jsonify({
                    'status': 400,
                    'error': 'Bad request',
                    'message': f'Terminal can only connect to GROUND_STATION. Node {node_id} is {node.get("nodeType")}'
                }), 400
            
            if not node.get('isOperational', True):
                return jsonify({
                    'status': 400,
                    'error': 'Bad request',
                    'message': f'Ground Station {node_id} is not operational'
                }), 400
        
        # Simulate connection (90% success rate)
        success = random.random() > 0.1
        
        if success:
            # Update terminal
            connection_metrics = {
                'latencyMs': random.uniform(20, 120),
                'bandwidthMbps': random.uniform(10, 60),
                'packetLossRate': random.uniform(0, 0.01),
                'signalStrength': random.uniform(-80, -60)
            }
            
            collection.update_one(
                {'terminalId': terminal_id},
                {
                    '$set': {
                        'status': 'connected',
                        'connectedNodeId': node_id,
                        'connectionMetrics': connection_metrics,
                        'lastUpdated': datetime.now().isoformat()
                    }
                }
            )
            
            # Cập nhật resource utilization của GS (nhiều terminals → tăng utilization)
            try:
                from api.routing_bp import update_node_resource_utilization
                update_node_resource_utilization(node_id)
            except Exception as e:
                logger.warning(f"Error updating GS resource utilization: {e}")
            
            result = {
                'terminalId': terminal_id,
                'nodeId': node_id,
                'success': True,
                'latencyMs': connection_metrics['latencyMs'],
                'bandwidthMbps': connection_metrics['bandwidthMbps'],
                'packetLossRate': connection_metrics['packetLossRate'],
                'message': 'Connection established successfully',
                'timestamp': datetime.now().isoformat()
            }
        else:
            result = {
                'terminalId': terminal_id,
                'nodeId': node_id,
                'success': False,
                'message': 'Connection failed: Node out of range or unavailable',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Terminal {terminal_id} connection attempt: {'success' if success else 'failed'}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error connecting terminal: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@terminals_bp.route('/<terminal_id>/disconnect', methods=['POST'])
def disconnect_terminal(terminal_id: str):
    """Disconnect terminal from node"""
    try:
        collection = db.get_collection('terminals')
        terminal = collection.find_one({'terminalId': terminal_id})
        
        if not terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Terminal with ID {terminal_id} not found'
            }), 404
        
        # Lưu node_id trước khi disconnect để cập nhật resource
        old_node_id = terminal.get('connectedNodeId')
        
        # Update terminal
        collection.update_one(
            {'terminalId': terminal_id},
            {
                '$set': {
                    'status': 'idle',
                    'connectedNodeId': None,
                    'connectionMetrics': None,
                    'lastUpdated': datetime.now().isoformat()
                }
            }
        )
        
        # Cập nhật resource utilization của GS khi terminal disconnect
        if old_node_id:
            try:
                from api.routing_bp import update_node_resource_utilization
                update_node_resource_utilization(old_node_id)
                logger.info(f"Updated GS {old_node_id} resource after terminal {terminal_id} disconnected")
            except Exception as e:
                logger.warning(f"Error updating GS resource utilization after disconnect: {e}")
        
        # Get updated terminal
        updated_terminal = collection.find_one({'terminalId': terminal_id}, {'_id': 0})
        
        logger.info(f"Terminal {terminal_id} disconnected")
        return jsonify(updated_terminal), 200
        
    except Exception as e:
        logger.error(f"Error disconnecting terminal: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@terminals_bp.route('/<terminal_id>/connection-result', methods=['GET'])
def get_connection_result(terminal_id: str):
    """Get terminal connection result"""
    try:
        collection = db.get_collection('terminals')
        terminal = collection.find_one({'terminalId': terminal_id}, {'_id': 0})
        
        if not terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Terminal with ID {terminal_id} not found'
            }), 404
        
        if not terminal.get('connectedNodeId'):
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': 'No connection result available'
            }), 404
        
        result = {
            'terminalId': terminal_id,
            'nodeId': terminal['connectedNodeId'],
            'success': terminal.get('status') == 'connected',
            'latencyMs': terminal.get('connectionMetrics', {}).get('latencyMs'),
            'bandwidthMbps': terminal.get('connectionMetrics', {}).get('bandwidthMbps'),
            'packetLossRate': terminal.get('connectionMetrics', {}).get('packetLossRate'),
            'timestamp': terminal.get('lastUpdated')
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error fetching connection result: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@terminals_bp.route('/<terminal_id>', methods=['DELETE'])
def delete_terminal(terminal_id: str):
    """Delete terminal"""
    try:
        collection = db.get_collection('terminals')
        
        # Get terminal before deleting to update connected GS resource
        terminal = collection.find_one({'terminalId': terminal_id})
        if not terminal:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Terminal with ID {terminal_id} not found'
            }), 404
        
        connected_node_id = terminal.get('connectedNodeId')
        
        # Delete terminal
        result = collection.delete_one({'terminalId': terminal_id})
        
        # Update ground station resource utilization after deletion
        if connected_node_id:
            try:
                from api.routing_bp import update_node_resource_utilization
                update_node_resource_utilization(connected_node_id)
                logger.info(f"✅ Updated GS {connected_node_id} resource after deleting terminal {terminal_id}")
            except Exception as e:
                logger.warning(f"Error updating GS resource after terminal deletion: {e}")
        
        logger.info(f"Terminal {terminal_id} deleted")
        return jsonify({'message': 'Terminal deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error deleting terminal: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@terminals_bp.route('', methods=['DELETE'])
def delete_all_terminals():
    """Delete all terminals"""
    try:
        collection = db.get_collection('terminals')
        
        # Get all connected ground stations before deletion
        terminals = list(collection.find({'connectedNodeId': {'$exists': True, '$ne': None}}))
        connected_gs_ids = set(t['connectedNodeId'] for t in terminals if t.get('connectedNodeId'))
        
        # Delete all terminals
        result = collection.delete_many({})
        
        # Update all affected ground stations resource utilization
        if connected_gs_ids:
            try:
                from api.routing_bp import update_node_resource_utilization
                for gs_id in connected_gs_ids:
                    update_node_resource_utilization(gs_id)
                logger.info(f"✅ Updated {len(connected_gs_ids)} ground stations resource after clearing terminals")
            except Exception as e:
                logger.warning(f"Error updating GS resources after clearing terminals: {e}")
        
        logger.info(f"Deleted {result.deleted_count} terminals")
        return jsonify({
            'message': f'Deleted {result.deleted_count} terminals successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting terminals: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

