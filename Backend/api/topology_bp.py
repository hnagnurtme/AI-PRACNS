"""
Network Topology API Blueprint
Provides endpoints for network topology visualization and statistics
"""
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from models.database import db
import logging
import math

logger = logging.getLogger(__name__)

topology_bp = Blueprint('topology', __name__, url_prefix='/api/v1')

def calculate_distance(pos1: dict, pos2: dict) -> float:
    """Calculate distance between two positions in meters"""
    from math import radians, cos, sin, asin, sqrt
    
    lat1 = radians(pos1.get('latitude', 0))
    lon1 = radians(pos1.get('longitude', 0))
    lat2 = radians(pos2.get('latitude', 0))
    lon2 = radians(pos2.get('longitude', 0))
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in meters
    r = 6371000
    
    return c * r

def calculate_connection_metrics(node1: dict, node2: dict) -> dict:
    """Calculate connection metrics between two nodes"""
    distance = calculate_distance(node1.get('position', {}), node2.get('position', {}))
    distance_km = distance / 1000
    
    # Estimate latency based on distance (speed of light)
    speed_of_light = 299792458  # m/s
    propagation_delay = (distance / speed_of_light) * 1000  # ms
    processing_delay = node1.get('nodeProcessingDelayMs', 5) + node2.get('nodeProcessingDelayMs', 5)
    latency = propagation_delay + processing_delay
    
    # Estimate bandwidth based on node communication capabilities
    max_range1 = node1.get('communication', {}).get('maxRangeKm', 2000) * 1000
    max_range2 = node2.get('communication', {}).get('maxRangeKm', 2000) * 1000
    max_range = min(max_range1, max_range2)
    
    # Check if nodes are within range
    if distance > max_range * 1.5:  # Allow some margin
        return {
            'status': 'inactive',
            'latency': 0,
            'bandwidth': 0,
            'distance': distance_km
        }
    
    # Estimate bandwidth (decreases with distance)
    max_bandwidth = min(
        node1.get('communication', {}).get('maxBandwidthMbps', 100),
        node2.get('communication', {}).get('maxBandwidthMbps', 100)
    )
    
    # Bandwidth degradation with distance
    distance_factor = 1 - (distance / max_range) * 0.3
    bandwidth = max_bandwidth * max(0.1, distance_factor)
    
    # Estimate packet loss
    packet_loss = (node1.get('packetLossRate', 0) + node2.get('packetLossRate', 0)) / 2
    packet_loss += (distance / max_range) * 0.01  # Additional loss due to distance
    
    # Signal strength (decreases with distance)
    signal_strength = -60 - (distance / 1000) * 0.1  # dBm
    
    status = 'active'
    if packet_loss > 0.05 or latency > 500:
        status = 'degraded'
    if packet_loss > 0.1 or latency > 1000:
        status = 'inactive'
    
    return {
        'status': status,
        'latency': round(latency, 2),
        'bandwidth': round(bandwidth, 2),
        'distance': round(distance_km, 2),
        'packetLossRate': round(min(packet_loss, 1.0), 4),
        'signalStrength': round(signal_strength, 1)
    }

def build_network_connections(nodes: list) -> list:
    """Build connection matrix for all nodes"""
    connections = []
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i >= j:  # Avoid duplicates and self-connections
                continue
            
            metrics = calculate_connection_metrics(node1, node2)
            if metrics['status'] != 'inactive':
                connections.append({
                    'fromNodeId': node1['nodeId'],
                    'toNodeId': node2['nodeId'],
                    'latency': metrics['latency'],
                    'bandwidth': metrics['bandwidth'],
                    'status': metrics['status'],
                    'distance': metrics.get('distance', 0),
                    'packetLossRate': metrics.get('packetLossRate', 0),
                    'signalStrength': metrics.get('signalStrength', -70),
                    'lastUpdated': datetime.now().isoformat()
                })
    
    return connections

def calculate_statistics(nodes: list, terminals: list, connections: list) -> dict:
    """Calculate network statistics"""
    active_nodes = [n for n in nodes if n.get('isOperational', True)]
    active_connections = [c for c in connections if c.get('status') == 'active']
    
    total_bandwidth = sum(c.get('bandwidth', 0) for c in connections)
    avg_latency = sum(c.get('latency', 0) for c in connections) / len(connections) if connections else 0
    avg_packet_loss = sum(c.get('packetLossRate', 0) for c in connections) / len(connections) if connections else 0
    
    connected_terminals = [t for t in terminals if t.get('status') == 'connected']
    
    # Calculate utilization rate
    total_capacity = sum(n.get('communication', {}).get('maxBandwidthMbps', 100) for n in nodes)
    utilization_rate = (total_bandwidth / total_capacity * 100) if total_capacity > 0 else 0
    
    # Determine network health
    if avg_packet_loss < 0.01 and avg_latency < 100 and len(active_nodes) == len(nodes):
        health = 'healthy'
    elif avg_packet_loss < 0.05 and avg_latency < 300:
        health = 'degraded'
    else:
        health = 'critical'
    
    return {
        'totalNodes': len(nodes),
        'activeNodes': len(active_nodes),
        'totalTerminals': len(terminals),
        'connectedTerminals': len(connected_terminals),
        'activeConnections': len(active_connections),
        'totalConnections': len(connections),
        'totalBandwidth': round(total_bandwidth, 2),
        'averageLatency': round(avg_latency, 2),
        'averagePacketLoss': round(avg_packet_loss, 4),
        'networkHealth': health,
        'utilizationRate': round(utilization_rate, 2)
    }

@topology_bp.route('/topology', methods=['GET'])
def get_topology():
    """Get complete network topology"""
    try:
        # Get all nodes
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Get all terminals
        terminals_collection = db.get_collection('terminals')
        terminals = list(terminals_collection.find({}, {'_id': 0}))
        
        # Build connections
        connections = build_network_connections(nodes)
        
        # Calculate statistics
        statistics = calculate_statistics(nodes, terminals, connections)
        
        topology = {
            'networkId': 'main-network',
            'networkName': 'SAGIN Network',
            'description': 'Main SAGIN network topology',
            'nodes': nodes,
            'terminals': terminals,
            'connections': connections,
            'statistics': statistics,
            'createdAt': datetime.now().isoformat(),
            'updatedAt': datetime.now().isoformat()
        }
        
        logger.info(f"Topology requested: {len(nodes)} nodes, {len(terminals)} terminals, {len(connections)} connections")
        return jsonify(topology), 200
        
    except Exception as e:
        logger.error(f"Error getting topology: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@topology_bp.route('/topology/statistics', methods=['GET'])
def get_topology_statistics():
    """Get network statistics only"""
    try:
        # Get all nodes
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Get all terminals
        terminals_collection = db.get_collection('terminals')
        terminals = list(terminals_collection.find({}, {'_id': 0}))
        
        # Build connections
        connections = build_network_connections(nodes)
        
        # Calculate statistics
        statistics = calculate_statistics(nodes, terminals, connections)
        
        logger.info(f"Statistics requested: {statistics['networkHealth']} health")
        return jsonify(statistics), 200
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@topology_bp.route('/topology/connections', methods=['GET'])
def get_connections():
    """Get network connections only"""
    try:
        # Get all nodes
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Build connections
        connections = build_network_connections(nodes)
        
        logger.info(f"Connections requested: {len(connections)} connections")
        return jsonify(connections), 200
        
    except Exception as e:
        logger.error(f"Error getting connections: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@topology_bp.route('/topology/nodes/<node_id>/analysis', methods=['GET'])
def get_node_analysis(node_id: str):
    """Get detailed analysis for a specific node"""
    try:
        nodes_collection = db.get_collection('nodes')
        terminals_collection = db.get_collection('terminals')
        
        # Get target node
        target_node = nodes_collection.find_one({'nodeId': node_id}, {'_id': 0})
        if not target_node:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Node {node_id} not found'
            }), 404
        
        # Get all nodes
        all_nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        # Calculate link metrics with all neighbors
        link_metrics = []
        for node in all_nodes:
            if node['nodeId'] == node_id:
                continue
            
            metrics = calculate_connection_metrics(target_node, node)
            if metrics['status'] != 'inactive':
                # Calculate quality score (0-100)
                latency_score = max(0, 100 - (metrics['latency'] / 10))  # Lower latency = higher score
                bandwidth_score = min(100, (metrics['bandwidth'] / 100) * 100)  # Higher bandwidth = higher score
                loss_score = max(0, 100 - (metrics['packetLossRate'] * 1000))  # Lower loss = higher score
                signal_score = max(0, 100 + metrics['signalStrength'])  # Higher signal = higher score
                
                total_score = (latency_score * 0.3 + bandwidth_score * 0.3 + loss_score * 0.2 + signal_score * 0.2)
                
                quality = 'excellent' if total_score >= 80 else 'good' if total_score >= 60 else 'fair' if total_score >= 40 else 'poor'
                
                link_metrics.append({
                    'nodeId': node['nodeId'],
                    'nodeName': node.get('nodeName', node['nodeId']),
                    'latency': metrics['latency'],
                    'bandwidth': metrics['bandwidth'],
                    'packetLoss': metrics.get('packetLossRate', 0),
                    'signalStrength': metrics.get('signalStrength', -70),
                    'distance': metrics.get('distance', 0),
                    'quality': quality,
                    'score': round(total_score, 2)
                })
        
        # Sort by score (best first), then by distance (closest first)
        link_metrics.sort(key=lambda x: (x['score'], -x['distance']), reverse=True)
        best_links = link_metrics[:10]  # Top 10 best links (highest score, closest distance)
        
        # Find upcoming satellites (for any node, find satellites that will be in range soon)
        upcoming_satellites = []
        target_pos = target_node.get('position', {})
        target_max_range = target_node.get('communication', {}).get('maxRangeKm', 2000) * 1000  # meters
        
        for node in all_nodes:
            if 'SATELLITE' in node.get('nodeType', ''):
                node_pos = node.get('position', {})
                current_distance = calculate_distance(target_pos, node_pos)
                
                # Check if satellite is currently in range
                if current_distance <= target_max_range * 1.1:  # Currently in range (with margin)
                    continue
                
                # Calculate if satellite will come into range
                # Simplified orbital prediction: assume LEO satellites move at ~7.8 km/s
                # For more accurate prediction, would need orbital mechanics calculations
                satellite_speed = 7800  # m/s for LEO
                
                # Estimate time to reach range based on current distance and speed
                # Assume satellite is moving towards the target (simplified)
                distance_to_range = current_distance - (target_max_range * 1.1)
                
                if distance_to_range > 0:
                    # Estimate time based on distance and speed
                    # Add some randomness/variation (satellites don't move directly towards target)
                    estimated_seconds = (distance_to_range / satellite_speed) * 1.5  # Factor for indirect path
                    
                    # Cap at reasonable time (e.g., 2 hours max)
                    estimated_seconds = min(estimated_seconds, 7200)  # 2 hours max
                    
                    if estimated_seconds > 60:  # Only show if more than 1 minute away
                        estimated_arrival = datetime.now() + timedelta(seconds=int(estimated_seconds))
                        
                        # Estimate metrics when in range
                        # Latency based on distance when in range (will be at max_range)
                        estimated_latency = round((target_max_range / 299792458) * 1000 + 10, 2)  # Speed of light + processing
                        
                        # Bandwidth based on node capabilities
                        estimated_bandwidth = min(
                            target_node.get('communication', {}).get('maxBandwidthMbps', 100),
                            node.get('communication', {}).get('maxBandwidthMbps', 100)
                        ) * 0.8  # 80% of max when in optimal range
                        
                        upcoming_satellites.append({
                            'nodeId': node['nodeId'],
                            'nodeName': node.get('nodeName', node['nodeId']),
                            'nodeType': node.get('nodeType', ''),
                            'currentPosition': node_pos,
                            'currentDistance': round(current_distance / 1000, 2),  # km
                            'estimatedArrivalTime': estimated_arrival.isoformat(),
                            'estimatedArrivalIn': int(estimated_seconds),
                            'willBeInRange': True,
                            'estimatedLatency': round(estimated_latency, 2),
                            'estimatedBandwidth': round(estimated_bandwidth, 2)
                        })
        
        # Sort upcoming satellites by arrival time (soonest first)
        upcoming_satellites.sort(key=lambda x: x['estimatedArrivalIn'])
        
        # Find nodes about to degrade
        degrading_nodes = []
        for node in all_nodes:
            if node['nodeId'] == node_id:
                continue
            
            degradation_reasons = []
            severity = 'minor'
            
            # Check various degradation factors
            if node.get('batteryChargePercent', 100) < 20:
                degradation_reasons.append('Battery low')
                severity = 'critical' if node.get('batteryChargePercent', 100) < 10 else 'warning'
            
            if node.get('resourceUtilization', 0) > 85:
                degradation_reasons.append('High resource utilization')
                severity = 'critical' if node.get('resourceUtilization', 0) > 95 else 'warning'
            
            if node.get('packetLossRate', 0) > 0.05:
                degradation_reasons.append('High packet loss')
                severity = 'critical' if node.get('packetLossRate', 0) > 0.1 else 'warning'
            
            if node.get('nodeProcessingDelayMs', 0) > 300:
                degradation_reasons.append('High latency')
                severity = 'critical' if node.get('nodeProcessingDelayMs', 0) > 500 else 'warning'
            
            queue_ratio = (node.get('currentPacketCount', 0) / node.get('packetBufferCapacity', 1)) * 100
            if queue_ratio > 80:
                degradation_reasons.append('Queue nearly full')
                severity = 'critical' if queue_ratio > 90 else 'warning'
            
            if degradation_reasons:
                # Estimate degradation time (simplified prediction)
                predicted_time = datetime.now() + timedelta(minutes=15)  # Assume 15 minutes
                degrading_nodes.append({
                    'nodeId': node['nodeId'],
                    'nodeName': node.get('nodeName', node['nodeId']),
                    'nodeType': node.get('nodeType', ''),
                    'currentMetrics': {
                        'latency': node.get('nodeProcessingDelayMs', 0),
                        'packetLoss': node.get('packetLossRate', 0),
                        'utilization': node.get('resourceUtilization', 0),
                        'queueRatio': queue_ratio,
                        'battery': node.get('batteryChargePercent', 100)
                    },
                    'predictedDegradationTime': predicted_time.isoformat(),
                    'predictedDegradationIn': 900,  # 15 minutes in seconds
                    'degradationReason': degradation_reasons,
                    'severity': severity
                })
        
        # Sort by severity and time
        degrading_nodes.sort(key=lambda x: (
            0 if x['severity'] == 'critical' else 1 if x['severity'] == 'warning' else 2,
            x['predictedDegradationIn']
        ))
        
        analysis = {
            'nodeId': target_node['nodeId'],
            'nodeName': target_node.get('nodeName', target_node['nodeId']),
            'bestLinks': best_links,
            'upcomingSatellites': upcoming_satellites[:5],  # Top 5 upcoming
            'degradingNodes': degrading_nodes[:5],  # Top 5 degrading
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Node analysis requested for {node_id}: {len(best_links)} links, {len(upcoming_satellites)} upcoming, {len(degrading_nodes)} degrading")
        return jsonify(analysis), 200
        
    except Exception as e:
        logger.error(f"Error getting node analysis: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500
