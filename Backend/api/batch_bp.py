"""
Batch Packet API Blueprint
Provides endpoints for generating and streaming batch packets via WebSocket
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from models.database import db
from api.routing_bp import calculate_path, calculate_path_dijkstra, calculate_path_rl, find_nearest_node
from services.network_analyzer import NetworkAnalyzer
import logging
import random
import threading
import time
import json

logger = logging.getLogger(__name__)

batch_bp = Blueprint('batch', __name__, url_prefix='/api/v1/batch')

# WebSocket message queue for batch packets
_batch_message_queue = []
_websocket_clients = set()
_batch_generation_active = False
_batch_generation_thread = None

def create_packet_from_path(path, source_terminal, dest_terminal, algorithm, service_qos=None):
    """Create a Packet object from a routing path"""
    import base64
    
    # Generate hop records from path
    hop_records = []
    total_latency = 0
    total_distance = 0
    
    for i in range(len(path['path']) - 1):
        from_seg = path['path'][i]
        to_seg = path['path'][i + 1]
        
        # Calculate distance
        if from_seg['position'] and to_seg['position']:
            from api.routing_bp import calculate_distance
            distance_km = calculate_distance(from_seg['position'], to_seg['position']) / 1000
        else:
            distance_km = 0
        
        # Estimate latency per hop
        latency_ms = path['estimatedLatency'] / max(path['hops'], 1) if path['hops'] > 0 else 0
        
        total_latency += latency_ms
        total_distance += distance_km
        
        # Get node resource info if available
        nodes_collection = db.get_collection('nodes')
        from_node = nodes_collection.find_one({'nodeId': from_seg['id']}, {'_id': 0}) if from_seg['type'] == 'node' else None
        
        hop_record = {
            'fromNodeId': from_seg['id'],
            'toNodeId': to_seg['id'],
            'latencyMs': latency_ms,
            'timestampMs': int(datetime.now().timestamp() * 1000) + i * 10,
            'distanceKm': distance_km,
            'fromNodePosition': from_seg['position'],
            'toNodePosition': to_seg['position'],
            'fromNodeBufferState': {
                'queueSize': from_node.get('currentPacketCount', 0) if from_node else 0,
                'bandwidthUtilization': from_node.get('resourceUtilization', 0) / 100 if from_node else 0
            },
            'routingDecisionInfo': {
                'algorithm': 'ReinforcementLearning' if algorithm == 'rl' else 'Dijkstra'
            },
            'nodeLoadPercent': from_node.get('resourceUtilization', 0) if from_node else 0
        }
        hop_records.append(hop_record)
    
    # Create packet
    packet = {
        'packetId': f"PKT-{int(datetime.now().timestamp() * 1000)}-{random.randint(1000, 9999)}",
        'sourceUserId': source_terminal['terminalId'],
        'destinationUserId': dest_terminal['terminalId'],
        'stationSource': source_terminal['terminalId'],
        'stationDest': dest_terminal['terminalId'],
        'type': 'BATCH',
        'timeSentFromSourceMs': int(datetime.now().timestamp() * 1000),
        'payloadDataBase64': base64.b64encode(b'batch_packet_data').decode('utf-8'),
        'payloadSizeByte': 1024,
        'serviceQoS': {
            'serviceType': service_qos.get('serviceType', 'TEXT_MESSAGE') if service_qos else 'TEXT_MESSAGE',
            'defaultPriority': service_qos.get('priority', 5) if service_qos else 5,
            'maxLatencyMs': service_qos.get('maxLatencyMs', 100) if service_qos else 100,
            'maxJitterMs': 10,
            'minBandwidthMbps': service_qos.get('minBandwidthMbps', 1) if service_qos else 1,
            'maxLossRate': service_qos.get('maxLossRate', 0.01) if service_qos else 0.01
        },
        'currentHoldingNodeId': path['path'][-1]['id'] if path['path'] else '',
        'nextHopNodeId': '',
        'pathHistory': [seg['id'] for seg in path['path']],
        'hopRecords': hop_records,
        'accumulatedDelayMs': total_latency,
        'priorityLevel': service_qos.get('priority', 5) if service_qos else 5,
        'maxAcceptableLatencyMs': service_qos.get('maxLatencyMs', 100) if service_qos else 100,
        'maxAcceptableLossRate': service_qos.get('maxLossRate', 0.01) if service_qos else 0.01,
        'dropped': False,
        'analysisData': {
            'avgLatency': total_latency / max(len(hop_records), 1),
            'avgDistanceKm': total_distance / max(len(hop_records), 1),
            'routeSuccessRate': 1.0,
            'totalDistanceKm': total_distance,
            'totalLatencyMs': total_latency
        },
        'isUseRL': algorithm == 'rl',
        'TTL': 100
    }
    
    return packet

@batch_bp.route('/generate', methods=['POST'])
def generate_batch():
    """Generate a batch of packet pairs (Dijkstra vs RL)"""
    try:
        data = request.get_json() or {}
        pair_count = data.get('pairCount', 10)
        scenario = data.get('scenario', 'NORMAL')
        save_to_db = data.get('saveToDb', True)  # Option to save batch results
        
        # Get terminals
        terminals_collection = db.get_collection('terminals')
        terminals = list(terminals_collection.find({}, {'_id': 0}))
        
        if len(terminals) < 2:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'Need at least 2 terminals to generate batch'
            }), 400
        
        # Get nodes
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        if len(nodes) == 0:
            return jsonify({
                'status': 400,
                'error': 'Bad request',
                'message': 'No operational nodes available'
            }), 400
        
        # Generate batch
        batch_id = f"BATCH-{int(datetime.now().timestamp() * 1000)}"
        packets = []
        
        for i in range(pair_count):
            # Random source and destination
            source_terminal = random.choice(terminals)
            dest_terminal = random.choice(terminals)
            while dest_terminal['terminalId'] == source_terminal['terminalId']:
                dest_terminal = random.choice(terminals)
            
            # Random QoS
            service_qos = {
                'maxLatencyMs': random.randint(50, 200),
                'minBandwidthMbps': random.uniform(0.5, 5.0),
                'maxLossRate': random.uniform(0.001, 0.05),
                'priority': random.randint(1, 10),
                'serviceType': random.choice(['TEXT_MESSAGE', 'IMAGE_TRANSFER', 'VIDEO_STREAM'])
            }
            
            # Calculate paths for both algorithms (pass QoS to prevent drops)
            path_dijkstra = calculate_path_dijkstra(source_terminal, dest_terminal, nodes)
            
            # Try RL, skip pair nếu RL không available (chưa có model)
            try:
                path_rl = calculate_path_rl(source_terminal, dest_terminal, nodes, service_qos=service_qos)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"⚠️ RL not available for batch {i+1}: {e}")
                # Skip pair nếu RL không hoạt động
                continue
            
            # Create packets
            dijkstra_packet = create_packet_from_path(path_dijkstra, source_terminal, dest_terminal, 'dijkstra', service_qos)
            rl_packet = create_packet_from_path(path_rl, source_terminal, dest_terminal, 'rl', service_qos)
            
            packets.append({
                'dijkstraPacket': dijkstra_packet,
                'rlPacket': rl_packet
            })
        
        batch = {
            'batchId': batch_id,
            'totalPairPackets': pair_count,
            'packets': packets,
            'scenario': scenario
        }
        
        # Save batch results to database for analysis and history
        if save_to_db:
            try:
                batch_results_collection = db.get_collection('batch_results')
                
                # Calculate statistics for the batch
                dijkstra_latencies = [p['dijkstraPacket']['analysisData']['totalLatencyMs'] for p in packets if not p['dijkstraPacket'].get('dropped', False)]
                rl_latencies = [p['rlPacket']['analysisData']['totalLatencyMs'] for p in packets if not p['rlPacket'].get('dropped', False)]
                
                dijkstra_distances = [p['dijkstraPacket']['analysisData']['totalDistanceKm'] for p in packets if not p['dijkstraPacket'].get('dropped', False)]
                rl_distances = [p['rlPacket']['analysisData']['totalDistanceKm'] for p in packets if not p['rlPacket'].get('dropped', False)]
                
                dijkstra_hops = [len(p['dijkstraPacket']['hopRecords']) for p in packets if not p['dijkstraPacket'].get('dropped', False)]
                rl_hops = [len(p['rlPacket']['hopRecords']) for p in packets if not p['rlPacket'].get('dropped', False)]
                
                batch_record = {
                    'batchId': batch_id,
                    'timestamp': datetime.now(),
                    'scenario': scenario,
                    'totalPairs': pair_count,
                    'statistics': {
                        'dijkstra': {
                            'avgLatencyMs': sum(dijkstra_latencies) / len(dijkstra_latencies) if dijkstra_latencies else 0,
                            'avgDistanceKm': sum(dijkstra_distances) / len(dijkstra_distances) if dijkstra_distances else 0,
                            'avgHops': sum(dijkstra_hops) / len(dijkstra_hops) if dijkstra_hops else 0,
                            'droppedCount': sum(1 for p in packets if p['dijkstraPacket'].get('dropped', False)),
                            'successRate': len(dijkstra_latencies) / pair_count if pair_count > 0 else 0
                        },
                        'rl': {
                            'avgLatencyMs': sum(rl_latencies) / len(rl_latencies) if rl_latencies else 0,
                            'avgDistanceKm': sum(rl_distances) / len(rl_distances) if rl_distances else 0,
                            'avgHops': sum(rl_hops) / len(rl_hops) if rl_hops else 0,
                            'droppedCount': sum(1 for p in packets if p['rlPacket'].get('dropped', False)),
                            'successRate': len(rl_latencies) / pair_count if pair_count > 0 else 0
                        },
                        'comparison': {
                            'latencyImprovement': ((sum(dijkstra_latencies) - sum(rl_latencies)) / sum(dijkstra_latencies) * 100) if dijkstra_latencies and rl_latencies else 0,
                            'distanceImprovement': ((sum(dijkstra_distances) - sum(rl_distances)) / sum(dijkstra_distances) * 100) if dijkstra_distances and rl_distances else 0,
                            'rlWins': sum(1 for i in range(min(len(dijkstra_latencies), len(rl_latencies))) if rl_latencies[i] < dijkstra_latencies[i]),
                            'dijkstraWins': sum(1 for i in range(min(len(dijkstra_latencies), len(rl_latencies))) if dijkstra_latencies[i] < rl_latencies[i])
                        }
                    },
                    'networkState': {
                        'totalNodes': len(nodes),
                        'totalTerminals': len(terminals),
                        'operationalNodes': len([n for n in nodes if n.get('isOperational', True)])
                    },
                    # Store full packets for detailed analysis
                    'packets': packets[:10]  # Store first 10 pairs as samples
                }
                
                batch_results_collection.insert_one(batch_record)
                logger.info(f"Saved batch {batch_id} results to database")
            except Exception as save_error:
                logger.warning(f"Failed to save batch results to database: {save_error}")
        
        # Add to message queue for WebSocket
        _batch_message_queue.append(batch)
        
        logger.info(f"Generated batch {batch_id} with {pair_count} pairs")
        return jsonify(batch), 200
        
    except Exception as e:
        logger.error(f"Error generating batch: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@batch_bp.route('/suggest-batch-params', methods=['GET'])
def suggest_batch_params():
    """Suggest batch parameters based on network analysis"""
    try:
        # Get terminals and nodes
        terminals_collection = db.get_collection('terminals')
        terminals = list(terminals_collection.find({}, {'_id': 0}))
        
        nodes_collection = db.get_collection('nodes')
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        
        if len(terminals) < 2:
            return jsonify({
                'suggestedPairCount': 5,
                'suggestedPairs': [],
                'recommendations': [{
                    'type': 'warning',
                    'message': 'Insufficient terminals. Need at least 2 terminals.',
                    'suggestions': ['Add more terminals to the network']
                }]
            }), 200
        
        # Initialize analyzer
        from config import Config
        config = Config.get_yaml_config()
        analyzer = NetworkAnalyzer(config)
        
        # Analyze overloaded nodes
        overload_analysis = analyzer.analyze_overloaded_nodes(
            nodes=nodes,
            terminals=terminals,
            db=db,
            threshold_utilization=0.7,
            threshold_packet_loss=0.03,
            focus_ground_stations=True
        )
        
        # Suggest pair count based on network size and health
        total_terminals = len(terminals)
        overload_percentage = overload_analysis['summary']['overload_percentage']
        
        # Base pair count: 10-20% of possible pairs
        max_possible_pairs = total_terminals * (total_terminals - 1) // 2
        base_pair_count = max(5, min(50, int(max_possible_pairs * 0.15)))
        
        # Adjust based on network health
        if overload_percentage > 50:
            suggested_pair_count = max(5, base_pair_count // 2)  # Reduce load
            health_note = 'Network is heavily loaded. Reduced batch size recommended.'
        elif overload_percentage > 30:
            suggested_pair_count = int(base_pair_count * 0.8)
            health_note = 'Network has moderate load. Slightly reduced batch size.'
        else:
            suggested_pair_count = base_pair_count
            health_note = 'Network is healthy. Normal batch size recommended.'
        
        # Suggest interesting terminal pairs
        suggested_pairs = []
        
        # 1. Long-distance pairs (global coverage test)
        terminals_with_pos = [t for t in terminals if t.get('position')]
        if len(terminals_with_pos) >= 2:
            max_distance = 0
            best_long_pair = None
            for i, t1 in enumerate(terminals_with_pos):
                for t2 in terminals_with_pos[i+1:]:
                    dist = analyzer._haversine_distance(
                        t1['position'], t2['position']
                    )
                    if dist > max_distance:
                        max_distance = dist
                        best_long_pair = (t1, t2)
            
            if best_long_pair:
                suggested_pairs.append({
                    'type': 'long_distance',
                    'sourceTerminalId': best_long_pair[0]['terminalId'],
                    'sourceTerminalName': best_long_pair[0].get('terminalName', 'Unknown'),
                    'destTerminalId': best_long_pair[1]['terminalId'],
                    'destTerminalName': best_long_pair[1].get('terminalName', 'Unknown'),
                    'distance_km': round(max_distance, 2),
                    'reason': 'Longest distance pair - tests global coverage',
                    'priority': 'high'
                })
        
        # 2. Short-distance pairs (low-latency test)
        if len(terminals_with_pos) >= 2:
            min_distance = float('inf')
            best_short_pair = None
            for i, t1 in enumerate(terminals_with_pos):
                for t2 in terminals_with_pos[i+1:]:
                    dist = analyzer._haversine_distance(
                        t1['position'], t2['position']
                    )
                    if dist < min_distance and dist > 100:  # At least 100km apart
                        min_distance = dist
                        best_short_pair = (t1, t2)
            
            if best_short_pair:
                suggested_pairs.append({
                    'type': 'short_distance',
                    'sourceTerminalId': best_short_pair[0]['terminalId'],
                    'sourceTerminalName': best_short_pair[0].get('terminalName', 'Unknown'),
                    'destTerminalId': best_short_pair[1]['terminalId'],
                    'destTerminalName': best_short_pair[1].get('terminalName', 'Unknown'),
                    'distance_km': round(min_distance, 2),
                    'reason': 'Shortest viable distance - tests low-latency routing',
                    'priority': 'medium'
                })
        
        # 3. High-traffic area pairs (based on overloaded nodes)
        if overload_analysis['overloaded_nodes']:
            # Find terminals near overloaded ground stations
            overloaded_grounds = [
                n for n in overload_analysis['overloaded_nodes']
                if n.get('nodeType') == 'GROUND_STATION'
            ][:2]
            
            if len(overloaded_grounds) >= 1 and len(terminals_with_pos) >= 2:
                # Find terminal closest to overloaded ground station
                for ground in overloaded_grounds:
                    ground_pos = ground.get('position')
                    if not ground_pos:
                        continue
                    
                    closest_terminal = min(
                        terminals_with_pos,
                        key=lambda t: analyzer._haversine_distance(t['position'], ground_pos)
                    )
                    
                    # Pick a random distant terminal
                    other_terminals = [t for t in terminals_with_pos if t != closest_terminal]
                    if other_terminals:
                        dest_terminal = random.choice(other_terminals)
                        suggested_pairs.append({
                            'type': 'high_traffic',
                            'sourceTerminalId': closest_terminal['terminalId'],
                            'sourceTerminalName': closest_terminal.get('terminalName', 'Unknown'),
                            'destTerminalId': dest_terminal['terminalId'],
                            'destTerminalName': dest_terminal.get('terminalName', 'Unknown'),
                            'distance_km': round(analyzer._haversine_distance(
                                closest_terminal['position'], dest_terminal['position']
                            ), 2),
                            'reason': f'Tests routing through overloaded area ({ground.get("nodeName", "Unknown")})',
                            'priority': 'high'
                        })
        
        # Build recommendations
        recommendations = [
            {
                'type': 'info',
                'message': health_note,
                'suggestions': [
                    f'Suggested batch size: {suggested_pair_count} pairs',
                    f'Network health: {100 - overload_percentage:.1f}% available capacity',
                    f'Total possible pairs: {max_possible_pairs}'
                ]
            }
        ]
        
        # Add overload recommendations
        if overload_percentage > 30:
            recommendations.append({
                'type': 'warning',
                'message': f'{overload_analysis["summary"]["overloaded_count"]} nodes are overloaded',
                'suggestions': [
                    f'Consider reducing batch size to avoid further stress',
                    f'Focus on testing alternative routing paths',
                    f'Monitor {overload_analysis["summary"]["overloaded_count"]} overloaded nodes'
                ]
            })
        
        # Add pair suggestions
        if suggested_pairs:
            recommendations.append({
                'type': 'success',
                'message': f'Found {len(suggested_pairs)} interesting test scenarios',
                'suggestions': [
                    f'{pair["type"].replace("_", " ").title()}: {pair["sourceTerminalName"]} → {pair["destTerminalName"]} ({pair["distance_km"]} km)'
                    for pair in suggested_pairs
                ]
            })
        
        # 4. Add link quality predictions for top suggested pair
        link_quality_predictions = None
        best_time_recommendation = None
        if suggested_pairs and len(suggested_pairs) > 0:
            try:
                # Predict for the highest priority pair
                top_pair = suggested_pairs[0]
                source_terminal = next((t for t in terminals if t['terminalId'] == top_pair['sourceTerminalId']), None)
                dest_terminal = next((t for t in terminals if t['terminalId'] == top_pair['destTerminalId']), None)
                
                if source_terminal and dest_terminal:
                    source_pos = source_terminal.get('position')
                    dest_pos = dest_terminal.get('position')
                    
                    if source_pos and dest_pos:
                        # Predict next 6 hours with 1-hour steps
                        link_predictions = analyzer.predict_link_quality_over_time(
                            source_pos=source_pos,
                            dest_pos=dest_pos,
                            nodes=nodes,
                            time_horizon_hours=6,
                            time_step_minutes=60
                        )
                        
                        if link_predictions['predictions']:
                            link_quality_predictions = {
                                'pairInfo': {
                                    'source': top_pair['sourceTerminalName'],
                                    'destination': top_pair['destTerminalName'],
                                    'distance_km': top_pair['distance_km']
                                },
                                'bestTime': link_predictions['summary']['best_time'],
                                'bestQuality': round(link_predictions['summary']['best_quality'], 3),
                                'worstTime': link_predictions['summary']['worst_time'],
                                'worstQuality': round(link_predictions['summary']['worst_quality'], 3),
                                'averageQuality': round(link_predictions['summary']['average_quality'], 3),
                                'predictions': [
                                    {
                                        'timestamp': p['timestamp'],
                                        'quality': round(p['best_link']['quality_score'], 3),
                                        'latency_ms': round(p['best_link']['estimated_latency_ms'], 2),
                                        'snr_db': round(p['best_link']['estimated_snr_db'], 1),
                                        'satellite': p['best_link']['satellite_name']
                                    }
                                    for p in link_predictions['predictions']
                                ]
                            }
                            
                            best_time_recommendation = link_predictions['summary']['recommendation']
                            
                            # Add time-based recommendation
                            recommendations.append({
                                'type': 'info',
                                'message': f'Link Quality Analysis for {top_pair["sourceTerminalName"]} → {top_pair["destTerminalName"]}',
                                'suggestions': [
                                    f'Best transmission time: {link_predictions["summary"]["best_time"]} (quality: {link_predictions["summary"]["best_quality"]:.3f})',
                                    f'Average quality: {link_predictions["summary"]["average_quality"]:.3f}',
                                    best_time_recommendation
                                ]
                            })
            except Exception as e:
                logger.warning(f"Could not generate link quality predictions: {e}")
        
        # 5. Add node placement recommendations
        placement_recommendations = None
        try:
            placement_analysis = analyzer.recommend_node_placement(
                nodes=nodes,
                terminals=terminals,
                db=db,
                target_coverage=0.9,
                max_recommendations=3  # Top 3 locations
            )
            
            if placement_analysis['recommendations']:
                placement_recommendations = {
                    'currentCoverage': round(placement_analysis['coverage_analysis']['coverage_percentage'], 1),
                    'gapsIdentified': placement_analysis['coverage_analysis']['gaps_identified'],
                    'locations': [
                        {
                            'rank': rec['rank'],
                            'type': rec['recommended_type'],
                            'latitude': round(rec['latitude'], 2),
                            'longitude': round(rec['longitude'], 2),
                            'priorityScore': round(rec['priority_score'], 1),
                            'reason': rec['reason'],
                            'benefits': rec['expected_benefits'][:2]  # Top 2 benefits
                        }
                        for rec in placement_analysis['recommendations'][:3]
                    ]
                }
                
                # Add placement recommendation
                if placement_analysis['coverage_analysis']['gaps_identified'] > 0:
                    recommendations.append({
                        'type': 'warning',
                        'message': f'Network Coverage: {placement_analysis["coverage_analysis"]["coverage_percentage"]:.1f}% ({placement_analysis["coverage_analysis"]["gaps_identified"]} gaps identified)',
                        'suggestions': [
                            f'Recommend adding {len(placement_analysis["recommendations"])} new nodes',
                            f'Top priority: {placement_analysis["recommendations"][0]["recommended_type"]} at ({placement_analysis["recommendations"][0]["latitude"]:.1f}°, {placement_analysis["recommendations"][0]["longitude"]:.1f}°)',
                            placement_analysis["recommendations"][0]["reason"]
                        ]
                    })
        except Exception as e:
            logger.warning(f"Could not generate node placement recommendations: {e}")
        
        response = {
            'suggestedPairCount': suggested_pair_count,
            'suggestedPairs': suggested_pairs[:5],  # Top 5 suggestions
            'linkQualityPrediction': link_quality_predictions,
            'nodePlacementRecommendations': placement_recommendations,
            'overloadAnalysis': {
                'overloadedNodes': [
                    {
                        'nodeName': node['nodeName'],
                        'nodeType': node.get('nodeType', 'UNKNOWN'),
                        'utilization': round(node['utilization'], 2),
                        'packetLoss': round(node['packetLoss'], 3),
                        'overloadScore': round(node['overloadScore'], 2)
                    }
                    for node in overload_analysis['overloaded_nodes'][:5]
                ],
                'atRiskNodes': overload_analysis['summary']['at_risk_count'],
                'recommendations': [
                    {
                        'priority': rec['priority'],
                        'message': rec['message'],
                        'suggestions': rec['suggestions'][:2]
                    }
                    for rec in overload_analysis['recommendations'][:3]
                ]
            },
            'networkHealth': {
                'totalNodes': len(nodes),
                'totalTerminals': total_terminals,
                'overloadedNodes': overload_analysis['summary']['overloaded_count'],
                'overloadPercentage': round(overload_percentage, 1),
                'availableCapacity': round(100 - overload_percentage, 1)
            },
            'recommendations': recommendations
        }
        
        logger.info(f"Generated batch suggestions: {suggested_pair_count} pairs, {len(suggested_pairs)} interesting scenarios")
        
        # Save suggestions to database for history tracking
        try:
            suggestions_collection = db.get_collection('network_suggestions')
            suggestion_record = {
                'timestamp': datetime.now(),
                'suggestedPairCount': suggested_pair_count,
                'networkHealth': response.get('networkHealth'),
                'overloadedNodesCount': len(response.get('overloadAnalysis', {}).get('overloadedNodes', [])),
                'recommendations': response.get('recommendations', []),
                'linkQualityPredictions': response.get('linkQualityPrediction'),
                'nodePlacementRecommendations': response.get('nodePlacementRecommendations'),
                'totalNodes': len(nodes),
                'totalTerminals': len(terminals),
                'scenario': 'auto_analysis'
            }
            suggestions_collection.insert_one(suggestion_record)
            logger.info(f"Saved network suggestion to database at {datetime.now()}")
        except Exception as save_error:
            logger.warning(f"Failed to save suggestion to database: {save_error}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error generating batch suggestions: {str(e)}")
        return jsonify({
            'suggestedPairCount': 10,
            'suggestedPairs': [],
            'recommendations': [{
                'type': 'error',
                'message': f'Error analyzing network: {str(e)}',
                'suggestions': ['Using default parameters']
            }]
        }), 200  # Return 200 with defaults to not break UI


@batch_bp.route('/start-streaming', methods=['POST'])
def start_streaming():
    """Start streaming batches via WebSocket"""
    global _batch_generation_active, _batch_generation_thread
    
    data = request.get_json() or {}
    interval = data.get('interval', 5)  # seconds between batches
    pair_count = data.get('pairCount', 10)
    
    if _batch_generation_active:
        return jsonify({
            'status': 400,
            'error': 'Bad request',
            'message': 'Batch generation is already active'
        }), 400
    
    _batch_generation_active = True
    
    def generate_and_queue():
        while _batch_generation_active:
            try:
                # Generate batch
                terminals_collection = db.get_collection('terminals')
                terminals = list(terminals_collection.find({}, {'_id': 0}))
                
                if len(terminals) < 2:
                    time.sleep(interval)
                    continue
                
                nodes_collection = db.get_collection('nodes')
                nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
                
                if len(nodes) == 0:
                    time.sleep(interval)
                    continue
                
                batch_id = f"BATCH-{int(datetime.now().timestamp() * 1000)}"
                packets = []
                
                for i in range(pair_count):
                    source_terminal = random.choice(terminals)
                    dest_terminal = random.choice(terminals)
                    while dest_terminal['terminalId'] == source_terminal['terminalId']:
                        dest_terminal = random.choice(terminals)
                    
                    service_qos = {
                        'maxLatencyMs': random.randint(50, 200),
                        'minBandwidthMbps': random.uniform(0.5, 5.0),
                        'maxLossRate': random.uniform(0.001, 0.05),
                        'priority': random.randint(1, 10),
                        'serviceType': random.choice(['TEXT_MESSAGE', 'IMAGE_TRANSFER', 'VIDEO_STREAM'])
                    }
                    
                    path_dijkstra = calculate_path_dijkstra(source_terminal, dest_terminal, nodes)
                    
                    # Try RL, skip pair nếu RL không available
                    try:
                        path_rl = calculate_path_rl(source_terminal, dest_terminal, nodes, service_qos=service_qos)
                    except (RuntimeError, ValueError) as e:
                        logger.debug(f"RL not available for streaming batch: {e}")
                        continue
                    
                    dijkstra_packet = create_packet_from_path(path_dijkstra, source_terminal, dest_terminal, 'dijkstra', service_qos)
                    rl_packet = create_packet_from_path(path_rl, source_terminal, dest_terminal, 'rl', service_qos)
                    
                    packets.append({
                        'dijkstraPacket': dijkstra_packet,
                        'rlPacket': rl_packet
                    })
                
                batch = {
                    'batchId': batch_id,
                    'totalPairPackets': pair_count,
                    'packets': packets,
                    'scenario': 'NORMAL'  # Default scenario for auto-generated batches
                }
                
                _batch_message_queue.append(batch)
                logger.info(f"Generated and queued batch {batch_id}")
                
            except Exception as e:
                logger.error(f"Error in batch generation thread: {str(e)}")
            
            time.sleep(interval)
    
    _batch_generation_thread = threading.Thread(target=generate_and_queue, daemon=True)
    _batch_generation_thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Started streaming batches every {interval} seconds',
        'interval': interval,
        'pairCount': pair_count
    }), 200

@batch_bp.route('/stop-streaming', methods=['POST'])
def stop_streaming():
    """Stop streaming batches"""
    global _batch_generation_active
    
    _batch_generation_active = False
    
    return jsonify({
        'success': True,
        'message': 'Stopped streaming batches'
    }), 200

@batch_bp.route('/queue', methods=['GET'])
def get_queue():
    """Get current message queue (for debugging)"""
    return jsonify({
        'queueLength': len(_batch_message_queue),
        'batches': _batch_message_queue[-10:]  # Last 10 batches
    }), 200

@batch_bp.route('/queue/clear', methods=['POST'])
def clear_queue():
    """Clear message queue"""
    global _batch_message_queue
    _batch_message_queue.clear()
    
    return jsonify({
        'success': True,
        'message': 'Queue cleared'
    }), 200

# Function to get next batch from queue (for WebSocket handler)
def get_next_batch():
    """Get next batch from queue and remove it"""
    if _batch_message_queue:
        return _batch_message_queue.pop(0)
    return None

@batch_bp.route('/poll', methods=['GET'])
def poll_batch():
    """Poll for next batch from queue (for frontend polling)"""
    batch = get_next_batch()
    if batch:
        return jsonify(batch), 200
    else:
        return jsonify({
            'status': 'no_data',
            'message': 'No batch available'
        }), 204  # No Content


# ==================== HISTORY & ANALYTICS ENDPOINTS 

@batch_bp.route('/history/suggestions', methods=['GET'])
def get_suggestions_history():
    """Get history of network suggestions"""
    try:
        limit = int(request.args.get('limit', 20))
        skip = int(request.args.get('skip', 0))
        
        suggestions_collection = db.get_collection('network_suggestions')
        
        # Get suggestions sorted by timestamp (newest first)
        suggestions = list(suggestions_collection.find(
            {},
            {'_id': 0}
        ).sort('timestamp', -1).skip(skip).limit(limit))
        
        # Get total count
        total = suggestions_collection.count_documents({})
        
        return jsonify({
            'suggestions': suggestions,
            'total': total,
            'limit': limit,
            'skip': skip
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting suggestions history: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve suggestions history',
            'message': str(e)
        }), 500


@batch_bp.route('/history/results', methods=['GET'])
def get_batch_results_history():
    """Get history of batch test results"""
    try:
        limit = int(request.args.get('limit', 20))
        skip = int(request.args.get('skip', 0))
        scenario = request.args.get('scenario')  # Optional filter
        
        batch_results_collection = db.get_collection('batch_results')
        
        # Build query
        query = {}
        if scenario:
            query['scenario'] = scenario
        
        # Get batch results sorted by timestamp (newest first)
        results = list(batch_results_collection.find(
            query,
            {'_id': 0, 'packets': 0}  # Exclude full packet data for performance
        ).sort('timestamp', -1).skip(skip).limit(limit))
        
        # Get total count
        total = batch_results_collection.count_documents(query)
        
        return jsonify({
            'results': results,
            'total': total,
            'limit': limit,
            'skip': skip
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting batch results history: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve batch results history',
            'message': str(e)
        }), 500


@batch_bp.route('/history/results/<batch_id>', methods=['GET'])
def get_batch_result_details(batch_id):
    """Get detailed results for a specific batch"""
    try:
        batch_results_collection = db.get_collection('batch_results')
        
        result = batch_results_collection.find_one(
            {'batchId': batch_id},
            {'_id': 0}
        )
        
        if not result:
            return jsonify({
                'error': 'Batch not found',
                'message': f'No batch found with ID {batch_id}'
            }), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting batch result details: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve batch details',
            'message': str(e)
        }), 500


@batch_bp.route('/analytics/comparison', methods=['GET'])
def get_algorithm_comparison():
    """Get aggregated comparison analytics between Dijkstra and RL"""
    try:
        days = int(request.args.get('days', 7))  # Last 7 days by default
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        batch_results_collection = db.get_collection('batch_results')
        
        # Aggregate statistics
        pipeline = [
            {'$match': {'timestamp': {'$gte': cutoff_date}}},
            {'$group': {
                '_id': None,
                'totalBatches': {'$sum': 1},
                'avgDijkstraLatency': {'$avg': '$statistics.dijkstra.avgLatencyMs'},
                'avgRlLatency': {'$avg': '$statistics.rl.avgLatencyMs'},
                'avgDijkstraDistance': {'$avg': '$statistics.dijkstra.avgDistanceKm'},
                'avgRlDistance': {'$avg': '$statistics.rl.avgDistanceKm'},
                'avgDijkstraHops': {'$avg': '$statistics.dijkstra.avgHops'},
                'avgRlHops': {'$avg': '$statistics.rl.avgHops'},
                'avgDijkstraSuccess': {'$avg': '$statistics.dijkstra.successRate'},
                'avgRlSuccess': {'$avg': '$statistics.rl.successRate'},
                'totalLatencyImprovement': {'$avg': '$statistics.comparison.latencyImprovement'},
                'totalDistanceImprovement': {'$avg': '$statistics.comparison.distanceImprovement'},
                'totalRlWins': {'$sum': '$statistics.comparison.rlWins'},
                'totalDijkstraWins': {'$sum': '$statistics.comparison.dijkstraWins'}
            }}
        ]
        
        result = list(batch_results_collection.aggregate(pipeline))
        
        if not result:
            return jsonify({
                'message': 'No data available for the specified period',
                'days': days
            }), 200
        
        stats = result[0]
        stats.pop('_id', None)
        
        return jsonify({
            'period': f'Last {days} days',
            'statistics': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting comparison analytics: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve analytics',
            'message': str(e)
        }), 500


@batch_bp.route('/analytics/trends', methods=['GET'])
def get_network_trends():
    """Get network health trends over time"""
    try:
        days = int(request.args.get('days', 7))
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        suggestions_collection = db.get_collection('network_suggestions')
        
        # Get suggestions over time
        pipeline = [
            {'$match': {'timestamp': {'$gte': cutoff_date}}},
            {'$sort': {'timestamp': 1}},
            {'$project': {
                '_id': 0,
                'timestamp': 1,
                'overloadedNodesCount': 1,
                'networkHealth.overloadPercentage': 1,
                'networkHealth.availableCapacity': 1,
                'networkHealth.totalNodes': 1,
                'networkHealth.totalTerminals': 1
            }}
        ]
        
        trends = list(suggestions_collection.aggregate(pipeline))
        
        return jsonify({
            'period': f'Last {days} days',
            'dataPoints': len(trends),
            'trends': trends
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting network trends: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve trends',
            'message': str(e)
        }), 500


@batch_bp.route('/history/comparisons', methods=['GET'])
def get_comparisons_history():
    """Get algorithm comparison history from database"""
    try:
        # Get query parameters
        limit = request.args.get('limit', default=50, type=int)
        skip = request.args.get('skip', default=0, type=int)
        scenario = request.args.get('scenario', default=None, type=str)
        algorithm1 = request.args.get('algorithm1', default=None, type=str)
        algorithm2 = request.args.get('algorithm2', default=None, type=str)
        
        # Limit bounds
        limit = min(max(1, limit), 200)
        skip = max(0, skip)
        
        comparisons_collection = db.get_collection('algorithm_comparisons')
        
        # Build query filter
        query_filter = {}
        if scenario:
            query_filter['scenario'] = scenario
        if algorithm1:
            query_filter['algorithm1.name'] = algorithm1
        if algorithm2:
            query_filter['algorithm2.name'] = algorithm2
        
        # Get total count
        total_count = comparisons_collection.count_documents(query_filter)
        
        # Get paginated results
        comparisons = list(
            comparisons_collection
            .find(query_filter, {'_id': 0})
            .sort('createdAt', -1)
            .skip(skip)
            .limit(limit)
        )
        
        return jsonify({
            'total': total_count,
            'limit': limit,
            'skip': skip,
            'count': len(comparisons),
            'comparisons': comparisons
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting comparison history: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve comparison history',
            'message': str(e)
        }), 500


@batch_bp.route('/history/comparisons/stats', methods=['GET'])
def get_comparisons_stats():
    """Get aggregate statistics from comparison history"""
    try:
        comparisons_collection = db.get_collection('algorithm_comparisons')
        
        # Get scenario parameter
        scenario = request.args.get('scenario', default=None, type=str)
        query_filter = {'scenario': scenario} if scenario else {}
        
        # Aggregate statistics
        pipeline = [
            {'$match': query_filter},
            {'$group': {
                '_id': None,
                'totalComparisons': {'$sum': 1},
                'avgLatencyDiff': {'$avg': '$comparison.latencyDifference'},
                'avgDistanceDiff': {'$avg': '$comparison.distanceDifference'},
                'avgHopsDiff': {'$avg': '$comparison.hopsDifference'},
                'scenarios': {'$addToSet': '$scenario'},
                'algorithmPairs': {'$addToSet': {
                    'alg1': '$algorithm1.name',
                    'alg2': '$algorithm2.name'
                }}
            }}
        ]
        
        stats = list(comparisons_collection.aggregate(pipeline))
        
        if not stats:
            return jsonify({
                'totalComparisons': 0,
                'message': 'No comparison data available'
            }), 200
        
        result = stats[0]
        result.pop('_id', None)
        
        # Get comparison counts by algorithm pair
        pair_pipeline = [
            {'$match': query_filter},
            {'$group': {
                '_id': {
                    'alg1': '$algorithm1.name',
                    'alg2': '$algorithm2.name',
                    'scenario': '$scenario'
                },
                'count': {'$sum': 1},
                'avgLatencyDiff': {'$avg': '$comparison.latencyDifference'},
                'bestLatencyWinner': {'$push': '$comparison.bestLatency'}
            }}
        ]
        
        pair_stats = list(comparisons_collection.aggregate(pair_pipeline))
        result['pairStatistics'] = pair_stats
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting comparison stats: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve comparison statistics',
            'message': str(e)
        }), 500
