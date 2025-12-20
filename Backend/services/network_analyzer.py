"""
Network Analyzer Service
Analyzes network using RL insights and provides recommendations:
- Detect overloaded nodes
- Recommend node placement locations
- Predict link quality over time
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from collections import defaultdict

from environment.constants import (
    EARTH_RADIUS_M, M_TO_KM, KM_TO_M,
    UTILIZATION_HIGH_PERCENT, UTILIZATION_MEDIUM_PERCENT,
    UTILIZATION_CRITICAL_PERCENT, UTILIZATION_MAX_PERCENT,
    BATTERY_MAX_PERCENT, PACKET_LOSS_HIGH,
    DISTANCE_NEAR_DEST_M, DISTANCE_CLOSE_DEST_M, DISTANCE_FAR_DEST_M,
    SPEED_OF_LIGHT_MPS, MS_PER_SECOND
)

logger = logging.getLogger(__name__)

# Network Analyzer specific constants
OVERLOAD_SCORE_WEIGHT_UTILIZATION = 0.4
OVERLOAD_SCORE_WEIGHT_PACKET_LOSS = 0.3
OVERLOAD_SCORE_WEIGHT_QUEUE = 0.2
OVERLOAD_SCORE_WEIGHT_BATTERY = 0.1
AT_RISK_SCORE_THRESHOLD = 0.6

OVERLOAD_UTILIZATION_THRESHOLD = UTILIZATION_HIGH_PERCENT
OVERLOAD_PACKET_LOSS_THRESHOLD = 0.05
OVERLOAD_QUEUE_RATIO_THRESHOLD = 0.7
OVERLOAD_UTILIZATION_MEDIUM = UTILIZATION_MEDIUM_PERCENT
OVERLOAD_PACKET_LOSS_MEDIUM = 0.03

NEARBY_TERMINAL_RANGE_KM = 500
NEARBY_NODE_RANGE_KM = 1000
COVERAGE_GAP_DISTANCE_KM = 2000
COVERAGE_CHECK_DISTANCE_KM = 2000

LINK_QUALITY_WEIGHT_DISTANCE = 0.4
LINK_QUALITY_WEIGHT_ELEVATION = 0.3
LINK_QUALITY_WEIGHT_TYPE = 0.2
LINK_QUALITY_WEIGHT_ATMOSPHERIC = 0.1

LINK_QUALITY_EXCELLENT = 0.8
LINK_QUALITY_GOOD = 0.6
LINK_QUALITY_MODERATE = 0.4
LINK_LATENCY_EXCELLENT_MS = 100.0

LEO_ORBITAL_VELOCITY_DEG_PER_HOUR = 225.0
MEO_ORBITAL_VELOCITY_DEG_PER_HOUR = 30.0
LEO_ORBIT_PERIOD_HOURS = 1.5
LEO_LATITUDE_VARIATION_DEG = 30.0

GEO_SATELLITE_TYPE_BONUS = 0.35
MEO_SATELLITE_TYPE_BONUS = 0.25
LEO_SATELLITE_TYPE_BONUS = 0.15

GEO_BASE_SNR_DB = 25.0
MEO_BASE_SNR_DB = 20.0
LEO_BASE_SNR_DB = 15.0

PROCESSING_DELAY_MS = 5.0


class NetworkAnalyzer:
    """Network analyzer sử dụng RL insights để đưa ra recommendations"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def analyze_overloaded_nodes(
        self,
        nodes: List[Dict],
        terminals: List[Dict],
        db=None,
        threshold_utilization: float = OVERLOAD_UTILIZATION_THRESHOLD / 100.0,
        threshold_packet_loss: float = OVERLOAD_PACKET_LOSS_THRESHOLD,
        focus_ground_stations: bool = True
    ) -> Dict:
        """
        Phân tích và phát hiện nodes quá tải
        Tập trung vào ground stations nếu focus_ground_stations=True
        
        Returns:
            {
                'overloaded_nodes': [...],
                'at_risk_nodes': [...],
                'recommendations': [...]
            }
        """
        overloaded = []
        at_risk = []
        
        # Load traffic demand nếu có database
        traffic_analysis = None
        if db:
            try:
                traffic_analysis = self.analyze_traffic_demand(db, time_window_hours=24)
            except Exception as e:
                logger.warning(f"Could not analyze traffic demand: {e}")
        
        # Filter nodes - focus on ground stations nếu được yêu cầu
        nodes_to_analyze = nodes
        if focus_ground_stations:
            nodes_to_analyze = [n for n in nodes if n.get('nodeType') == 'GROUND_STATION']
        
        for node in nodes_to_analyze:
            if not node.get('isOperational', True):
                continue
            
            utilization = node.get('resourceUtilization', 0) / UTILIZATION_MAX_PERCENT
            packet_loss = node.get('packetLossRate', 0)
            battery = node.get('batteryChargePercent', BATTERY_MAX_PERCENT)
            queue_length = node.get('currentPacketCount', 0)
            capacity = node.get('packetBufferCapacity', 1000)
            queue_ratio = queue_length / max(capacity, 1)
            
            overload_score = (
                utilization * OVERLOAD_SCORE_WEIGHT_UTILIZATION +
                packet_loss * OVERLOAD_SCORE_WEIGHT_PACKET_LOSS +
                queue_ratio * OVERLOAD_SCORE_WEIGHT_QUEUE +
                (1.0 - battery / BATTERY_MAX_PERCENT) * OVERLOAD_SCORE_WEIGHT_BATTERY
            )
            
            # Get traffic data nếu có
            traffic_data = {}
            if traffic_analysis:
                node_traffic = traffic_analysis['node_traffic'].get(node.get('nodeId'), {})
                traffic_data = {
                    'total_packets': node_traffic.get('total_packets', 0),
                    'total_bytes': node_traffic.get('total_bytes', 0),
                    'incoming_traffic': node_traffic.get('incoming_traffic', 0),
                    'outgoing_traffic': node_traffic.get('outgoing_traffic', 0)
                }
            
            node_info = {
                'nodeId': node.get('nodeId'),
                'nodeName': node.get('nodeName', node.get('nodeId')),
                'nodeType': node.get('nodeType'),
                'position': node.get('position'),
                'utilization': utilization * 100,
                'packetLoss': packet_loss * 100,
                'battery': battery,
                'queueRatio': queue_ratio * 100,
                'overloadScore': overload_score * 100,
                'traffic_data': traffic_data
            }
            
            if utilization > threshold_utilization or packet_loss > threshold_packet_loss:
                overloaded.append(node_info)
            elif overload_score > AT_RISK_SCORE_THRESHOLD:
                at_risk.append(node_info)
        
        # Sort by overload score
        overloaded.sort(key=lambda x: x['overloadScore'], reverse=True)
        at_risk.sort(key=lambda x: x['overloadScore'], reverse=True)
        
        # Generate recommendations với focus vào ground stations
        recommendations = []
        for node in overloaded[:5]:  # Top 5 most overloaded
            traffic_info = ""
            if node.get('traffic_data', {}).get('total_packets', 0) > 0:
                traffic_info = f", handling {node['traffic_data']['total_packets']} packets"
            
            suggestions = [
                'Add new ground station nearby to offload traffic',
                'Distribute traffic to nearby satellites or other ground stations',
                'Consider upgrading node capacity if expansion not possible'
            ]
            
            if node.get('nodeType') == 'GROUND_STATION':
                suggestions.insert(0, 'URGENT: Add backup ground station to reduce load')
            
            recommendations.append({
                'type': 'overload_warning',
                'nodeId': node['nodeId'],
                'message': f"Ground station {node['nodeName']} is overloaded: {node['utilization']:.1f}% utilization, {node['packetLoss']:.2f}% packet loss{traffic_info}",
                'priority': 'high' if node.get('nodeType') == 'GROUND_STATION' else 'medium',
                'suggestions': suggestions,
                'traffic_load': node.get('traffic_data', {})
            })
        
        return {
            'overloaded_nodes': overloaded,
            'at_risk_nodes': at_risk,
            'recommendations': recommendations,
            'summary': {
                'total_nodes_analyzed': len(nodes_to_analyze),
                'total_nodes': len(nodes),
                'overloaded_count': len(overloaded),
                'at_risk_count': len(at_risk),
                'overload_percentage': len(overloaded) / max(len(nodes_to_analyze), 1) * 100,
                'focus_ground_stations': focus_ground_stations,
                'traffic_analysis_available': traffic_analysis is not None
            }
        }
    
    def analyze_traffic_demand(
        self,
        db,
        time_window_hours: int = 24
    ) -> Dict:
        """
        Phân tích traffic demand từ database
        Sử dụng traffic_demand collection và packets collection
        """
        from datetime import datetime, timedelta
        
        node_traffic = defaultdict(lambda: {
            'total_packets': 0,
            'total_bytes': 0,
            'incoming_traffic': 0,
            'outgoing_traffic': 0,
            'source_terminals': set(),
            'dest_terminals': set(),
            'peak_utilization': 0.0,
            'avg_utilization': 0.0
        })
        
        # 1. Load từ traffic_demand collection (tổng hợp theo ngày)
        try:
            traffic_demand_collection = db.get_collection('traffic_demand')
            today = datetime.now().date().isoformat()
            
            # Lấy traffic demand cho nodes
            node_demands = list(traffic_demand_collection.find({
                'nodeId': {'$exists': True},
                'date': today
            }, {'_id': 0}))
            
            for demand in node_demands:
                node_id = demand.get('nodeId')
                if node_id:
                    node_traffic[node_id]['total_packets'] = demand.get('totalPackets', 0)
                    node_traffic[node_id]['total_bytes'] = demand.get('totalBytes', 0)
                    node_traffic[node_id]['incoming_traffic'] = demand.get('incomingTraffic', 0)
                    node_traffic[node_id]['outgoing_traffic'] = demand.get('outgoingTraffic', 0)
                    node_traffic[node_id]['source_terminals'] = set(demand.get('sourceTerminals', []))
                    node_traffic[node_id]['dest_terminals'] = set(demand.get('destTerminals', []))
        except Exception as e:
            logger.warning(f"Could not load traffic_demand: {e}")
        
        # 2. Load từ packets collection (nếu có, để có thêm chi tiết)
        try:
            packets_collection = db.get_collection('packets')
            time_threshold = datetime.now() - timedelta(hours=time_window_hours)
            
            # Get packets trong time window
            packets = list(packets_collection.find({
                'sentAt': {'$gte': time_threshold.isoformat()}
            }, {'_id': 0, 'path': 1, 'packetSize': 1, 'sourceTerminalId': 1, 'destinationTerminalId': 1}))
            
            for packet in packets:
                path = packet.get('path', {})
                path_segments = path.get('path', [])
                packet_size = packet.get('packetSize', 0)
                
                # Count traffic through each node in path
                for segment in path_segments:
                    if segment.get('type') == 'node':
                        node_id = segment.get('id')
                        if node_id:
                            node_traffic[node_id]['total_packets'] += 1
                            node_traffic[node_id]['total_bytes'] += packet_size
                            node_traffic[node_id]['incoming_traffic'] += packet_size
                            node_traffic[node_id]['outgoing_traffic'] += packet_size
                            
                            # Track terminals
                            source_id = packet.get('sourceTerminalId')
                            dest_id = packet.get('destinationTerminalId')
                            if source_id:
                                node_traffic[node_id]['source_terminals'].add(source_id)
                            if dest_id:
                                node_traffic[node_id]['dest_terminals'].add(dest_id)
        except Exception as e:
            logger.warning(f"Could not load packets: {e}")
        
        # 3. Get current node utilization và merge với traffic data
        try:
            nodes_collection = db.get_collection('nodes')
            current_nodes = list(nodes_collection.find({'isOperational': True}, {
                '_id': 0,
                'nodeId': 1,
                'nodeType': 1,
                'resourceUtilization': 1,
                'currentPacketCount': 1,
                'packetBufferCapacity': 1,
                'position': 1
            }))
            
            for node in current_nodes:
                node_id = node.get('nodeId')
                utilization = node.get('resourceUtilization', 0)
                
                if node_id in node_traffic:
                    node_traffic[node_id]['peak_utilization'] = max(
                        node_traffic[node_id]['peak_utilization'],
                        utilization
                    )
                    node_traffic[node_id]['avg_utilization'] = utilization
                    node_traffic[node_id]['node_type'] = node.get('nodeType')
                    node_traffic[node_id]['position'] = node.get('position')
                else:
                    # Initialize if not in traffic data
                    node_traffic[node_id] = {
                        'total_packets': 0,
                        'total_bytes': 0,
                        'incoming_traffic': 0,
                        'outgoing_traffic': 0,
                        'source_terminals': set(),
                        'dest_terminals': set(),
                        'peak_utilization': utilization,
                        'avg_utilization': utilization,
                        'node_type': node.get('nodeType'),
                        'position': node.get('position')
                    }
        except Exception as e:
            logger.warning(f"Could not load node data: {e}")
        
        # Convert sets to lists for JSON serialization
        for node_id, data in node_traffic.items():
            data['source_terminals'] = list(data['source_terminals'])
            data['dest_terminals'] = list(data['dest_terminals'])
            data['source_count'] = len(data['source_terminals'])
            data['dest_count'] = len(data['dest_terminals'])
        
        # Focus on ground stations
        ground_station_traffic = {
            node_id: data for node_id, data in node_traffic.items()
            if data.get('node_type') == 'GROUND_STATION'
        }
        
        return {
            'node_traffic': dict(node_traffic),
            'ground_station_traffic': ground_station_traffic,
            'time_window_hours': time_window_hours,
            'summary': {
                'total_nodes_with_traffic': len(node_traffic),
                'ground_stations_with_traffic': len(ground_station_traffic),
                'total_packets': sum(d['total_packets'] for d in node_traffic.values()),
                'total_bytes': sum(d['total_bytes'] for d in node_traffic.values())
            }
        }
    
    def recommend_node_placement(
        self,
        nodes: List[Dict],
        terminals: List[Dict],
        db=None,
        target_coverage: float = 0.9,
        max_recommendations: int = 5
    ) -> Dict:
        """
        Đề xuất vị trí tốt để thêm nodes mới dựa trên:
        - Traffic demand từ database
        - Overloaded ground stations
        - Coverage gaps
        - Traffic density
        """
        recommendations = []
        
        # 1. Phân tích traffic demand từ database
        traffic_analysis = None
        if db:
            try:
                traffic_analysis = self.analyze_traffic_demand(db, time_window_hours=24)
            except Exception as e:
                logger.warning(f"Could not analyze traffic demand: {e}")
        
        # 2. Tìm overloaded ground stations
        ground_stations = [n for n in nodes if n.get('nodeType') == 'GROUND_STATION' and n.get('isOperational', True)]
        overloaded_ground_stations = []
        
        for gs in ground_stations:
            utilization = gs.get('resourceUtilization', 0)
            packet_loss = gs.get('packetLossRate', 0)
            queue_ratio = gs.get('currentPacketCount', 0) / max(gs.get('packetBufferCapacity', 1000), 1)
            
            # Tính traffic load từ traffic analysis nếu có
            traffic_load = 0
            if traffic_analysis:
                gs_traffic = traffic_analysis['ground_station_traffic'].get(gs.get('nodeId'), {})
                traffic_load = gs_traffic.get('total_packets', 0) + gs_traffic.get('total_bytes', 0) / 1000
            
            if (utilization > OVERLOAD_UTILIZATION_MEDIUM or 
                packet_loss > OVERLOAD_PACKET_LOSS_MEDIUM or 
                queue_ratio > OVERLOAD_QUEUE_RATIO_THRESHOLD):
                overloaded_ground_stations.append({
                    'node': gs,
                    'utilization': utilization,
                    'packet_loss': packet_loss,
                    'queue_ratio': queue_ratio,
                    'traffic_load': traffic_load,
                    'overload_score': (
                        (utilization / UTILIZATION_MAX_PERCENT) * 0.5 + 
                        packet_loss * 0.3 + 
                        queue_ratio * 0.2
                    )
                })
        
        # Sort by overload score
        overloaded_ground_stations.sort(key=lambda x: x['overload_score'], reverse=True)
        
        # 3. Đề xuất vị trí để giảm tải cho overloaded ground stations
        terminal_positions = [t.get('position') for t in terminals if t.get('position')]
        node_positions = [n.get('position') for n in nodes if n.get('position') and n.get('isOperational', True)]
        
        if not terminal_positions:
            return {'recommendations': [], 'coverage_analysis': {}, 'traffic_analysis': traffic_analysis}
        
        # Tìm terminals gần overloaded ground stations
        coverage_gaps = []
        
        for overloaded_gs in overloaded_ground_stations[:max_recommendations]:
            gs = overloaded_gs['node']
            gs_pos = gs.get('position')
            if not gs_pos:
                continue
            
            nearby_terminals = []
            for term in terminals:
                term_pos = term.get('position')
                if term_pos:
                    dist = self._haversine_distance(gs_pos, term_pos)
                    if dist < NEARBY_TERMINAL_RANGE_KM:
                        nearby_terminals.append({
                            'terminal': term,
                            'position': term_pos,
                            'distance_to_gs': dist
                        })
            
            nearby_nodes = sum(
                1 for np in node_positions
                if self._haversine_distance(gs_pos, np) < NEARBY_NODE_RANGE_KM
            )
            
            # Đề xuất vị trí mới để giảm tải
            # Tìm vị trí tốt nhất: gần terminals nhưng không quá gần ground station hiện tại
            for term_info in nearby_terminals[:3]:  # Top 3 terminals gần nhất
                term_pos = term_info['position']
                
                priority = (
                    len(nearby_terminals) * 10 +
                    (NEARBY_TERMINAL_RANGE_KM - term_info['distance_to_gs']) / 10 +
                    -nearby_nodes * 5
                )
                
                coverage_gaps.append({
                    'position': term_pos,
                    'priority': priority,
                    'nearby_terminals': len(nearby_terminals),
                    'nearby_nodes': nearby_nodes,
                    'recommended_type': 'GROUND_STATION',  # Luôn đề xuất ground station để giảm tải
                    'overloaded_gs_id': gs.get('nodeId'),
                    'overloaded_gs_name': gs.get('nodeName', gs.get('nodeId')),
                    'overload_score': overloaded_gs['overload_score'],
                    'reason': f"To offload traffic from overloaded {gs.get('nodeName', gs.get('nodeId'))} ({overloaded_gs['utilization']:.1f}% utilization)"
                })
        
        # 4. Nếu không có overloaded ground stations, phân tích coverage gaps như cũ
        if not coverage_gaps:
            # Fallback to coverage-based recommendations
            for term_pos in terminal_positions[:max_recommendations * 2]:
                min_dist = min(
                    (self._haversine_distance(term_pos, np) for np in node_positions),
                    default=float('inf')
                )
                
                if min_dist > COVERAGE_GAP_DISTANCE_KM:
                    nearby_terminals = sum(
                        1 for tp in terminal_positions
                        if self._haversine_distance(term_pos, tp) < NEARBY_NODE_RANGE_KM
                    )
                    
                    nearby_nodes = sum(
                        1 for np in node_positions
                        if self._haversine_distance(term_pos, np) < COVERAGE_GAP_DISTANCE_KM
                    )
                    
                    coverage_gaps.append({
                        'position': term_pos,
                        'priority': nearby_terminals * 2 - nearby_nodes,
                        'nearby_terminals': nearby_terminals,
                        'nearby_nodes': nearby_nodes,
                        'recommended_type': self._recommend_node_type(term_pos, nodes),
                        'reason': f"Coverage gap: {nearby_terminals} terminals nearby, only {nearby_nodes} nodes"
                    })
        
        # Sort by priority
        coverage_gaps.sort(key=lambda x: x['priority'], reverse=True)
        
        # 5. Generate recommendations
        for i, gap in enumerate(coverage_gaps[:max_recommendations]):
            expected_benefits = []
            
            if 'overloaded_gs_id' in gap:
                # Recommendation để giảm tải
                expected_benefits = [
                    f"Reduce load on {gap['overloaded_gs_name']} by {gap['overload_score']*100:.1f}%",
                    f"Serve {gap['nearby_terminals']} terminals in the area",
                    f"Improve network reliability and reduce packet loss",
                    f"Distribute traffic load more evenly"
                ]
            else:
                # Coverage-based recommendation
                expected_benefits = [
                    f"Improve coverage for {gap['nearby_terminals']} terminals",
                    f"Reduce average distance to nearest node",
                    f"Improve network resilience"
                ]
            
            recommendations.append({
                'rank': i + 1,
                'position': gap['position'],
                'latitude': gap['position'].get('latitude'),
                'longitude': gap['position'].get('longitude'),
                'recommended_type': gap['recommended_type'],
                'priority_score': gap['priority'],
                'reason': gap.get('reason', f"Coverage gap: {gap['nearby_terminals']} terminals nearby"),
                'expected_benefits': expected_benefits,
                'overloaded_gs_id': gap.get('overloaded_gs_id'),
                'overloaded_gs_name': gap.get('overloaded_gs_name'),
                'overload_score': gap.get('overload_score', 0)
            })
        
        # 6. Coverage analysis
        total_terminals = len(terminals)
        if node_positions:
            covered_terminals = sum(
                1 for term_pos in terminal_positions
                if min((self._haversine_distance(term_pos, np) for np in node_positions), default=float('inf')) < COVERAGE_CHECK_DISTANCE_KM
            )
            coverage_percentage = covered_terminals / max(total_terminals, 1) * 100
        else:
            covered_terminals = 0
            coverage_percentage = 0.0
        
        return {
            'recommendations': recommendations,
            'coverage_analysis': {
                'total_terminals': total_terminals,
                'covered_terminals': covered_terminals,
                'coverage_percentage': coverage_percentage,
                'gaps_identified': len(coverage_gaps),
                'overloaded_ground_stations': len(overloaded_ground_stations)
            },
            'traffic_analysis': traffic_analysis['summary'] if traffic_analysis else None
        }
    
    def predict_link_quality_over_time(
        self,
        source_pos: Dict,
        dest_pos: Dict,
        nodes: List[Dict],
        time_horizon_hours: int = 24,
        time_step_minutes: int = 60
    ) -> Dict:
        """
        Dự đoán chất lượng liên kết theo thời gian dựa trên:
        - Vị trí satellite (nếu là satellite)
        - Khoảng cách thay đổi
        - Signal quality
        """
        predictions = []
        
        # Lấy satellites
        satellites = [
            n for n in nodes
            if n.get('nodeType') in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']
            and n.get('isOperational', True)
            and n.get('position')
        ]
        
        current_time = datetime.now()
        
        for step in range(0, time_horizon_hours * 60, time_step_minutes):
            time_point = current_time + timedelta(minutes=step)
            
            # Dự đoán vị trí satellites tại thời điểm này
            best_link = None
            best_quality = 0.0
            
            for sat in satellites:
                # Dự đoán vị trí satellite (simplified - dựa trên orbital parameters)
                predicted_pos = self._predict_satellite_position(
                    sat, time_point
                )
                
                # Tính chất lượng liên kết
                dist_to_source = self._haversine_distance(source_pos, predicted_pos)
                dist_to_dest = self._haversine_distance(predicted_pos, dest_pos)
                total_dist = dist_to_source + dist_to_dest
                
                # Link quality score (lower distance = better)
                # Consider: distance, elevation angle, signal strength
                quality_score = self._calculate_link_quality(
                    source_pos, predicted_pos, dest_pos,
                    sat.get('nodeType'),
                    total_dist
                )
                
                if quality_score > best_quality:
                    best_quality = quality_score
                    best_link = {
                        'satellite_id': sat.get('nodeId'),
                        'satellite_name': sat.get('nodeName', sat.get('nodeId')),
                        'satellite_type': sat.get('nodeType'),
                        'position': predicted_pos,
                        'distance_to_source_km': dist_to_source,
                        'distance_to_dest_km': dist_to_dest,
                        'total_distance_km': total_dist,
                        'quality_score': quality_score,
                        'estimated_latency_ms': self._estimate_latency(total_dist),
                        'estimated_snr_db': self._estimate_snr(total_dist, sat.get('nodeType'))
                    }
            
            if best_link:
                predictions.append({
                    'timestamp': time_point.isoformat(),
                    'time_offset_hours': step / 60.0,
                    'best_link': best_link,
                    'recommendation': self._get_time_recommendation(best_link, step)
                })
        
        # Tìm thời điểm tốt nhất
        if predictions:
            best_time = max(predictions, key=lambda x: x['best_link']['quality_score'])
            worst_time = min(predictions, key=lambda x: x['best_link']['quality_score'])
            
            return {
                'predictions': predictions,
                'summary': {
                    'best_time': best_time['timestamp'],
                    'best_quality': best_time['best_link']['quality_score'],
                    'worst_time': worst_time['timestamp'],
                    'worst_quality': worst_time['best_link']['quality_score'],
                    'average_quality': np.mean([p['best_link']['quality_score'] for p in predictions]),
                    'recommendation': f"Best link quality at {best_time['timestamp']} with {best_time['best_link']['satellite_name']}"
                }
            }
        
        return {'predictions': [], 'summary': {}}
    
    def _haversine_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate Haversine distance in km"""
        if not pos1 or not pos2:
            return float('inf')
        
        lat1, lon1 = math.radians(pos1.get('latitude', 0)), math.radians(pos1.get('longitude', 0))
        lat2, lon2 = math.radians(pos2.get('latitude', 0)), math.radians(pos2.get('longitude', 0))
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        earth_radius_km = EARTH_RADIUS_M / M_TO_KM
        return earth_radius_km * c
    
    def _recommend_node_type(self, position: Dict, existing_nodes: List[Dict]) -> str:
        """Đề xuất loại node dựa trên vị trí và existing nodes"""
        # Đếm số lượng mỗi loại node
        node_types = defaultdict(int)
        for node in existing_nodes:
            if node.get('isOperational', True):
                node_types[node.get('nodeType', 'UNKNOWN')] += 1
        
        # Nếu thiếu ground stations, đề xuất ground station
        if node_types['GROUND_STATION'] < 10:
            return 'GROUND_STATION'
        
        # Nếu thiếu LEO satellites, đề xuất LEO
        if node_types['LEO_SATELLITE'] < 20:
            return 'LEO_SATELLITE'
        
        # Default: LEO satellite (flexible và coverage tốt)
        return 'LEO_SATELLITE'
    
    def _predict_satellite_position(
        self,
        satellite: Dict,
        target_time: datetime
    ) -> Dict:
        """
        Dự đoán vị trí satellite tại thời điểm target_time
        Improved model với orbital mechanics
        """
        current_pos = satellite.get('position', {})
        if not current_pos:
            return current_pos
        
        sat_type = satellite.get('nodeType', 'LEO_SATELLITE')
        
        if sat_type == 'GEO_SATELLITE':
            # GEO satellites are stationary
            return current_pos
        
        # Tính thời gian từ hiện tại (sử dụng time_offset_hours từ prediction)
        time_diff_hours = (target_time - datetime.now()).total_seconds() / 3600.0
        
        if sat_type == 'LEO_SATELLITE':
            base_velocity = LEO_ORBITAL_VELOCITY_DEG_PER_HOUR
            sat_id_hash = hash(satellite.get('nodeId', '')) % 100
            angular_velocity = base_velocity + (sat_id_hash - 50) * 2.0
        else:  # MEO
            base_velocity = MEO_ORBITAL_VELOCITY_DEG_PER_HOUR
            sat_id_hash = hash(satellite.get('nodeId', '')) % 100
            angular_velocity = base_velocity + (sat_id_hash - 50) * 0.5
        
        # Update longitude
        new_lon = current_pos.get('longitude', 0) + angular_velocity * time_diff_hours
        
        # Normalize longitude to [-180, 180]
        new_lon = ((new_lon + 180) % 360) - 180
        
        base_lat = current_pos.get('latitude', 0)
        if sat_type == 'LEO_SATELLITE':
            orbit_phase = (time_diff_hours * 360.0 / LEO_ORBIT_PERIOD_HOURS) % 360
            lat_variation = LEO_LATITUDE_VARIATION_DEG * math.sin(math.radians(orbit_phase))
            new_lat = max(-85, min(85, base_lat + lat_variation))
        else:
            new_lat = base_lat
        
        return {
            'latitude': new_lat,
            'longitude': new_lon,
            'altitude': current_pos.get('altitude', current_pos.get('altitudeKm', 0))
        }
    
    def _calculate_link_quality(
        self,
        source_pos: Dict,
        satellite_pos: Dict,
        dest_pos: Dict,
        sat_type: str,
        total_distance_km: float
    ) -> float:
        """Tính chất lượng liên kết (0-1, higher is better) với realistic variation"""
        # Base quality từ distance (shorter is better)
        max_distance = 40000.0  # Max possible distance (half Earth circumference)
        distance_quality = 1.0 - min(total_distance_km / max_distance, 1.0)
        
        type_bonus = {
            'GEO_SATELLITE': GEO_SATELLITE_TYPE_BONUS,
            'MEO_SATELLITE': MEO_SATELLITE_TYPE_BONUS,
            'LEO_SATELLITE': LEO_SATELLITE_TYPE_BONUS
        }.get(sat_type, 0.1)
        
        # Elevation angle factor (simplified - based on satellite position)
        # Better elevation when satellite is more directly overhead
        sat_lat = satellite_pos.get('latitude', 0)
        source_lat = source_pos.get('latitude', 0)
        dest_lat = dest_pos.get('latitude', 0)
        
        # Average latitude difference (lower is better)
        avg_lat_diff = abs(sat_lat - (source_lat + dest_lat) / 2.0)
        elevation_quality = 1.0 - min(avg_lat_diff / 90.0, 1.0)  # 0-90° range
        
        # Atmospheric conditions (deterministic based on position hash)
        pos_hash = hash(f"{satellite_pos.get('latitude', 0)}_{satellite_pos.get('longitude', 0)}") % 100
        atmospheric_factor = 0.85 + (pos_hash / 100.0) * 0.15
        
        quality = (
            distance_quality * LINK_QUALITY_WEIGHT_DISTANCE +
            elevation_quality * LINK_QUALITY_WEIGHT_ELEVATION +
            type_bonus * LINK_QUALITY_WEIGHT_TYPE +
            atmospheric_factor * LINK_QUALITY_WEIGHT_ATMOSPHERIC
        )
        
        return max(0.0, min(1.0, quality))
    
    def _estimate_latency(self, distance_km: float) -> float:
        """Estimate latency in ms"""
        speed_of_light_km_s = SPEED_OF_LIGHT_MPS / M_TO_KM
        propagation_delay = (distance_km / speed_of_light_km_s) * MS_PER_SECOND
        return propagation_delay + PROCESSING_DELAY_MS
    
    def _estimate_snr(self, distance_km: float, sat_type: str) -> float:
        """Estimate Signal-to-Noise Ratio in dB"""
        base_snr = {
            'GEO_SATELLITE': GEO_BASE_SNR_DB,
            'MEO_SATELLITE': MEO_BASE_SNR_DB,
            'LEO_SATELLITE': LEO_BASE_SNR_DB
        }.get(sat_type, LEO_BASE_SNR_DB)
        
        path_loss_db = 20 * math.log10(distance_km / M_TO_KM)
        snr = base_snr - path_loss_db
        
        return max(0.0, snr)
    
    def _get_time_recommendation(self, link: Dict, time_offset_minutes: int) -> str:
        """Generate recommendation for this time point"""
        quality = link['quality_score']
        latency = link['estimated_latency_ms']
        
        if quality > LINK_QUALITY_EXCELLENT and latency < LINK_LATENCY_EXCELLENT_MS:
            return "Excellent link quality - recommended time for transmission"
        elif quality > LINK_QUALITY_GOOD:
            return "Good link quality - acceptable for transmission"
        elif quality > LINK_QUALITY_MODERATE:
            return "Moderate link quality - consider waiting for better conditions"
        else:
            return "Poor link quality - not recommended for transmission"

