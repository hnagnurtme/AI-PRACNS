"""
State Builder for SAGIN Routing
Builds state vector for RL agent with Dijkstra-like features
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
import logging
from collections import defaultdict

from environment.constants import (
    EARTH_RADIUS_M, M_TO_KM, DIST_NORM_MAX_M, DIST_NORM_CURRENT_M,
    DIST_NORM_EDGE_WEIGHT_KM, UTILIZATION_MAX_PERCENT,
    UTILIZATION_CRITICAL_PERCENT, UTILIZATION_HIGH_PERCENT,
    UTILIZATION_MEDIUM_PERCENT, TERMINAL_UTILIZATION_IMPACT,
    BATTERY_LOW_PERCENT, BATTERY_MAX_PERCENT, PACKET_LOSS_HIGH,
    GS_CONNECTION_OVERLOADED, GS_CONNECTION_HIGH, GS_CONNECTION_LOW,
    SCORE_WEIGHT_DIST_TO_DEST, SCORE_WEIGHT_DIST_TO_CURRENT,
    SCORE_WEIGHT_VISITED_PENALTY, SCORE_WEIGHT_QUALITY,
    BONUS_LEO_SATELLITE, BONUS_SATELLITE, BONUS_GS_LOAD_BALANCING,
    PENALTY_GS_TO_GS, PENALTY_GS_OVERLOADED, PENALTY_GS_HIGH_LOAD,
    BONUS_GS_LOW_LOAD, PENALTY_CRITICAL_UTILIZATION,
    PENALTY_HIGH_UTILIZATION, PENALTY_MEDIUM_UTILIZATION,
    PENALTY_LOW_BATTERY, PENALTY_HIGH_PACKET_LOSS,
    NORM_UTILIZATION, NORM_PACKET_BUFFER, NORM_PROCESSING_DELAY_MS,
    NORM_BANDWIDTH_MBPS, NORM_ALTITUDE_M, NORM_LATENCY_MS,
    NORM_BANDWIDTH_REQ_MBPS, NORM_LOSS_RATE, NORM_EDGE_WEIGHT_KM,
    NODE_TYPE_SATELLITE, NODE_TYPE_AERIAL, NODE_TYPE_GROUND_STATION,
    QUALITY_WEIGHT_RESOURCE, QUALITY_WEIGHT_RELIABILITY,
    QUALITY_WEIGHT_ENERGY, QUALITY_WEIGHT_PERFORMANCE,
    DEFAULT_MAX_RANGE_KM, SATELLITE_RANGE_MARGIN, GS_RANGE_MARGIN,
    DIJKSTRA_DROP_THRESHOLD, DIJKSTRA_PENALTY_THRESHOLD,
    DIJKSTRA_PENALTY_MULTIPLIER
)

logger = logging.getLogger(__name__)


class RoutingStateBuilder:
    """
    State builder tối ưu với feature engineering tiên tiến, giữ nguyên tên class
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        state_config = self.config.get('state_builder', {})
        
        self.max_nodes = state_config.get('max_nodes', 30)
        self.max_terminals = state_config.get('max_terminals', 2)
        self.node_feature_dim = state_config.get('node_feature_dim', 18)
        self.terminal_feature_dim = 6
        self.global_feature_dim = 8
        self.include_dijkstra_features = state_config.get('include_dijkstra_features', True)
        
        self.state_dim = (
            self.max_nodes * self.node_feature_dim +
            self.max_terminals * self.terminal_feature_dim +
            self.global_feature_dim
        )
        
        self._distance_cache = {}
        self._node_quality_cache = {}
    
    def build_state(
        self,
        nodes: List[Dict],
        source_terminal: Dict,
        dest_terminal: Dict,
        current_node: Optional[Dict] = None,
        service_qos: Optional[Dict] = None,
        topology: Optional[Dict] = None,
        scenario: Optional[Dict] = None,
        visited_nodes: Optional[List[str]] = None
    ) -> np.ndarray:
        """Build state vector for RL agent"""
        filtered_nodes = self._smart_node_filtering(
            nodes, source_terminal, dest_terminal, current_node, visited_nodes
        )
        
        node_features = self._build_optimized_node_features(
            filtered_nodes, source_terminal, dest_terminal, current_node, visited_nodes
        )
        
        terminal_features = self._build_optimized_terminal_features(
            source_terminal, dest_terminal, service_qos
        )
        
        global_features = self._build_optimized_global_features(
            nodes, service_qos, topology, scenario, current_node, visited_nodes
        )
        
        state = np.concatenate([
            node_features.flatten(),
            terminal_features.flatten(), 
            global_features
        ])
        
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
            
        return state.astype(np.float32)
    
    def _smart_node_filtering(
        self,
        nodes: List[Dict],
        source_terminal: Dict,
        dest_terminal: Dict,
        current_node: Optional[Dict] = None,
        visited_nodes: Optional[List[str]] = None
    ) -> List[Dict]:
        """Filter and score nodes for routing"""
        operational_nodes = [
            n for n in nodes 
            if n.get('isOperational', True) and n.get('position')
        ]
        
        if not operational_nodes:
            return []
            
        visited_set = set(visited_nodes or [])
        dest_pos = dest_terminal.get('position')
        current_pos = current_node.get('position') if current_node else None
        
        def compute_node_score(node):
            node_pos = node.get('position')
            if not node_pos:
                return float('inf')
                
            node_id = node.get('nodeId')
            dist_to_dest = self._calculate_distance(node_pos, dest_pos)
            visited_penalty = 10.0 if node_id in visited_set else 0.0
            quality_score = self._compute_node_quality(node)
            
            dist_to_current = 0.0
            if current_pos:
                dist_to_current = self._calculate_distance(current_pos, node_pos)
                current_node_type = current_node.get('nodeType', '')
                node_type = node.get('nodeType', '')
                
                default_range_m = DEFAULT_MAX_RANGE_KM * M_TO_KM
                current_max_range = current_node.get('communication', {}).get('maxRangeKm', DEFAULT_MAX_RANGE_KM) * M_TO_KM
                node_max_range = node.get('communication', {}).get('maxRangeKm', DEFAULT_MAX_RANGE_KM) * M_TO_KM
                max_range = min(current_max_range, node_max_range)
                
                if current_node_type == 'GROUND_STATION' and node_type == 'GROUND_STATION':
                    if dist_to_current > max_range * GS_RANGE_MARGIN:
                        return float('inf')
                else:
                    if dist_to_current > max_range * SATELLITE_RANGE_MARGIN:
                        return float('inf')
            
            utilization = node.get('resourceUtilization', 0)
            battery = node.get('batteryChargePercent', BATTERY_MAX_PERCENT)
            packet_loss = node.get('packetLossRate', 0)
            
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from models.database import db
                terminals_collection = db.get_collection('terminals')
                connection_count = terminals_collection.count_documents({
                    'connectedNodeId': node.get('nodeId'),
                    'status': {'$in': ['connected', 'transmitting']}
                })
                estimated_utilization = min(UTILIZATION_MAX_PERCENT, utilization + (connection_count * TERMINAL_UTILIZATION_IMPACT))
            except Exception as e:
                logger.debug(f"Could not get connection count for {node.get('nodeId')}: {e}")
                estimated_utilization = utilization
                connection_count = 0
            
            problem_penalty = 0.0
            if estimated_utilization >= UTILIZATION_CRITICAL_PERCENT:
                problem_penalty += PENALTY_CRITICAL_UTILIZATION
            elif estimated_utilization >= UTILIZATION_HIGH_PERCENT:
                problem_penalty += PENALTY_HIGH_UTILIZATION
            elif estimated_utilization >= UTILIZATION_MEDIUM_PERCENT:
                problem_penalty += PENALTY_MEDIUM_UTILIZATION
            
            if battery < BATTERY_LOW_PERCENT:
                problem_penalty += PENALTY_LOW_BATTERY
            if packet_loss > PACKET_LOSS_HIGH:
                problem_penalty += PENALTY_HIGH_PACKET_LOSS
            
            node_type = node.get('nodeType', '')
            if node_type == 'GROUND_STATION':
                if connection_count > GS_CONNECTION_OVERLOADED:
                    problem_penalty += PENALTY_GS_OVERLOADED
                elif connection_count > GS_CONNECTION_HIGH:
                    problem_penalty += PENALTY_GS_HIGH_LOAD
                elif connection_count <= GS_CONNECTION_LOW:
                    problem_penalty += BONUS_GS_LOW_LOAD
            
            type_bonus = 0.0
            if current_node and current_node.get('nodeType') == 'GROUND_STATION':
                if node_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']:
                    type_bonus = BONUS_SATELLITE
                    if node_type == 'LEO_SATELLITE':
                        type_bonus = BONUS_LEO_SATELLITE
                elif node_type == 'GROUND_STATION':
                    current_connection_count = 0
                    try:
                        from models.database import db
                        terminals_collection = db.get_collection('terminals')
                        current_connection_count = terminals_collection.count_documents({
                            'connectedNodeId': current_node.get('nodeId'),
                            'status': {'$in': ['connected', 'transmitting']}
                        })
                    except:
                        pass
                    
                    current_utilization = current_node.get('resourceUtilization', 0) + (current_connection_count * TERMINAL_UTILIZATION_IMPACT)
                    if current_utilization < UTILIZATION_MEDIUM_PERCENT:
                        type_bonus = PENALTY_GS_TO_GS
                    else:
                        type_bonus = 0.0
                        if estimated_utilization < current_utilization - 20.0:
                            type_bonus = BONUS_GS_LOAD_BALANCING

            score = (
                dist_to_dest * SCORE_WEIGHT_DIST_TO_DEST +
                dist_to_current * SCORE_WEIGHT_DIST_TO_CURRENT +
                visited_penalty * SCORE_WEIGHT_VISITED_PENALTY +
                (1 - quality_score) * SCORE_WEIGHT_QUALITY +
                problem_penalty +
                type_bonus
            )
            
            return score
        
        operational_nodes.sort(key=compute_node_score)
        return operational_nodes[:self.max_nodes]
    
    def _build_optimized_node_features(
        self,
        nodes: List[Dict],
        source_terminal: Dict,
        dest_terminal: Dict,
        current_node: Optional[Dict],
        visited_nodes: Optional[List[str]]
    ) -> np.ndarray:
        """
        Xây dựng features node tối ưu với Dijkstra-like features (Phase 1 Enhancement)
        Features: 18 dimensions (increased from 12)
        """
        features = np.zeros((self.max_nodes, self.node_feature_dim))
        
        source_pos = source_terminal.get('position')
        dest_pos = dest_terminal.get('position')
        current_pos = current_node.get('position') if current_node else None
        visited_set = set(visited_nodes or [])
        
        for i, node in enumerate(nodes):
            if i >= self.max_nodes:
                break
                
            pos = node.get('position', {})
            node_id = node.get('nodeId')
            
            features[i, 0] = node.get('resourceUtilization', 0) / NORM_UTILIZATION
            capacity = max(node.get('packetBufferCapacity', NORM_PACKET_BUFFER), 1)
            features[i, 1] = min(node.get('currentPacketCount', 0) / capacity, 1.0)
            features[i, 2] = min(node.get('packetLossRate', 0), 1.0)
            features[i, 3] = node.get('batteryChargePercent', BATTERY_MAX_PERCENT) / BATTERY_MAX_PERCENT
            
            features[i, 4] = min(node.get('nodeProcessingDelayMs', 0) / NORM_PROCESSING_DELAY_MS, 1.0)
            bandwidth = node.get('communication', {}).get('bandwidth', 100)
            features[i, 5] = min(bandwidth / NORM_BANDWIDTH_MBPS, 1.0)
            
            features[i, 6] = 1.0 if node.get('isOperational', True) else 0.0
            features[i, 7] = 1.0 if node_id in visited_set else 0.0
            
            if dest_pos and pos:
                dist_to_dest = self._calculate_distance(pos, dest_pos)
                features[i, 8] = min(dist_to_dest / DIST_NORM_MAX_M, 1.0)
            
            if current_pos and pos:
                dist_to_current = self._calculate_distance(current_pos, pos)
                features[i, 9] = min(dist_to_current / DIST_NORM_CURRENT_M, 1.0)
            
            node_type = node.get('nodeType', 'UNKNOWN')
            type_encoding = {
                'SATELLITE': NODE_TYPE_SATELLITE,
                'AERIAL': NODE_TYPE_AERIAL, 
                'GROUND_STATION': NODE_TYPE_GROUND_STATION
            }
            features[i, 10] = type_encoding.get(node_type, 0.0)
            
            features[i, 11] = self._compute_node_quality(node)
            
            if self.include_dijkstra_features and self.node_feature_dim >= 18:
                cpu_util = node.get('cpu', {}).get('utilization', 0) / NORM_UTILIZATION
                features[i, 12] = cpu_util
                
                mem_util = node.get('memory', {}).get('utilization', 0) / NORM_UTILIZATION
                features[i, 13] = mem_util
                
                bw_util = node.get('bandwidth', {}).get('utilization', 0) / NORM_UTILIZATION
                features[i, 14] = bw_util
                
                max_util = max(cpu_util, mem_util, bw_util)
                features[i, 15] = max_util
                
                if current_node and current_pos and pos:
                    dijkstra_weight = self._estimate_dijkstra_edge_weight(
                        current_node, node, current_pos, pos
                    )
                    features[i, 16] = min(dijkstra_weight / DIST_NORM_EDGE_WEIGHT_KM, NORM_EDGE_WEIGHT_KM)
                else:
                    features[i, 16] = 0.0
                
                if source_pos and pos:
                    dist_to_source = self._calculate_distance(pos, source_pos)
                    features[i, 17] = min(dist_to_source / DIST_NORM_MAX_M, 1.0)
                else:
                    features[i, 17] = 0.0
            
        return features
    
    def _estimate_dijkstra_edge_weight(
        self,
        current_node: Dict,
        next_node: Dict,
        current_pos: Dict,
        next_pos: Dict,
        drop_threshold: float = DIJKSTRA_DROP_THRESHOLD,
        penalty_threshold: float = DIJKSTRA_PENALTY_THRESHOLD,
        penalty_multiplier: float = DIJKSTRA_PENALTY_MULTIPLIER
    ) -> float:
        """Estimate Dijkstra edge weight for next_node"""
        distance_m = self._calculate_distance(current_pos, next_pos)
        base_distance_km = distance_m / M_TO_KM
        
        cpu = next_node.get('cpu', {}).get('utilization', 0)
        mem = next_node.get('memory', {}).get('utilization', 0)
        bw = next_node.get('bandwidth', {}).get('utilization', 0)
        max_util = max(cpu, mem, bw)
        
        if max_util >= drop_threshold:
            return float('inf')
        
        if max_util >= penalty_threshold:
            excess = (max_util - penalty_threshold) / (UTILIZATION_MAX_PERCENT - penalty_threshold)
            penalty = base_distance_km * (penalty_multiplier - 1.0) * excess
            return base_distance_km + penalty
        
        return base_distance_km
    
    def _build_optimized_terminal_features(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        service_qos: Optional[Dict]
    ) -> np.ndarray:
        """Build terminal features"""
        features = np.zeros((self.max_terminals, self.terminal_feature_dim))
        
        source_pos = source_terminal.get('position', {})
        features[0, 0] = source_pos.get('latitude', 0) / 90.0
        features[0, 1] = source_pos.get('longitude', 0) / 180.0
        features[0, 2] = min(source_pos.get('altitude', 0) / NORM_ALTITUDE_M, 1.0)
        
        dest_pos = dest_terminal.get('position', {})
        features[1, 0] = dest_pos.get('latitude', 0) / 90.0
        features[1, 1] = dest_pos.get('longitude', 0) / 180.0
        features[1, 2] = min(dest_pos.get('altitude', 0) / NORM_ALTITUDE_M, 1.0)
        
        if service_qos:
            features[0, 3] = min(service_qos.get('maxLatencyMs', NORM_LATENCY_MS) / NORM_LATENCY_MS, 1.0)
            features[0, 4] = min(service_qos.get('minBandwidthMbps', 10) / NORM_BANDWIDTH_REQ_MBPS, 1.0)
            features[0, 5] = min(service_qos.get('maxLossRate', NORM_LOSS_RATE) / NORM_LOSS_RATE, 1.0)
        else:
            features[0, 3] = 1.0
            features[0, 4] = 0.1
            features[0, 5] = 1.0
            
        return features
    
    def _build_optimized_global_features(
        self,
        nodes: List[Dict],
        service_qos: Optional[Dict],
        topology: Optional[Dict],
        scenario: Optional[Dict],
        current_node: Optional[Dict],
        visited_nodes: Optional[List[str]]
    ) -> np.ndarray:
        """Build global features"""
        features = np.zeros(self.global_feature_dim)
        
        if nodes:
            operational_nodes = [n for n in nodes if n.get('isOperational', True)]
            if operational_nodes:
                features[0] = np.mean([n.get('resourceUtilization', 0) for n in operational_nodes]) / NORM_UTILIZATION
                features[1] = np.mean([min(n.get('packetLossRate', 0), 1.0) for n in operational_nodes])
                
                congested_nodes = [n for n in operational_nodes if n.get('resourceUtilization', 0) > UTILIZATION_HIGH_PERCENT]
                features[2] = len(congested_nodes) / max(len(operational_nodes), 1)
                
                features[3] = len(operational_nodes) / max(len(nodes), 1)
        
        if current_node:
            features[4] = current_node.get('resourceUtilization', 0) / NORM_UTILIZATION
            features[5] = min(current_node.get('packetLossRate', 0), 1.0)
            
            if visited_nodes:
                features[6] = min(len(visited_nodes) / 10.0, 1.0)
        
        if scenario:
            scenario_type = scenario.get('scenarioType', 'NORMAL')
            if scenario_type == 'CONGESTION':
                features[7] = 1.0
            elif scenario_type == 'NODE_FAILURE':
                features[7] = 0.5
            else:
                features[7] = 0.0
                
        return features
    
    def _compute_node_quality(self, node: Dict) -> float:
        """Compute node quality score (0-1, higher = better)"""
        cache_key = node.get('nodeId')
        if cache_key in self._node_quality_cache:
            return self._node_quality_cache[cache_key]
        
        utilization = node.get('resourceUtilization', 0) / NORM_UTILIZATION
        loss_rate = min(node.get('packetLossRate', 0), 1.0)
        battery = node.get('batteryChargePercent', BATTERY_MAX_PERCENT) / BATTERY_MAX_PERCENT
        delay = min(node.get('nodeProcessingDelayMs', 0) / NORM_PROCESSING_DELAY_MS, 1.0)
        
        quality_score = (
            (1 - utilization) * QUALITY_WEIGHT_RESOURCE +
            (1 - loss_rate) * QUALITY_WEIGHT_RELIABILITY +
            battery * QUALITY_WEIGHT_ENERGY +
            (1 - delay) * QUALITY_WEIGHT_PERFORMANCE
        )
        
        self._node_quality_cache[cache_key] = quality_score
        return quality_score
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate distance with caching"""
        cache_key = tuple(sorted([str(pos1), str(pos2)]))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        if not pos1 or not pos2:
            return float('inf')
        
        lat1 = math.radians(pos1.get('latitude', 0))
        lon1 = math.radians(pos1.get('longitude', 0))
        lat2 = math.radians(pos2.get('latitude', 0))
        lon2 = math.radians(pos2.get('longitude', 0))
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        horizontal_dist = EARTH_RADIUS_M * c
        
        alt1 = pos1.get('altitude', 0)
        alt2 = pos2.get('altitude', 0)
        vertical_dist = abs(alt1 - alt2)
        
        distance = math.sqrt(horizontal_dist**2 + vertical_dist**2)
        self._distance_cache[cache_key] = distance
        return distance
    
    @property
    def state_dimension(self) -> int:
        """Get state dimension - giữ nguyên property"""
        return self.state_dim