"""
Optimized State Builder for SAGIN Routing
State vector được tối ưu cho hiệu suất và khả năng học, giữ nguyên interface cũ
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RoutingStateBuilder:
    """
    State builder tối ưu với feature engineering tiên tiến, giữ nguyên tên class
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        state_config = self.config.get('state_builder', {})
        
        # Tối ưu kích thước state nhưng giữ nguyên parameter names
        self.max_nodes = state_config.get('max_nodes', 30)  # Cân bằng giữa info và performance
        self.max_terminals = state_config.get('max_terminals', 2)
        
        # Phase 1 Enhancement: Thêm Dijkstra-like features
        # Tăng node_feature_dim từ 12 → 18 để capture đủ thông tin như Dijkstra
        self.node_feature_dim = state_config.get('node_feature_dim', 18)  # Increased from 12
        self.terminal_feature_dim = 6  # Giữ nguyên
        self.global_feature_dim = 8  # Giữ nguyên
        
        # Enable Dijkstra-aware features
        self.include_dijkstra_features = state_config.get('include_dijkstra_features', True)
        
        self.state_dim = (
            self.max_nodes * self.node_feature_dim +
            self.max_terminals * self.terminal_feature_dim +
            self.global_feature_dim
        )
        
        # Cache để tăng tốc tính toán
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
        """Build optimized state vector - giữ nguyên interface"""
        
        # Lọc nodes thông minh với scoring đa tiêu chí
        filtered_nodes = self._smart_node_filtering(
            nodes, source_terminal, dest_terminal, current_node, visited_nodes
        )
        
        # Xây dựng features tối ưu
        node_features = self._build_optimized_node_features(
            filtered_nodes, source_terminal, dest_terminal, current_node, visited_nodes
        )
        
        terminal_features = self._build_optimized_terminal_features(
            source_terminal, dest_terminal, service_qos
        )
        
        global_features = self._build_optimized_global_features(
            nodes, service_qos, topology, scenario, current_node, visited_nodes
        )
        
        # Kết hợp features
        state = np.concatenate([
            node_features.flatten(),
            terminal_features.flatten(), 
            global_features
        ])
        
        # Đảm bảo dimension
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
        """Lọc nodes thông minh với scoring đa tiêu chí"""
        
        operational_nodes = [
            n for n in nodes 
            if n.get('isOperational', True) and n.get('position')
        ]
        
        if not operational_nodes:
            return []
            
        visited_set = set(visited_nodes or [])
        dest_pos = dest_terminal.get('position')
        current_pos = current_node.get('position') if current_node else None
        
        # Tính điểm tổng hợp cho mỗi node
        def compute_node_score(node):
            node_pos = node.get('position')
            if not node_pos:
                return float('inf')
                
            node_id = node.get('nodeId')
            
            # 1. Khoảng cách đến destination (quan trọng nhất)
            dist_to_dest = self._calculate_distance(node_pos, dest_pos)
            
            # 2. Tránh nodes đã visited
            visited_penalty = 10.0 if node_id in visited_set else 0.0
            
            # 3. Chất lượng node (resource, loss rate, etc.)
            quality_score = self._compute_node_quality(node)
            
            # 4. Khoảng cách đến current node (nếu có)
            dist_to_current = 0.0
            if current_pos:
                dist_to_current = self._calculate_distance(current_pos, node_pos)
                # Kiểm tra connectivity (strict for ground stations, lenient for satellites)
                current_node_type = current_node.get('nodeType', '')
                node_type = node.get('nodeType', '')
                
                current_max_range = current_node.get('communication', {}).get('maxRangeKm', 2000) * 1000
                node_max_range = node.get('communication', {}).get('maxRangeKm', 2000) * 1000
                max_range = min(current_max_range, node_max_range)
                
                # Ground stations: strict range (no margin)
                if current_node_type == 'GROUND_STATION' and node_type == 'GROUND_STATION':
                    if dist_to_current > max_range:
                        return float('inf')
                else:
                    # Satellite connections: allow 10% margin for orbital movement
                    if dist_to_current > max_range * 1.1:
                        return float('inf')
            
            # Penalty cho nodes có vấn đề nghiêm trọng trong stress scenarios
            utilization = node.get('resourceUtilization', 0)
            battery = node.get('batteryChargePercent', 100)
            packet_loss = node.get('packetLossRate', 0)
            
            # Tính utilization thực tế dựa trên số terminals connected
            # Nhiều terminals → utilization cao → RL nên tránh
            # SỬ DỤNG CÙNG UTILITY FUNCTION với routing_env để tránh circular import
            try:
                # Import locally để tránh circular import
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from models.database import db
                terminals_collection = db.get_collection('terminals')
                connection_count = terminals_collection.count_documents({
                    'connectedNodeId': node.get('nodeId'),
                    'status': {'$in': ['connected', 'transmitting']}
                })
                # Mỗi terminal thêm ~7% utilization (giống routing_env.py)
                estimated_utilization = min(100.0, utilization + (connection_count * 7.0))
            except Exception as e:
                logger.debug(f"Could not get connection count for {node.get('nodeId')}: {e}")
                estimated_utilization = utilization
                connection_count = 0
            
            problem_penalty = 0.0
            if estimated_utilization > 90:
                problem_penalty += 400000.0  # Tăng từ 300000 → 400000 - RẤT NGUY HIỂM
            elif estimated_utilization > 80:
                problem_penalty += 250000.0  # Tăng từ 200000 → 250000 - Nguy hiểm
            elif estimated_utilization > 70:
                problem_penalty += 120000.0  # Tăng từ 100000 → 120000 - Cảnh báo cao
            
            if battery < 20:
                problem_penalty += 200000.0  # Battery rất thấp
            if packet_loss > 0.1:
                problem_penalty += 100000.0  # Packet loss cao
            
            # Penalty đặc biệt cho GS có quá nhiều terminals (quá tải)
            node_type = node.get('nodeType', '')
            if node_type == 'GROUND_STATION':
                if connection_count > 15:
                    problem_penalty += 200000.0  # Tăng từ 150000 → 200000 - GS QUÁ TẢI
                elif connection_count > 10:
                    problem_penalty += 100000.0  # Tăng từ 80000 → 100000 - GS tải cao
                elif connection_count <= 3:
                    problem_penalty -= 80000.0  # Tăng bonus từ -50000 → -80000 cho GS ít tải
            
            # Bonus/penalty cho node type - CÂN BẰNG hơn
            type_bonus = 0.0
            
            # Ưu tiên satellites khi current node là ground station (tránh đi qua nhiều ground stations)
            if current_node and current_node.get('nodeType') == 'GROUND_STATION':
                if node_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']:
                    type_bonus = -80000.0  # GIẢM từ -150000 → -80000 (vừa phải)
                    if node_type == 'LEO_SATELLITE':
                        type_bonus = -100000.0  # GIẢM từ -200000 → -100000 (LEO tốt hơn)
                elif node_type == 'GROUND_STATION':
                    # CHỈ penalty nếu current GS KHÔNG quá tải
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
                    
                    current_utilization = current_node.get('resourceUtilization', 0) + (current_connection_count * 7.0)
                    if current_utilization < 70.0:
                        # Current GS ok, penalty cho việc đi sang GS khác
                        type_bonus = 80000.0  # Tăng từ 50000 → 80000
                    else:
                        # Current GS quá tải, CHO PHÉP đi sang GS khác (không penalty)
                        type_bonus = 0.0
                        # Bonus nếu next GS ít tải hơn
                        if estimated_utilization < current_utilization - 20.0:
                            type_bonus = -100000.0  # BONUS cho load balancing
            

            score = (
                dist_to_dest * 0.7 +           # TĂNG: 0.5 → 0.7 - Ưu tiên MẠNH gần destination
                dist_to_current * 0.1 +        # GIẢM: 0.2 → 0.1 - Không quá quan tâm current
                visited_penalty * 1000000 +    # Tránh visited nodes
                (1 - quality_score) * 500000 +  # Ưu tiên node chất lượng cao
                problem_penalty +              # Tránh nodes có vấn đề nghiêm trọng
                type_bonus                     # Bonus/penalty cho node type
            )
            
            return score
        
        # Sắp xếp theo score
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
            
            # ===== EXISTING FEATURES (0-11) =====
            # 1. Resource features
            features[i, 0] = node.get('resourceUtilization', 0) / 100.0
            capacity = max(node.get('packetBufferCapacity', 1000), 1)
            features[i, 1] = min(node.get('currentPacketCount', 0) / capacity, 1.0)
            features[i, 2] = min(node.get('packetLossRate', 0), 1.0)
            features[i, 3] = node.get('batteryChargePercent', 100) / 100.0
            
            # 2. Performance features
            features[i, 4] = min(node.get('nodeProcessingDelayMs', 0) / 50.0, 1.0)
            bandwidth = node.get('communication', {}).get('bandwidth', 100)
            features[i, 5] = min(bandwidth / 1000.0, 1.0)
            
            # 3. Status features
            features[i, 6] = 1.0 if node.get('isOperational', True) else 0.0
            features[i, 7] = 1.0 if node_id in visited_set else 0.0
            
            # 4. Position features
            if dest_pos and pos:
                dist_to_dest = self._calculate_distance(pos, dest_pos)
                features[i, 8] = min(dist_to_dest / 20000000.0, 1.0)
            
            if current_pos and pos:
                dist_to_current = self._calculate_distance(current_pos, pos)
                features[i, 9] = min(dist_to_current / 1000000.0, 1.0)
            
            # 5. Node type (legacy encoding - will be replaced by one-hot)
            node_type = node.get('nodeType', 'UNKNOWN')
            type_encoding = {
                'SATELLITE': 0.2,
                'AERIAL': 0.5, 
                'GROUND_STATION': 0.8
            }
            features[i, 10] = type_encoding.get(node_type, 0.0)
            
            # 6. Quality score tổng hợp
            features[i, 11] = self._compute_node_quality(node)
            
            # ===== NEW DIJKSTRA-LIKE FEATURES (12-17) =====
            if self.include_dijkstra_features and self.node_feature_dim >= 18:
                # 12. Separate CPU utilization (match Dijkstra's get_node_utilization)
                cpu_util = node.get('cpu', {}).get('utilization', 0) / 100.0
                features[i, 12] = cpu_util
                
                # 13. Separate Memory utilization
                mem_util = node.get('memory', {}).get('utilization', 0) / 100.0
                features[i, 13] = mem_util
                
                # 14. Separate Bandwidth utilization
                bw_util = node.get('bandwidth', {}).get('utilization', 0) / 100.0
                features[i, 14] = bw_util
                
                # 15. Max utilization (like Dijkstra's max(cpu, mem, bw))
                max_util = max(cpu_util, mem_util, bw_util)
                features[i, 15] = max_util
                
                # 16. Dijkstra edge weight estimate (normalized)
                if current_node and current_pos and pos:
                    dijkstra_weight = self._estimate_dijkstra_edge_weight(
                        current_node, node, current_pos, pos
                    )
                    features[i, 16] = min(dijkstra_weight / 10000.0, 10.0)  # Cap at 10.0
                else:
                    features[i, 16] = 0.0
                
                # 17. Distance to source (normalized)
                if source_pos and pos:
                    dist_to_source = self._calculate_distance(pos, source_pos)
                    features[i, 17] = min(dist_to_source / 20000000.0, 1.0)
                else:
                    features[i, 17] = 0.0
            
        return features
    
    def _estimate_dijkstra_edge_weight(
        self,
        current_node: Dict,
        next_node: Dict,
        current_pos: Dict,
        next_pos: Dict,
        drop_threshold: float = 95.0,
        penalty_threshold: float = 80.0,
        penalty_multiplier: float = 3.0
    ) -> float:
        """
        Estimate Dijkstra edge weight cho next_node
        Match logic từ calculate_path_dijkstra()
        """
        # Base distance (in km)
        distance_m = self._calculate_distance(current_pos, next_pos)
        base_distance_km = distance_m / 1000.0
        
        # Get max utilization (like Dijkstra's get_node_utilization)
        cpu = next_node.get('cpu', {}).get('utilization', 0)
        mem = next_node.get('memory', {}).get('utilization', 0)
        bw = next_node.get('bandwidth', {}).get('utilization', 0)
        max_util = max(cpu, mem, bw)
        
        # Drop check (match drop_threshold = 95%)
        if max_util >= drop_threshold:
            return float('inf')  # Effectively drop node
        
        # Resource penalty (match penalty_threshold = 80%, multiplier = 3.0x)
        if max_util >= penalty_threshold:
            excess = (max_util - penalty_threshold) / (100 - penalty_threshold)
            penalty = base_distance_km * (penalty_multiplier - 1.0) * excess
            return base_distance_km + penalty
        
        # No penalty
        return base_distance_km
    
    def _build_optimized_terminal_features(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        service_qos: Optional[Dict]
    ) -> np.ndarray:
        """Xây dựng features terminal tối ưu"""
        features = np.zeros((self.max_terminals, self.terminal_feature_dim))
        
        # Source terminal
        source_pos = source_terminal.get('position', {})
        features[0, 0] = source_pos.get('latitude', 0) / 90.0
        features[0, 1] = source_pos.get('longitude', 0) / 180.0
        features[0, 2] = min(source_pos.get('altitude', 0) / 50000.0, 1.0)
        
        # Destination terminal  
        dest_pos = dest_terminal.get('position', {})
        features[1, 0] = dest_pos.get('latitude', 0) / 90.0
        features[1, 1] = dest_pos.get('longitude', 0) / 180.0
        features[1, 2] = min(dest_pos.get('altitude', 0) / 50000.0, 1.0)
        
        # QoS features
        if service_qos:
            features[0, 3] = min(service_qos.get('maxLatencyMs', 1000) / 1000.0, 1.0)
            features[0, 4] = min(service_qos.get('minBandwidthMbps', 10) / 100.0, 1.0)
            features[0, 5] = min(service_qos.get('maxLossRate', 0.1) / 0.1, 1.0)
        else:
            features[0, 3] = 1.0  # No latency constraint
            features[0, 4] = 0.1  # Default bandwidth
            features[0, 5] = 1.0  # No loss constraint
            
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
        """Xây dựng global features tối ưu"""
        features = np.zeros(self.global_feature_dim)
        
        # Network health metrics
        if nodes:
            operational_nodes = [n for n in nodes if n.get('isOperational', True)]
            if operational_nodes:
                features[0] = np.mean([n.get('resourceUtilization', 0) for n in operational_nodes]) / 100.0
                features[1] = np.mean([min(n.get('packetLossRate', 0), 1.0) for n in operational_nodes])
                
                # Network congestion indicator
                congested_nodes = [n for n in operational_nodes if n.get('resourceUtilization', 0) > 80]
                features[2] = len(congested_nodes) / max(len(operational_nodes), 1)
                
                features[3] = len(operational_nodes) / max(len(nodes), 1)
        
        # Current node context
        if current_node:
            features[4] = current_node.get('resourceUtilization', 0) / 100.0
            features[5] = min(current_node.get('packetLossRate', 0), 1.0)
            
            # Progress indicator (dựa trên số nodes visited)
            if visited_nodes:
                features[6] = min(len(visited_nodes) / 10.0, 1.0)
        
        # Scenario awareness
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
        """Tính chất lượng tổng hợp của node (0-1, higher = better)"""
        cache_key = node.get('nodeId')
        if cache_key in self._node_quality_cache:
            return self._node_quality_cache[cache_key]
        
        # Tính score chất lượng dựa trên multiple factors
        utilization = node.get('resourceUtilization', 0) / 100.0
        loss_rate = min(node.get('packetLossRate', 0), 1.0)
        battery = node.get('batteryChargePercent', 100) / 100.0
        delay = min(node.get('nodeProcessingDelayMs', 0) / 50.0, 1.0)
        
        # Kết hợp scores (higher = better)
        quality_score = (
            (1 - utilization) * 0.3 +      # Resource availability
            (1 - loss_rate) * 0.3 +        # Reliability
            battery * 0.2 +                # Energy
            (1 - delay) * 0.2              # Performance
        )
        
        self._node_quality_cache[cache_key] = quality_score
        return quality_score
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Tính distance với cache để tăng tốc"""
        cache_key = tuple(sorted([str(pos1), str(pos2)]))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        if not pos1 or not pos2:
            return float('inf')
        
        # Haversine calculation
        lat1 = math.radians(pos1.get('latitude', 0))
        lon1 = math.radians(pos1.get('longitude', 0))
        lat2 = math.radians(pos2.get('latitude', 0))
        lon2 = math.radians(pos2.get('longitude', 0))
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        R = 6371000  # Earth radius in meters
        horizontal_dist = R * c
        
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