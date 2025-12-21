"""
Optimized RL Routing Service
Sử dụng DuelingDQN agent tối ưu để routing với performance cao
"""
import os
import sys
import numpy as np
import torch
from typing import Dict, List, Optional
import logging
from pathlib import Path
import time
from functools import lru_cache

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from agent.dueling_dqn import DuelingDQNAgent
from environment.state_builder import RoutingStateBuilder
from environment.routing_env import RoutingEnvironment
from environment.constants import (
    BATTERY_MAX_PERCENT,
    NORM_PACKET_BUFFER,
    M_TO_KM
)

logger = logging.getLogger(__name__)


class RLRoutingService:
    """Optimized service để routing sử dụng DuelingDQN agent"""

    def __init__(self, config: Dict = None):
        self.config = config or Config.get_yaml_config()

        # Initialize state builder
        self.state_builder = RoutingStateBuilder(self.config)

        # Initialize RL agent (lazy loading)
        self.agent = None
        self.agent_loaded = False
        self.state_dim = self.state_builder.state_dimension
        
        # Cache để tăng performance
        self._model_cache = {}
        self._node_cache = {}
        self._qos_cache = {}
        
        # Performance monitoring
        self.request_count = 0
        self.avg_response_time = 0.0
    
    def _load_agent(self, nodes: List[Dict]):
        """Optimized lazy load RL agent"""
        if not self.agent_loaded:
            try:
                # Determine action dimension
                max_actions = min(len(nodes), self.state_builder.max_nodes)
                action_dim = max(max_actions, 1)
                
                # Initialize agent
                self.agent = DuelingDQNAgent(
                    state_dim=self.state_dim,
                    action_dim=action_dim,
                    config=self.config
                )
                
                # Try to load best model với cache
                model_loaded = self._load_best_model()
                
                if not model_loaded:
                    logger.warning("RL Model not found, using random policy")
                
                self.agent.eval()
                self.agent_loaded = True
                
                logger.info("RL Agent loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading RL agent: {e}", exc_info=True)
                self.agent = None
                self.agent_loaded = True
    
    def _load_best_model(self) -> bool:
        """Load best model với caching và fallbacks"""
        model_path = self.config.get('rl_agent', {}).get('model_path', './models/rl_agent')
        model_path = Path(model_path)
        
        # Danh sách model files để thử, ưu tiên best model
        model_files = [
            model_path.parent / 'best_models' / 'best_model.pt',
            model_path / 'best_model.pt',
            model_path / 'final_model.pt',
            model_path / 'model.pt',
        ]
        
        for model_file in model_files:
            if model_file.exists() and model_file.is_file():
                try:
                    # Check cache
                    cache_key = str(model_file)
                    if cache_key in self._model_cache:
                        self.agent.load_state_dict(self._model_cache[cache_key])
                    else:
                        self.agent.load(str(model_file))
                        # Cache model state
                        self._model_cache[cache_key] = self.agent.get_state_dict()
                    
                    logger.info(f"RL Model loaded from {model_file}")
                    return True
                    
                except Exception as e:
                    logger.debug(f"Failed to load from {model_file}: {e}")
                    continue
        
        return False
    
    def calculate_path_rl(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict],
        service_qos: Optional[Dict] = None,
        topology: Optional[Dict] = None,
        scenario: Optional[Dict] = None
    ) -> Dict:
        """
        Optimized path calculation với performance monitoring
        
        ⚠️ LƯU Ý: RL Routing Service hiện tại còn YẾU KÉM so với Dijkstra.
        Xem chi tiết trong calculate_path_rl() ở api/routing_bp.py để biết lý do.
        
        Các vấn đề chính:
        - Phụ thuộc vào model quality và training
        - Có giới hạn max_steps (6-8) so với Dijkstra không giới hạn
        - Không đảm bảo optimality như Dijkstra
        - Có thể fail và cần fallback
        - Phức tạp hơn trong debug và maintenance
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            processed_nodes = self._preprocess_nodes(nodes, service_qos)
            
            # 🔍 DEBUG: Log node count and resource states
            gs_nodes = [n for n in processed_nodes if n.get('nodeType') == 'GROUND_STATION']
            logger.info(f"🔍 RL using {len(processed_nodes)} nodes ({len(gs_nodes)} ground stations)")
            for gs in gs_nodes[:5]:  # Log first 5 GS
                logger.info(f"   GS {gs.get('nodeId')}: util={gs.get('resourceUtilization', 0):.1f}%, loss={gs.get('packetLossRate', 0)*100:.2f}%")
            
            if not processed_nodes:
                logger.error("❌ No valid nodes after QoS filtering - cannot route")
                raise ValueError("No valid nodes available for routing")
            
            # Load agent nếu cần
            self._load_agent(processed_nodes)
            
            # Nếu agent không available, FAIL (không fallback)
            if self.agent is None:
                logger.error("❌ RL Agent not loaded - please train model first")
                raise RuntimeError("RL model not available. Please train the model using: python training/train.py")
            
            # Tính path bằng RL
            path = self._calculate_rl_path(
                source_terminal, dest_terminal, processed_nodes, service_qos
            )
            
            # Validate và enhance path
            enhanced_path = self._enhance_path_metrics(path, processed_nodes, service_qos)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time)
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"❌ RL routing failed: {e}", exc_info=True)
            # Re-raise error để endpoint biết có lỗi, KHÔNG fallback
            raise RuntimeError(f"RL routing failed: {e}") from e
    
    def _calculate_rl_path(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict],
        service_qos: Optional[Dict]
    ) -> Dict:
        """Calculate path using RL agent with terminal→GS→satellite→GS→terminal logic"""
        
        from api.routing_bp import find_best_ground_station, calculate_distance
        
        source_gs = find_best_ground_station(source_terminal, nodes)
        dest_gs = find_best_ground_station(dest_terminal, nodes)
        
        if not source_gs or not dest_gs:
            logger.error(f"❌ RL: Cannot find ground stations for terminals")
            raise ValueError("No suitable ground stations found for terminals")
        
        if source_gs:
            source_distance_km = calculate_distance(source_terminal.get('position'), source_gs.get('position')) / M_TO_KM
            logger.info(
                f"🤖 RL: Using BEST Ground Station {source_gs['nodeId']} "
                f"for terminal {source_terminal.get('terminalId')} "
                f"(distance: {source_distance_km:.1f}km, "
                f"utilization: {source_gs.get('resourceUtilization', 0):.1f}%, "
                f"battery: {source_gs.get('batteryChargePercent', BATTERY_MAX_PERCENT):.1f}%, "
                f"packet_loss: {source_gs.get('packetLossRate', 0)*100:.2f}% - "
                f"RESOURCE-OPTIMIZED selection)"
            )
        
        if dest_gs:
            dest_distance_km = calculate_distance(dest_terminal.get('position'), dest_gs.get('position')) / M_TO_KM
            logger.info(
                f"🤖 RL: Using BEST Ground Station {dest_gs['nodeId']} "
                f"for terminal {dest_terminal.get('terminalId')} "
                f"(distance: {dest_distance_km:.1f}km, "
                f"utilization: {dest_gs.get('resourceUtilization', 0):.1f}%, "
                f"battery: {dest_gs.get('batteryChargePercent', BATTERY_MAX_PERCENT):.1f}%, "
                f"packet_loss: {dest_gs.get('packetLossRate', 0)*100:.2f}% - "
                f"RESOURCE-OPTIMIZED selection)"
            )
        
        logger.info(f"🛰️ RL routing: {source_terminal.get('terminalId')} → {source_gs['nodeId']} → satellites → {dest_gs['nodeId']} → {dest_terminal.get('terminalId')}")
        
        # Terminals với explicit GS
        terminals = [source_terminal, dest_terminal]
        
        # Sử dụng environment để tìm path từ source_gs → dest_gs
        env = RoutingEnvironment(
            nodes=nodes,
            terminals=terminals,
            config=self.config,
            max_steps=15  # Giảm steps để tăng performance
        )
        
        # Reset với specific terminals và ground stations
        state, info = env.reset(
            options={
                'source_terminal_id': source_terminal.get('terminalId'),
                'dest_terminal_id': dest_terminal.get('terminalId'),
                'source_ground_station': source_gs,  # 🔥 NEW: Explicit GS
                'dest_ground_station': dest_gs,      # 🔥 NEW: Explicit GS
                'service_qos': service_qos
            }
        )
        
        # Route sử dụng agent (deterministic)
        done = False
        step_count = 0
        max_steps = 15  # 🔥 FIX: Tăng từ 6 lên 15 để đảm bảo tìm được path đầy đủ
        # max_steps = 6 quá thấp có thể khiến RL dừng sớm và tìm được path không đầy đủ
        
        while not done and step_count < max_steps:
            action = self.agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            state = next_state
            step_count += 1
        
        # Lấy kết quả path
        path = env.get_path_result()
        
        # 🔍 DEBUG: Log path segments để debug 
        path_segments = path.get('path', [])
        logger.info(f"🔍 RL Path segments ({len(path_segments)} total):")
        for i, seg in enumerate(path_segments):
            seg_type = seg.get('type', 'unknown')
            seg_id = seg.get('id', 'N/A')
            seg_name = seg.get('name', 'N/A')
            logger.info(f"   {i+1}. [{seg_type}] {seg_name} ({seg_id})")
        
        if not path or len(path.get('path', [])) < 3:  # Ít nhất source, 1 node, dest
            logger.warning("RL path finding failed, using heuristic")
            return self._fallback_to_heuristic(source_terminal, dest_terminal, nodes, service_qos)
        
        path['algorithm'] = 'rl_optimized'
        return path
    
    def _enhance_path_metrics(
        self,
        path: Dict,
        nodes: List[Dict],
        service_qos: Optional[Dict]
    ) -> Dict:
        """Enhance path với additional metrics và validation"""
        if not path:
            return path
        
        # Tính toán advanced metrics
        enhanced_path = path.copy()
        
        # QoS validation
        if service_qos:
            enhanced_path['qosMet'] = self._validate_qos_compliance(path, service_qos, nodes)
            enhanced_path['qosWarnings'] = self._get_qos_warnings(path, service_qos, nodes)
            enhanced_path['dropProbability'] = self._calculate_drop_probability(path, nodes, service_qos)
        
        # Path quality metrics
        enhanced_path['pathEfficiency'] = self._calculate_path_efficiency(path)
        enhanced_path['resourceUtilization'] = self._calculate_avg_resource_utilization(path, nodes)
        enhanced_path['reliabilityScore'] = self._calculate_reliability_score(path, nodes)
        
        # Performance metrics
        enhanced_path['calculatedAt'] = time.time()
        enhanced_path['algorithmVersion'] = 'optimized_v1.0'
        
        return enhanced_path
    
    def _preprocess_nodes(self, nodes: List[Dict], service_qos: Optional[Dict]) -> List[Dict]:
        """Pre-process nodes với QoS filtering optimized"""
        if not service_qos:
            return nodes
        
        # QoS filtering với caching
        cache_key = f"qos_{hash(str(service_qos))}"
        if cache_key in self._qos_cache:
            filtered_nodes = self._qos_cache[cache_key]
            # Nếu cache empty, xóa và tính lại (có thể do filter quá strict trước đó)
            if not filtered_nodes:
                logger.debug(f"QoS cache empty, recalculating...")
                del self._qos_cache[cache_key]
                filtered_nodes = self._filter_nodes_by_qos(nodes, service_qos)
                self._qos_cache[cache_key] = filtered_nodes
        else:
            filtered_nodes = self._filter_nodes_by_qos(nodes, service_qos)
            self._qos_cache[cache_key] = filtered_nodes
        
        logger.info(f"QoS preprocessing: {len(nodes)} → {len(filtered_nodes)} nodes")
        return filtered_nodes
    
    def _filter_nodes_by_qos(self, nodes: List[Dict], service_qos: Dict) -> List[Dict]:
        """Optimized QoS filtering với satellite prioritization và EXEMPT Ground Stations"""
        max_latency = service_qos.get('maxLatencyMs', float('inf'))
        min_bandwidth = service_qos.get('minBandwidthMbps', 0)
        max_loss_rate = service_qos.get('maxLossRate', 1.0)
        
        filtered_nodes = []
        
        for node in nodes:
            # Quick checks first
            if not node.get('isOperational', True):
                continue
            
            node_type = node.get('nodeType', '')
            
            # 🔥 FIX: EXEMPT Ground Stations từ QoS filtering (luôn giữ lại GS)
            if node_type == 'GROUND_STATION':
                # Ground stations LUÔN được giữ lại (cần thiết cho terminal connections)
                filtered_nodes.append(node)
                continue
            
            # QoS checks cho satellites - NỚI LỎNG hơn
            node_latency = node.get('nodeProcessingDelayMs', 0)
            node_loss_rate = node.get('packetLossRate', 0)
            node_bandwidth = node.get('communication', {}).get('bandwidth', 0)
            
            # Nới lỏng requirements cho satellites
            relaxed_bandwidth = min_bandwidth * 0.8 if min_bandwidth > 0 else 0
            relaxed_loss_rate = min(max_loss_rate * 1.2, 1.0)
            
            if (node_latency < 200 and
                node_bandwidth >= relaxed_bandwidth and 
                node_loss_rate <= relaxed_loss_rate):
                
                # 🆕 SATELLITE PRIORITY SCORE
                # Satellites được ưu tiên cao hơn cho routing
                sat_node_type = node.get('nodeType', '')
                if 'LEO' in sat_node_type.upper():
                    node['_priority_score'] = 12.0  # LEO satellites - fastest
                elif 'MEO' in sat_node_type.upper():
                    node['_priority_score'] = 10.0  # MEO satellites
                elif 'GEO' in sat_node_type.upper():
                    node['_priority_score'] = 8.0   # GEO satellites
                else:
                    node['_priority_score'] = 0.0
                
                filtered_nodes.append(node)
        
        # Nếu filter quá nhiều (< 10 nodes), fallback về tất cả operational nodes
        if len(filtered_nodes) < 10:
            logger.warning(f"QoS filtering too strict ({len(filtered_nodes)} nodes), "
                          f"using all operational nodes instead")
            filtered_nodes = [n for n in nodes if n.get('isOperational', True)]
            # Add priority scores cho fallback nodes
            for node in filtered_nodes:
                if '_priority_score' not in node:
                    node_type = node.get('nodeType', '')
                    if 'LEO' in node_type.upper():
                        node['_priority_score'] = 12.0
                    elif 'MEO' in node_type.upper():
                        node['_priority_score'] = 10.0
                    elif 'GEO' in node_type.upper():
                        node['_priority_score'] = 8.0
                    elif node_type == 'GROUND_STATION':
                        # GS có priority thấp hơn satellites nhưng vẫn cần thiết
                        node['_priority_score'] = 5.0
                    else:
                        node['_priority_score'] = 0.0
        
        # 🆕 SORT BY PRIORITY SCORE (satellites first)
        filtered_nodes.sort(key=lambda n: n.get('_priority_score', 0), reverse=True)
        
        logger.debug(f"QoS filtering: {len(nodes)} -> {len(filtered_nodes)} nodes "
                    f"(satellites prioritized)")
        return filtered_nodes
    
    def _validate_qos_compliance(self, path: Dict, service_qos: Dict, nodes: List[Dict]) -> bool:
        """Validate QoS compliance với comprehensive checks"""
        if not path:
            return False
        
        # Latency check
        max_latency = service_qos.get('maxLatencyMs', float('inf'))
        path_latency = path.get('estimatedLatency', 0)
        if path_latency > max_latency:
            return False
        
        # Loss rate check
        max_loss_rate = service_qos.get('maxLossRate', 1.0)
        path_loss_rate = self._calculate_path_loss_rate(path, nodes)
        if path_loss_rate > max_loss_rate:
            return False
        
        # Bandwidth check
        min_bandwidth = service_qos.get('minBandwidthMbps', 0)
        if not self._validate_path_bandwidth(path, nodes, min_bandwidth):
            return False
        
        return True
    
    def _get_qos_warnings(self, path: Dict, service_qos: Dict, nodes: List[Dict]) -> List[str]:
        """Get detailed QoS warnings"""
        warnings = []
        
        if not path:
            return ["Invalid path"]
        
        max_latency = service_qos.get('maxLatencyMs', float('inf'))
        min_bandwidth = service_qos.get('minBandwidthMbps', 0)
        max_loss_rate = service_qos.get('maxLossRate', 1.0)
        
        # Latency warning
        path_latency = path.get('estimatedLatency', 0)
        if path_latency > max_latency * 0.8:  # 80% of max latency
            warnings.append(f"High latency: {path_latency:.1f}ms (threshold: {max_latency}ms)")
        
        # Loss rate warning
        path_loss_rate = self._calculate_path_loss_rate(path, nodes)
        if path_loss_rate > max_loss_rate * 0.7:
            warnings.append(f"High loss rate: {path_loss_rate:.3f} (max: {max_loss_rate})")
        
        # Resource utilization warning
        avg_util = self._calculate_avg_resource_utilization(path, nodes)
        if avg_util > 0.8:
            warnings.append(f"High resource utilization: {avg_util:.1%}")
        
        return warnings
    
    def _calculate_drop_probability(self, path: Dict, nodes: List[Dict], service_qos: Dict) -> float:
        """Tính drop probability với improved model"""
        if not path:
            return 1.0
        
        drop_prob = 0.0
        node_map = {node['nodeId']: node for node in nodes}
        
        # Path segments analysis
        segments = path.get('path', [])
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]
            
            if current['type'] == 'node' and current['id'] in node_map:
                node = node_map[current['id']]
                
                # Resource-based drop probability
                utilization = node.get('resourceUtilization', 0) / 100.0
                drop_prob += utilization * 0.1
                
                # Loss rate
                loss_rate = node.get('packetLossRate', 0)
                drop_prob += loss_rate * 0.3
                
                # Buffer overflow risk
                buffer_ratio = node.get('currentPacketCount', 0) / max(node.get('packetBufferCapacity', NORM_PACKET_BUFFER), 1)
                if buffer_ratio > 0.8:
                    drop_prob += 0.15
        
        # Latency violation risk
        max_latency = service_qos.get('maxLatencyMs', float('inf'))
        path_latency = path.get('estimatedLatency', 0)
        if path_latency > max_latency:
            excess_ratio = (path_latency - max_latency) / max_latency
            drop_prob += min(0.3, excess_ratio * 0.5)
        
        return min(1.0, drop_prob)
    
    def _calculate_path_efficiency(self, path: Dict) -> float:
        """Tính path efficiency score (0-1, higher = better)"""
        if not path or 'totalDistance' not in path:
            return 0.0
        
        # Dựa trên số hops và distance
        hops = path.get('hops', 1)
        distance = path.get('totalDistance', 0)
        
        # Ideal: ít hops, ngắn distance
        if hops <= 3 and distance < 1000:
            return 1.0
        elif hops <= 5 and distance < 5000:
            return 0.8
        elif hops <= 8 and distance < 10000:
            return 0.6
        else:
            return 0.4
    
    def _calculate_avg_resource_utilization(self, path: Dict, nodes: List[Dict]) -> float:
        """Tính average resource utilization along path"""
        node_map = {node['nodeId']: node for node in nodes}
        utilizations = []
        
        for segment in path.get('path', []):
            if segment['type'] == 'node' and segment['id'] in node_map:
                util = node_map[segment['id']].get('resourceUtilization', 0)
                utilizations.append(util)
        
        return np.mean(utilizations) / 100.0 if utilizations else 0.0
    
    def _calculate_reliability_score(self, path: Dict, nodes: List[Dict]) -> float:
        """Tính reliability score cho path"""
        node_map = {node['nodeId']: node for node in nodes}
        reliability_factors = []
        
        for segment in path.get('path', []):
            if segment['type'] == 'node' and segment['id'] in node_map:
                node = node_map[segment['id']]
                
                # Dựa trên multiple factors
                loss_rate = 1 - min(node.get('packetLossRate', 0), 1.0)
                battery = node.get('batteryChargePercent', BATTERY_MAX_PERCENT) / BATTERY_MAX_PERCENT
                healthy = 1.0 if node.get('healthy', True) else 0.5
                
                node_reliability = (loss_rate + battery + healthy) / 3
                reliability_factors.append(node_reliability)
        
        return np.mean(reliability_factors) if reliability_factors else 0.0
    
    def _calculate_path_loss_rate(self, path: Dict, nodes: List[Dict]) -> float:
        """Tính end-to-end loss rate cho path"""
        node_map = {node['nodeId']: node for node in nodes}
        total_loss = 0.0
        node_count = 0
        
        for segment in path.get('path', []):
            if segment['type'] == 'node' and segment['id'] in node_map:
                loss_rate = node_map[segment['id']].get('packetLossRate', 0)
                total_loss += loss_rate
                node_count += 1
        
        return total_loss if node_count == 0 else total_loss / node_count
    
    def _validate_path_bandwidth(self, path: Dict, nodes: List[Dict], min_bandwidth: float) -> bool:
        """Validate path có đáp ứng bandwidth requirements"""
        node_map = {node['nodeId']: node for node in nodes}
        
        for segment in path.get('path', []):
            if segment['type'] == 'node' and segment['id'] in node_map:
                node_bandwidth = node_map[segment['id']].get('communication', {}).get('bandwidth', 0)
                if node_bandwidth < min_bandwidth:
                    return False
        
        return True
    
    def _fallback_to_heuristic(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict],
        service_qos: Optional[Dict]
    ) -> Dict:
        """Optimized heuristic fallback"""
        try:
            from api.routing_bp import _calculate_path_rl_heuristic
            path = _calculate_path_rl_heuristic(source_terminal, dest_terminal, nodes)
            
            # Add QoS info
            if service_qos:
                path['qosMet'] = self._validate_qos_compliance(path, service_qos, nodes)
                path['qosWarnings'] = self._get_qos_warnings(path, service_qos, nodes)
                path['dropProbability'] = self._calculate_drop_probability(path, nodes, service_qos)
            
            path['algorithm'] = 'heuristic_fallback'
            return path
            
        except ImportError:
            # Basic fallback implementation
            return self._basic_heuristic_path(source_terminal, dest_terminal, nodes, service_qos)
    
    def _basic_heuristic_path(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict],
        service_qos: Optional[Dict]
    ) -> Dict:
        """Basic heuristic path calculation"""
        operational_nodes = [n for n in nodes if n.get('isOperational', True)]
        
        # Simple shortest path heuristic
        path_segments = [{
            'type': 'terminal',
            'id': source_terminal.get('terminalId'),
            'position': source_terminal.get('position')
        }]
        
        # Add closest node to source
        source_pos = source_terminal.get('position')
        if operational_nodes and source_pos:
            closest_node = min(
                operational_nodes,
                key=lambda n: self._calculate_distance(source_pos, n.get('position', {}))
            )
            path_segments.append({
                'type': 'node',
                'id': closest_node.get('nodeId'),
                'position': closest_node.get('position')
            })
        
        # Add destination
        path_segments.append({
            'type': 'terminal',
            'id': dest_terminal.get('terminalId'),
            'position': dest_terminal.get('position')
        })
        
        return {
            'source': {'terminalId': source_terminal.get('terminalId')},
            'destination': {'terminalId': dest_terminal.get('terminalId')},
            'path': path_segments,
            'totalDistance': 0,
            'estimatedLatency': 0,
            'hops': len(path_segments) - 1,
            'algorithm': 'basic_heuristic'
        }
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate distance sử dụng state builder"""
        return self.state_builder._calculate_distance(pos1, pos2)
    
    def _get_cache_key(self, nodes: List[Dict], service_qos: Optional[Dict]) -> str:
        """Generate cache key cho nodes và QoS"""
        node_ids = ''.join(sorted([n.get('nodeId', '') for n in nodes]))
        qos_str = str(service_qos) if service_qos else 'no_qos'
        return f"{hash(node_ids)}_{hash(qos_str)}"
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance metrics"""
        # Moving average
        self.avg_response_time = (
            self.avg_response_time * (self.request_count - 1) + response_time
        ) / self.request_count
        
        # Log periodically
        if self.request_count % 100 == 0:
            logger.info(
                f"RL Service Performance: "
                f"Requests: {self.request_count}, "
                f"Avg Response Time: {self.avg_response_time:.3f}s"
            )
    
    def get_performance_metrics(self) -> Dict:
        """Get service performance metrics"""
        return {
            'total_requests': self.request_count,
            'avg_response_time': self.avg_response_time,
            'agent_loaded': self.agent_loaded,
            'cache_size': len(self._node_cache) + len(self._qos_cache)
        }


# Global instance
_rl_routing_service = None


def get_rl_routing_service(config: Dict = None) -> RLRoutingService:
    """Get global RL routing service instance"""
    global _rl_routing_service
    if _rl_routing_service is None:
        _rl_routing_service = RLRoutingService(config)
    return _rl_routing_service