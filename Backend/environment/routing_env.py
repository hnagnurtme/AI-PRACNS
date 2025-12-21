"""
Routing Environment for SAGIN RL
Gymnasium environment for routing with Dijkstra-aligned rewards
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
from collections import deque

from environment.state_builder import RoutingStateBuilder
from environment.constants import (
    SPEED_OF_LIGHT_MPS, MS_PER_SECOND, M_TO_KM,
    DISTANCE_NEAR_DEST_M, DISTANCE_CLOSE_DEST_M, DISTANCE_FAR_DEST_M,
    DISTANCE_VERY_CLOSE_M, MIN_PATH_HOPS, MIN_PATH_SEGMENTS,
    REWARD_SUCCESS, REWARD_FAILURE, REWARD_STEP_PENALTY,
    REWARD_HOP_PENALTY, REWARD_GS_HOP_PENALTY, REWARD_LOAD_BALANCING,
    REWARD_LOOP_PENALTY, REWARD_DROP_NODE, PROGRESS_REWARD_SCALE,
    DISTANCE_REWARD_SCALE, QUALITY_REWARD_SCALE, PROXIMITY_BONUS_SCALE,
    PROGRESS_DIVISOR_M, DETOUR_PENALTY_DIVISOR_M, DETOUR_PENALTY_MULTIPLIER,
    DISTANCE_PENALTY_DIVISOR_M, PROXIMITY_CLOSE_M, PROXIMITY_FAR_M,
    PROXIMITY_BONUS_MULTIPLIER, BONUS_EXACT_DEST_GS, BONUS_QOS_COMPLIANCE,
    PENALTY_QOS_VIOLATION, EFFICIENCY_BONUS_PER_HOP, EFFICIENCY_PENALTY_PER_HOP,
    EFFICIENCY_EXTRA_PENALTY_BASE, EFFICIENCY_EXTRA_PENALTY_MULTIPLIER,
    DISTANCE_RATIO_EFFICIENT, DISTANCE_RATIO_ACCEPTABLE, DISTANCE_RATIO_POOR,
    BONUS_DISTANCE_EFFICIENT, BONUS_DISTANCE_ACCEPTABLE, PENALTY_DISTANCE_POOR,
    QUALITY_EXCELLENT, QUALITY_GOOD, QUALITY_BAD, BONUS_EXCELLENT_NODE,
    BONUS_GOOD_NODE, PENALTY_BAD_NODE, EXCESS_HOPS_THRESHOLD,
    EXCESS_HOPS_PENALTY_MULTIPLIER, UTILIZATION_HIGH_PERCENT,
    UTILIZATION_MEDIUM_PERCENT, UTILIZATION_MAX_PERCENT,
    TERMINAL_UTILIZATION_IMPACT, GS_CONNECTION_OVERLOADED, GS_CONNECTION_HIGH,
    BATTERY_MAX_PERCENT, DIJKSTRA_DROP_THRESHOLD, DIJKSTRA_PENALTY_THRESHOLD,
    DIJKSTRA_PENALTY_MULTIPLIER, DIJKSTRA_PROGRESS_SCALE,
    PROGRESS_CHECK_MIN_STEPS, PROGRESS_CHECK_WINDOW_SIZE,
    PROGRESS_MIN_THRESHOLD_M, PROGRESS_NO_PROGRESS_PENALTY,
    ADAPTIVE_MAX_STEPS_NETWORK_DIVISOR, ADAPTIVE_MAX_STEPS_MULTIPLIER,
    GS_MAX_DIRECT_RANGE_KM
)

logger = logging.getLogger(__name__)


def get_terminal_connection_count(node_id: str) -> int:
    """Get number of terminals connected to a node - safe version to avoid circular import"""
    try:
        from models.database import db
        terminals_collection = db.get_collection('terminals')
        count = terminals_collection.count_documents({
            'connectedNodeId': node_id,
            'status': {'$in': ['connected', 'transmitting']}
        })
        return count
    except Exception as e:
        logger.warning(f"Error counting terminals for {node_id}: {e}")
        return 0


class RoutingEnvironment(gym.Env):
    """
    Optimized environment cho routing với reward engineering tiên tiến
    Giữ nguyên interface và tên class
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        nodes: List[Dict],
        terminals: List[Dict],
        config: Dict = None,
        max_steps: int = 8
    ):
        super().__init__()
        
        self.config = config or {}
        self.nodes = nodes
        self.terminals = terminals
        
        training_config = self.config.get('training', {})
        adaptive_max_steps = training_config.get('adaptive_max_steps', True)
        
        base_max_steps = max_steps
        network_size = len(nodes)
        
        if adaptive_max_steps:
            estimated_max_hops = min(
                network_size // ADAPTIVE_MAX_STEPS_NETWORK_DIVISOR,
                base_max_steps * ADAPTIVE_MAX_STEPS_MULTIPLIER
            )
            self.max_steps = max(base_max_steps, estimated_max_hops)
            self.adaptive_max_steps = True
            if self.max_steps > base_max_steps:
                logger.info(
                    f"Dynamic max_steps: {self.max_steps} "
                    f"(operational_nodes={network_size}, base={base_max_steps}, "
                    f"estimated_hops={estimated_max_hops})"
                )
        else:
            self.max_steps = max_steps
            self.adaptive_max_steps = False
            logger.debug(f"Fixed max_steps: {self.max_steps} (adaptive disabled)")
        
        self.state_builder = RoutingStateBuilder(config)
        
        max_actions = min(len(nodes), self.state_builder.max_nodes)
        self.action_space = spaces.Discrete(max_actions)
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(self.state_builder.state_dimension,),
            dtype=np.float32
        )
        
        self.source_terminal = None
        self.dest_terminal = None
        self.current_node = None
        self.path = []
        self.visited_nodes = set()
        self.step_count = 0
        self.total_distance = 0.0
        self.total_latency = 0.0
        self.service_qos = None
        self.terminated = False
        
        reward_config = self.config.get('reward', {})
        self.use_dijkstra_aligned_rewards = reward_config.get('dijkstra_aligned', True)
        
        self.drop_threshold = reward_config.get('drop_threshold', DIJKSTRA_DROP_THRESHOLD)
        self.penalty_threshold = reward_config.get('penalty_threshold', DIJKSTRA_PENALTY_THRESHOLD)
        self.penalty_multiplier = reward_config.get('penalty_multiplier', DIJKSTRA_PENALTY_MULTIPLIER)
        
        self.success_reward = reward_config.get('success_reward', REWARD_SUCCESS)
        self.failure_penalty = reward_config.get('failure_penalty', REWARD_FAILURE)
        self.step_penalty = reward_config.get('step_penalty', REWARD_STEP_PENALTY)
        self.hop_penalty = reward_config.get('hop_penalty', REWARD_HOP_PENALTY)
        self.ground_station_hop_penalty = reward_config.get('ground_station_hop_penalty', REWARD_GS_HOP_PENALTY)
        self.progress_reward_scale = reward_config.get('progress_reward_scale', PROGRESS_REWARD_SCALE)
        self.distance_reward_scale = reward_config.get('distance_reward_scale', DISTANCE_REWARD_SCALE)
        self.quality_reward_scale = reward_config.get('quality_reward_scale', QUALITY_REWARD_SCALE)
        self.proximity_bonus_scale = reward_config.get('proximity_bonus_scale', PROXIMITY_BONUS_SCALE)
        
        self._node_cache = {node['nodeId']: node for node in nodes}
        self._terminal_cache = {terminal['terminalId']: terminal for terminal in terminals}
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.terminated = False
        
        source_ground_station = None
        dest_ground_station = None
        
        if options:
            source_terminal_id = options.get('source_terminal_id')
            dest_terminal_id = options.get('dest_terminal_id')
            self.service_qos = options.get('service_qos')
            
            source_ground_station = options.get('source_ground_station')
            dest_ground_station = options.get('dest_ground_station')
            
            self.source_terminal = self._terminal_cache.get(source_terminal_id)
            self.dest_terminal = self._terminal_cache.get(dest_terminal_id)
        else:
            if len(self.terminals) < 2:
                raise ValueError("Need at least 2 terminals")
            
            indices = self.np_random.choice(len(self.terminals), size=2, replace=False)
            self.source_terminal = self.terminals[indices[0]]
            self.dest_terminal = self.terminals[indices[1]]
        
        if not self.source_terminal or not self.dest_terminal:
            raise ValueError("Source or destination terminal not found")
        
        if source_ground_station:
            self.current_node = source_ground_station
            logger.info(f"RL starting from explicit source GS: {source_ground_station['nodeId']}")
        else:
            self.current_node = self._find_optimal_initial_node(
                self.source_terminal, self.dest_terminal
            )
        
        if not self.current_node:
            operational_nodes = [
                n for n in self.nodes 
                if n.get('isOperational', True) and n.get('position')
            ]
            if operational_nodes:
                dest_pos = self.dest_terminal.get('position')
                self.current_node = min(
                    operational_nodes,
                    key=lambda n: self._calculate_distance(
                        n.get('position'), dest_pos
                    ) if n.get('position') else float('inf')
                )
            else:
                raise ValueError("No operational nodes available")
        
        self.dest_ground_station = dest_ground_station
        
        self.path = [self.source_terminal, self.current_node]
        self.visited_nodes = {self.current_node.get('nodeId')}
        self.step_count = 0
        self.total_distance = 0.0
        self.total_latency = 0.0
        
        self.recent_distances = deque(maxlen=PROGRESS_CHECK_WINDOW_SIZE)
        initial_dist = self._calculate_distance(
            self.current_node.get('position'),
            self.dest_terminal.get('position')
        )
        self.recent_distances.append(initial_dist)
        
        state = self.state_builder.build_state(
            nodes=self.nodes,
            source_terminal=self.source_terminal,
            dest_terminal=self.dest_terminal,
            current_node=self.current_node,
            service_qos=self.service_qos,
            visited_nodes=list(self.visited_nodes)
        )
        
        info = {
            'path': self.path.copy(),
            'current_node': self.current_node.get('nodeId'),
            'distance_to_dest': self._calculate_distance(
                self.current_node.get('position'),
                self.dest_terminal.get('position')
            ),
            'hops': 1
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step function with reward engineering"""
        self.step_count += 1
        
        # Initialize progress to 0.0 to avoid UnboundLocalError
        progress = 0.0
        
        filtered_nodes = self.state_builder._smart_node_filtering(
            self.nodes, self.source_terminal, self.dest_terminal, 
            self.current_node, list(self.visited_nodes)
        )
        
        stress_aware_nodes = self._filter_stress_problematic_nodes(filtered_nodes)
        if len(stress_aware_nodes) > 0:
            filtered_nodes = stress_aware_nodes
        
        if action < 0 or action >= len(filtered_nodes) or not filtered_nodes:
            next_node = self._find_fallback_node()
            if not next_node:
                state = self.state_builder.build_state(
                    self.nodes, self.source_terminal, self.dest_terminal,
                    self.current_node, self.service_qos, list(self.visited_nodes)
                )
                info = {
                    'path': self.path.copy(),
                    'current_node': self.current_node.get('nodeId') if self.current_node else None,
                    'distance_to_dest': 0.0,
                    'hops': len(self.path) - 1,
                    'terminated': True,
                    'progress': 0.0,
                    'error': 'no_valid_nodes'
                }
                return state, self.failure_penalty, True, False, info
        else:
            next_node = filtered_nodes[action]
        
        # 🔧 CONNECTIVITY VALIDATION: Prevent out-of-range hops
        # This is a safety check - state_builder should filter, but we double-check
        if self.current_node and next_node:
            current_pos = self.current_node.get('position')
            next_pos = next_node.get('position')
            if current_pos and next_pos:
                hop_dist = self._calculate_distance(current_pos, next_pos)
                hop_dist_km = hop_dist / 1000.0
                
                current_type = self.current_node.get('nodeType', '')
                next_type = next_node.get('nodeType', '')
                
                # Get max range based on node types
                max_range_km = self._get_max_range_for_connection(current_type, next_type)
                
                is_invalid_hop = False
                if hop_dist_km > max_range_km:
                    is_invalid_hop = True
                    logger.warning(
                        f"🚫 BLOCKED invalid hop ({current_type} → {next_type}): "
                        f"{self.current_node.get('nodeId')} → {next_node.get('nodeId')} "
                        f"({hop_dist_km:.1f}km > {max_range_km:.1f}km)"
                    )
                
                if is_invalid_hop:
                    # Return penalty for invalid action
                    state = self.state_builder.build_state(
                        self.nodes, self.source_terminal, self.dest_terminal,
                        self.current_node, self.service_qos, list(self.visited_nodes)
                    )
                    reward = REWARD_FAILURE  # Invalid action penalty (use constant)
                    info = {
                        'path': self.path.copy(),
                        'invalid_hop': True,
                        'current_node': self.current_node.get('nodeId'),
                        'attempted_node': next_node.get('nodeId'),
                        'hop_distance_km': hop_dist_km,
                        'max_range_km': max_range_km,
                        'terminated': False,
                        'progress': 0.0
                    }
                    truncated = self.step_count >= self.max_steps
                    return state, reward, False, truncated, info
        
        next_node_id = next_node.get('nodeId')
        if next_node_id in self.visited_nodes:
            reward = REWARD_LOOP_PENALTY
            terminated = False
            truncated = self.step_count >= self.max_steps
            
            # Calculate progress for loop case
            dest_pos = self.dest_terminal.get('position')
            next_pos = next_node.get('position')
            if dest_pos and self.current_node and self.current_node.get('position') and next_pos:
                current_dist = self._calculate_distance(
                    self.current_node.get('position'), dest_pos
                )
                next_dist = self._calculate_distance(next_pos, dest_pos)
                progress = current_dist - next_dist
            else:
                progress = 0.0
            
            state = self.state_builder.build_state(
                self.nodes, self.source_terminal, self.dest_terminal,
                self.current_node, self.service_qos, list(self.visited_nodes)
            )
            
            info = {
                'path': self.path.copy(),
                'loop_detected': True,
                'current_node': next_node_id,
                'hops': len(self.path) - 1,
                'terminated': terminated,
                'progress': progress
            }
            # 🔧 REWARD CLIPPING: Also clip for loop detection
            reward = max(-100.0, min(600.0, reward))
            return state, reward, terminated, truncated, info
        
        self.path.append(next_node)
        self.visited_nodes.add(next_node_id)
        
        current_pos = self.current_node.get('position')
        next_pos = next_node.get('position')
        hop_distance = self._calculate_distance(current_pos, next_pos)
        self.total_distance += hop_distance
        
        propagation_delay = (hop_distance / SPEED_OF_LIGHT_MPS) * MS_PER_SECOND
        processing_delay = next_node.get('nodeProcessingDelayMs', 5)
        hop_latency = propagation_delay + processing_delay
        self.total_latency += hop_latency
        
        current_node_type = self.current_node.get('nodeType', '')
        next_node_type = next_node.get('nodeType', '')
        
        current_connection_count = get_terminal_connection_count(self.current_node.get('nodeId'))
        next_connection_count = get_terminal_connection_count(next_node.get('nodeId'))
        
        current_node_utilization = self.current_node.get('resourceUtilization', 0) + (current_connection_count * TERMINAL_UTILIZATION_IMPACT)
        next_node_utilization = next_node.get('resourceUtilization', 0) + (next_connection_count * TERMINAL_UTILIZATION_IMPACT)
        
        initial_reward = 0.0
        if current_node_type == 'GROUND_STATION' and next_node_type == 'GROUND_STATION':
            if current_node_utilization > UTILIZATION_HIGH_PERCENT and next_node_utilization < current_node_utilization - 20.0:
                initial_reward = REWARD_LOAD_BALANCING
                logger.debug(f"Load balancing bonus: current={current_node_utilization:.1f}%, next={next_node_utilization:.1f}%")
            else:
                initial_reward = self.ground_station_hop_penalty
                logger.debug(f"GS→GS penalty: {initial_reward}")
        
        dest_pos = self.dest_terminal.get('position')
        dist_to_dest = self._calculate_distance(next_pos, dest_pos)
        
        # Calculate progress for info dict (used in all cases)
        prev_dist = self._calculate_distance(
            self.current_node.get('position'), dest_pos
        )
        progress = prev_dist - dist_to_dest
        
        if self.step_count > PROGRESS_CHECK_MIN_STEPS:
            recent_progress = self._check_recent_progress(dist_to_dest)
            if not recent_progress:
                state = self.state_builder.build_state(
                    self.nodes, self.source_terminal, self.dest_terminal,
                    self.current_node, self.service_qos, list(self.visited_nodes)
                )
                info = {
                    'path': self.path.copy(),
                    'current_node': next_node_id,
                    'distance_to_dest': dist_to_dest,
                    'hops': len(self.path) - 1,
                    'terminated': False,
                    'progress': progress,
                    'reason': 'no_progress'
                }
                return state, PROGRESS_NO_PROGRESS_PENALTY, False, True, info
        
        # Update recent distances tracking
        self.recent_distances.append(dist_to_dest)
        
        reached_dest_gs = False
        if hasattr(self, 'dest_ground_station') and self.dest_ground_station:
            reached_dest_gs = (next_node_id == self.dest_ground_station['nodeId'])
        
        is_ground_station = next_node_type == 'GROUND_STATION'
        is_near_dest = dist_to_dest < DISTANCE_NEAR_DEST_M
        
        terminated = False
        reward = initial_reward
        
        has_min_hops = len(self.path) >= MIN_PATH_HOPS
        
        # 🔥 FIX: Chỉ terminate khi thực sự đến destination GS
        # Không cho phép early termination dựa trên distance để tránh "nhảy" trực tiếp
        # Điều này đảm bảo RL phải đi qua đầy đủ path giống Dijkstra
        if hasattr(self, 'dest_ground_station') and self.dest_ground_station:
            # Nếu có explicit dest_gs (từ reset options), CHỈ terminate khi đến đúng GS đó
            # Không cho phép early termination dựa trên distance
            if reached_dest_gs and has_min_hops:
                self.path.append(self.dest_terminal)
                terminated = True
                self.terminated = True
        else:
            # Fallback: Nếu không có explicit dest_gs, cho phép terminate khi gần destination
            # (cho backward compatibility)
            if reached_dest_gs or \
               (is_ground_station and is_near_dest and has_min_hops) or \
               (has_min_hops and dist_to_dest < DISTANCE_CLOSE_DEST_M):
                self.path.append(self.dest_terminal)
                terminated = True
                self.terminated = True
        
        if terminated:
            # 🔧 SIMPLIFIED SUCCESS REWARD
            # Base success: 500, small bonuses, no extreme penalties
            reward = self.success_reward  # 500
            
            # Small bonus for reaching exact destination GS
            if reached_dest_gs:
                reward += 50.0
                logger.info(f"RL reached exact destination GS: {self.dest_ground_station['nodeId']}")
            
            # Small efficiency bonus/penalty (capped)
            num_hops = len(self.path) - 2
            if num_hops <= 3:
                reward += 30.0  # Very efficient
            elif num_hops <= 5:
                reward += 10.0  # Good
            elif num_hops > 8:
                reward -= 30.0  # Too long (capped penalty)
                
        else:
            # 🔧 NORMALIZED REWARD FUNCTION - Scale to reasonable range
            # Target range: -50 to +50 per step (not counting terminal rewards)
            
            # Get initial distance for normalization
            initial_dist = self._calculate_distance(
                self.source_terminal.get('position'),
                self.dest_terminal.get('position')
            )
            
            # Component 1: Progress reward (ratio-based, not absolute)
            # This normalizes progress regardless of absolute distances
            if initial_dist > 0:
                progress_ratio = progress / initial_dist  # -1 to +1 range
                if progress > 0:
                    reward = progress_ratio * 30.0  # Max +30 per step
                else:
                    reward = progress_ratio * 50.0  # Max -50 per step (penalty stronger)
            else:
                reward = 0.0
            
            # Component 2: Small step penalty
            reward -= 1.0  # Encourage shorter paths
            
            # Component 3: Resource quality penalties (TRAP NODE DETECTION)
            # These help RL learn to avoid overloaded nodes that Dijkstra ignores
            # Using constants from environment.constants for thresholds and penalties
            from environment.constants import (
                TRAP_UTILIZATION_SEVERE, TRAP_UTILIZATION_HIGH, TRAP_UTILIZATION_MODERATE,
                TRAP_PACKET_LOSS_SEVERE, TRAP_PACKET_LOSS_HIGH, TRAP_PACKET_LOSS_MODERATE,
                TRAP_BATTERY_CRITICAL, TRAP_BATTERY_LOW, TRAP_BATTERY_MODERATE,
                TRAP_DELAY_HIGH, TRAP_DELAY_MODERATE,
                PENALTY_UTILIZATION_SEVERE, PENALTY_UTILIZATION_HIGH, PENALTY_UTILIZATION_MODERATE,
                PENALTY_PACKET_LOSS_SEVERE, PENALTY_PACKET_LOSS_HIGH, PENALTY_PACKET_LOSS_MODERATE,
                PENALTY_BATTERY_CRITICAL, PENALTY_BATTERY_LOW, PENALTY_BATTERY_MODERATE,
                PENALTY_DELAY_HIGH, PENALTY_DELAY_MODERATE
            )
            
            # 3a. Utilization penalty
            cpu = next_node.get('cpu', {}).get('utilization', 0)
            mem = next_node.get('memory', {}).get('utilization', 0)
            bw = next_node.get('bandwidth', {}).get('utilization', 0)
            max_util = max(cpu, mem, bw)
            
            if max_util >= TRAP_UTILIZATION_SEVERE:
                reward += PENALTY_UTILIZATION_SEVERE
            elif max_util >= TRAP_UTILIZATION_HIGH:
                reward += PENALTY_UTILIZATION_HIGH
            elif max_util >= TRAP_UTILIZATION_MODERATE:
                reward += PENALTY_UTILIZATION_MODERATE
            
            # 3b. Packet loss penalty
            packet_loss = next_node.get('packetLossRate', 0)
            if packet_loss >= TRAP_PACKET_LOSS_SEVERE:
                reward += PENALTY_PACKET_LOSS_SEVERE
            elif packet_loss >= TRAP_PACKET_LOSS_HIGH:
                reward += PENALTY_PACKET_LOSS_HIGH
            elif packet_loss >= TRAP_PACKET_LOSS_MODERATE:
                reward += PENALTY_PACKET_LOSS_MODERATE
            
            # 3c. Low battery penalty
            battery = next_node.get('batteryChargePercent', 100)
            if battery < TRAP_BATTERY_CRITICAL:
                reward += PENALTY_BATTERY_CRITICAL
            elif battery < TRAP_BATTERY_LOW:
                reward += PENALTY_BATTERY_LOW
            elif battery < TRAP_BATTERY_MODERATE:
                reward += PENALTY_BATTERY_MODERATE
            
            # 3d. High processing delay penalty
            delay = next_node.get('nodeProcessingDelayMs', 0)
            if delay >= TRAP_DELAY_HIGH:
                reward += PENALTY_DELAY_HIGH
            elif delay >= TRAP_DELAY_MODERATE:
                reward += PENALTY_DELAY_MODERATE
        
        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            # Simple failure penalty - not too extreme
            reward = -50.0  # Fixed penalty instead of accumulating
        
        self.current_node = next_node
        state = self.state_builder.build_state(
            nodes=self.nodes,
            source_terminal=self.source_terminal,
            dest_terminal=self.dest_terminal,
            current_node=self.current_node,
            service_qos=self.service_qos,
            visited_nodes=list(self.visited_nodes)
        )
        
        info = {
            'path': self.path.copy(),
            'current_node': next_node_id,
            'distance_to_dest': dist_to_dest,
            'total_distance': self.total_distance,
            'total_latency': self.total_latency,
            'hops': len(self.path) - 1,
            'terminated': terminated,
            'progress': progress if not terminated else 1.0
        }
        
        # 🔧 REWARD CLIPPING: Prevent extreme outliers
        # Clip to reasonable range: -100 to +600
        reward = max(-100.0, min(600.0, reward))
        
        return state, reward, terminated, truncated, info
    
    def _find_optimal_initial_node(
        self, 
        source_terminal: Dict, 
        dest_terminal: Dict
    ) -> Optional[Dict]:
        """Tìm initial node tối ưu cân bằng giữa source và destination"""
        source_pos = source_terminal.get('position')
        dest_pos = dest_terminal.get('position')
        
        if not source_pos or not dest_pos:
            return self._find_best_ground_station(source_terminal, self.nodes)
        
        operational_nodes = [
            n for n in self.nodes 
            if n.get('isOperational', True) and n.get('position')
        ]
        
        if not operational_nodes:
            return None
        
        # Tìm node cân bằng giữa khoảng cách đến source và destination
        def balance_score(node):
            node_pos = node.get('position')
            dist_to_source = self._calculate_distance(node_pos, source_pos)
            dist_to_dest = self._calculate_distance(node_pos, dest_pos)
            
            # Ưu tiên nodes gần source nhưng không quá xa destination
            balance = dist_to_source + dist_to_dest
            # Penalty cho nodes quá xa đường thẳng source-dest
            direct_dist = self._calculate_distance(source_pos, dest_pos)
            
            # Fix: Tránh division by zero khi source và dest ở cùng vị trí
            if direct_dist < 1.0:  # Nếu quá gần (< 1m)
                return balance  # Chỉ dùng tổng khoảng cách
            
            triangle_ratio = (dist_to_source + dist_to_dest) / direct_dist
            
            return balance * triangle_ratio
        
        return min(operational_nodes, key=balance_score)
    
    def _find_fallback_node(self) -> Optional[Dict]:
        """Fallback strategy khi không có valid actions.
        
        IMPORTANT: Must only return nodes that are within valid communication range
        from current_node. Uses same range validation as state_builder.
        """
        if not self.current_node:
            return None
            
        current_pos = self.current_node.get('position')
        current_type = self.current_node.get('nodeType', '')
        
        # Get all operational nodes first
        operational_nodes = [
            n for n in self.nodes 
            if n.get('isOperational', True) and n.get('position')
            and n.get('nodeId') not in self.visited_nodes  # Don't revisit
        ]
        
        if not operational_nodes:
            return None
        
        # Filter to only nodes within valid range
        reachable_nodes = []
        for node in operational_nodes:
            node_pos = node.get('position')
            node_type = node.get('nodeType', '')
            
            if not node_pos:
                continue
                
            dist_km = self._calculate_distance(current_pos, node_pos) / 1000.0
            max_range_km = self._get_max_range_for_connection(current_type, node_type)
            
            if dist_km <= max_range_km:
                reachable_nodes.append(node)
        
        if not reachable_nodes:
            logger.debug(f"No reachable nodes from {self.current_node.get('nodeId')}")
            return None
        
        # Among reachable nodes, prefer ground stations near destination
        dest_pos = self.dest_terminal.get('position')
        gs_nodes = [n for n in reachable_nodes if 'GS' in n.get('nodeId', '') or 'GROUND' in n.get('nodeType', '').upper()]
        
        if gs_nodes:
            return min(
                gs_nodes,
                key=lambda n: self._calculate_distance(n.get('position'), dest_pos)
            )
        
        # Otherwise, return closest reachable node to destination
        return min(
            reachable_nodes,
            key=lambda n: self._calculate_distance(n.get('position'), dest_pos)
        )
    
    def _estimate_optimal_hops(self) -> int:
        """Estimate optimal number of hops for path"""
        direct_dist = self._calculate_distance(
            self.source_terminal.get('position'),
            self.dest_terminal.get('position')
        )
        
        avg_hop_dist_m = 3000000
        optimal_hops = max(MIN_PATH_HOPS, int(direct_dist / avg_hop_dist_m) + 2)
        
        return min(optimal_hops, EFFICIENCY_EXTRA_PENALTY_BASE)
    
    def _find_best_ground_station(
        self, terminal: Dict, nodes: List[Dict]
    ) -> Optional[Dict]:
        """Tìm ground station tốt nhất cho terminal"""
        terminal_pos = terminal.get('position')
        if not terminal_pos:
            return None
        
        ground_stations = [
            n for n in nodes
            if n.get('nodeType') == 'GROUND_STATION'
            and n.get('isOperational', True)
            and n.get('position')
        ]
        
        if not ground_stations:
            return None
        
        # Find closest với quality consideration
        best_station = None
        best_score = float('inf')
        
        for station in ground_stations:
            distance = self._calculate_distance(
                terminal_pos, station.get('position')
            )
            quality = self.state_builder._compute_node_quality(station)
            
            # Score kết hợp distance và quality
            score = distance / 1000.0 * (1.1 - quality)  # Higher quality = better
            
            if score < best_score:
                best_score = score
                best_station = station
        
        return best_station
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate distance với cache"""
        if not pos1 or not pos2:
            return float('inf')
        
        # Sử dụng state builder's cached distance calculation
        return self.state_builder._calculate_distance(pos1, pos2)
    
    def _filter_stress_problematic_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """
        Filter out nodes với vấn đề nghiêm trọng trong stress scenarios
        Giúp RL học tránh các nodes có vấn đề
        
        IMPORTANT: This function receives ALREADY FILTERED nodes from state_builder.
        Do NOT re-add nodes from any other source as they may be out of range.
        """
        filtered = []
        for node in nodes:
            # Chỉ filter nếu node có vấn đề nghiêm trọng
            utilization = node.get('resourceUtilization', 0)
            battery = node.get('batteryChargePercent', 100)
            is_operational = node.get('isOperational', True)
            packet_loss = node.get('packetLossRate', 0)
            
            # Giữ node nếu:
            # 1. Operational
            # 2. Không có quá nhiều vấn đề cùng lúc
            if is_operational:
                # Chỉ filter nếu có nhiều vấn đề cùng lúc
                problem_count = 0
                if utilization > 0.9:
                    problem_count += 1
                if battery < 0.15:
                    problem_count += 1
                if packet_loss > 0.1:
                    problem_count += 1
                
                # Chỉ filter nếu có 2+ vấn đề nghiêm trọng
                if problem_count < 2:
                    filtered.append(node)
        
        # 🔧 FIX: If filter removed too many, return ORIGINAL filtered list (from state_builder)
        # Do NOT use `nodes` directly as fallback - those are already range-filtered
        # If we have very few nodes, just return what we have
        if len(filtered) < 3:
            # Return original input (already range-filtered by state_builder)
            # Sort by quality but keep all nodes from input
            nodes_sorted = sorted(
                nodes,  # These are already range-filtered
                key=lambda n: (
                    -n.get('resourceUtilization', 0),  # Lower is better
                    -n.get('batteryChargePercent', 100),  # Higher is better
                    n.get('packetLossRate', 0)  # Lower is better
                )
            )
            return nodes_sorted  # Return all, not just top half
        
        return filtered
    
    def _get_max_range_for_connection(self, source_type: str, dest_type: str) -> float:
        """Get max communication range based on node types.
        
        Uses constants from environment.constants for different satellite/GS combinations.
        """
        from environment.constants import (
            GS_MAX_DIRECT_RANGE_KM,
            GS_TO_LEO_MAX_RANGE_KM,
            GS_TO_MEO_MAX_RANGE_KM,
            GS_TO_GEO_MAX_RANGE_KM,
            LEO_MAX_RANGE_KM,
            LEO_TO_MEO_MAX_RANGE_KM,
            LEO_TO_GEO_MAX_RANGE_KM,
            MEO_MAX_RANGE_KM,
            MEO_TO_GEO_MAX_RANGE_KM,
            GEO_MAX_RANGE_KM,
            SATELLITE_RANGE_MARGIN
        )
        
        # Normalize types - extract orbital type from nodeType
        def get_orbital_type(node_type: str) -> str:
            if 'GROUND' in node_type.upper():
                return 'GS'
            elif 'LEO' in node_type.upper():
                return 'LEO'
            elif 'MEO' in node_type.upper():
                return 'MEO'
            elif 'GEO' in node_type.upper():
                return 'GEO'
            elif 'SATELLITE' in node_type.upper():
                return 'LEO'
            elif 'AERIAL' in node_type.upper():
                return 'LEO'
            else:
                return 'LEO'  # Default
        
        src = get_orbital_type(source_type)
        dst = get_orbital_type(dest_type)
        
        # Sort to make lookup symmetric (GS-LEO == LEO-GS)
        pair = tuple(sorted([src, dst]))
        
        # Range lookup table based on node type pairs
        # Keys are sorted alphabetically: GEO < GS < LEO < MEO
        range_table = {
            ('GS', 'GS'): GS_MAX_DIRECT_RANGE_KM,
            ('GS', 'LEO'): GS_TO_LEO_MAX_RANGE_KM,
            ('GS', 'MEO'): GS_TO_MEO_MAX_RANGE_KM,
            ('GEO', 'GS'): GS_TO_GEO_MAX_RANGE_KM,  # GEO < GS alphabetically
            ('LEO', 'LEO'): LEO_MAX_RANGE_KM,
            ('LEO', 'MEO'): LEO_TO_MEO_MAX_RANGE_KM,
            ('GEO', 'LEO'): LEO_TO_GEO_MAX_RANGE_KM,  # GEO < LEO
            ('MEO', 'MEO'): MEO_MAX_RANGE_KM,
            ('GEO', 'MEO'): MEO_TO_GEO_MAX_RANGE_KM,  # GEO < MEO
            ('GEO', 'GEO'): GEO_MAX_RANGE_KM,
        }
        
        max_range = range_table.get(pair, LEO_MAX_RANGE_KM)
        
        # Apply margin for dynamic orbital positions (except GS-GS)
        if 'GS' not in pair or pair != ('GS', 'GS'):
            max_range *= SATELLITE_RANGE_MARGIN
        
        return max_range
    
    def _calculate_dijkstra_aligned_reward(
        self,
        current_node: Dict,
        next_node: Dict,
        distance: float,
        dest_terminal_pos: Dict
    ) -> float:
        """Calculate reward aligned with Dijkstra's edge weights"""
        base_distance_km = distance / M_TO_KM
        base_reward = -base_distance_km
        
        cpu = next_node.get('cpu', {}).get('utilization', 0)
        mem = next_node.get('memory', {}).get('utilization', 0)
        bw = next_node.get('bandwidth', {}).get('utilization', 0)
        max_util = max(cpu, mem, bw)
        
        if max_util >= self.drop_threshold:
            return REWARD_DROP_NODE
        
        if max_util >= self.penalty_threshold:
            excess = (max_util - self.penalty_threshold) / (UTILIZATION_MAX_PERCENT - self.penalty_threshold)
            penalty = base_distance_km * (self.penalty_multiplier - 1.0) * excess
            base_reward -= penalty
        
        current_pos = current_node.get('position')
        next_pos = next_node.get('position')
        
        if current_pos and next_pos and dest_terminal_pos:
            current_to_dest = self._calculate_distance(current_pos, dest_terminal_pos)
            next_to_dest = self._calculate_distance(next_pos, dest_terminal_pos)
            
            progress_km = (current_to_dest - next_to_dest) / M_TO_KM
            progress_reward = progress_km * DIJKSTRA_PROGRESS_SCALE
            
            if next_to_dest < DISTANCE_VERY_CLOSE_M:
                return REWARD_SUCCESS
            
            return base_reward + progress_reward
        
        return base_reward
    
    def _check_recent_progress(self, current_dist: float) -> bool:
        """Check if agent is making progress towards destination"""
        if len(self.recent_distances) < PROGRESS_CHECK_WINDOW_SIZE:
            return True
        
        last_distances = list(self.recent_distances)[-PROGRESS_CHECK_WINDOW_SIZE:]
        total_progress = last_distances[0] - current_dist
        
        if total_progress < PROGRESS_MIN_THRESHOLD_M:
            logger.debug(f"No progress detected: {total_progress:.1f}m in last {PROGRESS_CHECK_WINDOW_SIZE} steps")
            return False
        
        return True
    
    def get_path_result(self) -> Dict:
        """Get final path result - đảm bảo format đúng và đầy đủ"""
        if not self.path or len(self.path) < 2:
            # Return empty path if no path found
            return {
                'source': {
                    'terminalId': self.source_terminal.get('terminalId') if self.source_terminal else '',
                    'position': self.source_terminal.get('position') if self.source_terminal else {}
                },
                'destination': {
                    'terminalId': self.dest_terminal.get('terminalId') if self.dest_terminal else '',
                    'position': self.dest_terminal.get('position') if self.dest_terminal else {}
                },
                'path': [],
                'totalDistance': 0,
                'estimatedLatency': 0,
                'hops': 0,
                'algorithm': 'rl_optimized',
                'success': False
            }
        
        # Build path segments - đảm bảo có source terminal ở đầu
        path_segments = []
        
        # Always start with source terminal
        if self.source_terminal:
            path_segments.append({
                'type': 'terminal',
                'id': self.source_terminal.get('terminalId'),
                'name': self.source_terminal.get('terminalName', self.source_terminal.get('terminalId')),
                'position': self.source_terminal.get('position')
            })
        
        # Add all nodes from path (skip source terminal if it's already in path)
        for item in self.path:
            if 'terminalId' in item:
                # Skip source terminal if already added
                if item.get('terminalId') == self.source_terminal.get('terminalId'):
                    continue
                # Add destination terminal
                path_segments.append({
                    'type': 'terminal',
                    'id': item.get('terminalId'),
                    'name': item.get('terminalName', item.get('terminalId')),
                    'position': item.get('position')
                })
            elif 'nodeId' in item:
                path_segments.append({
                    'type': 'node',
                    'id': item.get('nodeId'),
                    'name': item.get('nodeName', item.get('nodeId')),
                    'position': item.get('position')
                })
        
        # Always end with destination terminal if not already there
        if (not path_segments or 
            path_segments[-1].get('id') != self.dest_terminal.get('terminalId')):
            path_segments.append({
                'type': 'terminal',
                'id': self.dest_terminal.get('terminalId'),
                'name': self.dest_terminal.get('terminalName', self.dest_terminal.get('terminalId')),
                'position': self.dest_terminal.get('position')
            })
        
        # Calculate total metrics
        total_distance = 0.0
        total_latency = 0.0
        
        for i in range(len(path_segments) - 1):
            pos1 = path_segments[i].get('position')
            pos2 = path_segments[i + 1].get('position')
            if pos1 and pos2:
                dist = self._calculate_distance(pos1, pos2)
                total_distance += dist
                
                speed_of_light = 299792458
                propagation_delay = (dist / speed_of_light) * 1000
                processing_delay = 5  # Default processing delay
                total_latency += propagation_delay + processing_delay
        
        is_success = self.terminated and len(path_segments) >= MIN_PATH_SEGMENTS  # At least: source_terminal, source_node, dest_node, dest_terminal
        
        return {
            'source': {
                'terminalId': self.source_terminal.get('terminalId'),
                'position': self.source_terminal.get('position')
            },
            'destination': {
                'terminalId': self.dest_terminal.get('terminalId'),
                'position': self.dest_terminal.get('position')
            },
            'path': path_segments,
            'totalDistance': round(total_distance / 1000, 2),
            'estimatedLatency': round(total_latency, 2),
            'hops': len(path_segments) - 1,
            'algorithm': 'rl_optimized',
            'success': is_success
        }