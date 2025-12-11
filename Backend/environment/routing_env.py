"""
Optimized Routing Environment for SAGIN
Environment được tối ưu cho training hiệu quả và performance
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
from collections import deque

from environment.state_builder import RoutingStateBuilder

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
        max_steps: int = 8  # GIẢM: 10 → 8 để force shorter paths
    ):
        super().__init__()
        
        self.config = config or {}
        self.nodes = nodes
        self.terminals = terminals
        self.max_steps = max_steps
        
        # State builder
        self.state_builder = RoutingStateBuilder(config)
        
        # Action space
        max_actions = min(len(nodes), self.state_builder.max_nodes)
        self.action_space = spaces.Discrete(max_actions)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-1.0,  # Thay đổi để ổn định training
            high=2.0,
            shape=(self.state_builder.state_dimension,),
            dtype=np.float32
        )
        
        # Episode state
        self.source_terminal = None
        self.dest_terminal = None
        self.current_node = None
        self.path = []
        self.visited_nodes = set()
        self.step_count = 0
        self.total_distance = 0.0
        self.total_latency = 0.0
        self.service_qos = None
        self.terminated = False  # Track if episode terminated successfully
        
        # Optimized reward configuration - ƯU TIÊN GIẢM HOP/DISTANCE/LATENCY
        reward_config = self.config.get('reward', {})
        self.success_reward = reward_config.get('success_reward', 200.0)
        self.failure_penalty = reward_config.get('failure_penalty', -10.0)  # Giảm từ -30 xuống -10
        self.step_penalty = reward_config.get('step_penalty', -10.0)  # TĂNG: -8.0 → -10.0 - MỖI STEP ĐỀU TỐN KÉM
        self.hop_penalty = reward_config.get('hop_penalty', -15.0)  # TĂNG: -12.0 → -15.0 - HOP LÀ TỐN KÉM NHẤT
        self.ground_station_hop_penalty = reward_config.get('ground_station_hop_penalty', -25.0)  # TĂNG: -20 → -25
        self.progress_reward_scale = reward_config.get('progress_reward_scale', 80.0)  # GIẢM: 150 → 80 - Không thưởng quá nhiều cho progress
        self.distance_reward_scale = reward_config.get('distance_reward_scale', 10.0)  # Tăng từ 5.0 → 10.0: Distance quan trọng
        self.quality_reward_scale = reward_config.get('quality_reward_scale', 10.0)  # Giảm từ 30.0 → 10.0: Resource là mục tiêu thứ 2
        self.proximity_bonus_scale = reward_config.get('proximity_bonus_scale', 50.0)  # Bonus khi đến gần destination
        
        # Cache untuk performance
        self._node_cache = {node['nodeId']: node for node in nodes}
        self._terminal_cache = {terminal['terminalId']: terminal for terminal in terminals}
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment với optimizations và explicit ground stations"""
        super().reset(seed=seed)
        
        # Reset terminated flag
        self.terminated = False
        
        # Get terminals và ground stations từ options hoặc random
        source_ground_station = None
        dest_ground_station = None
        
        if options:
            source_terminal_id = options.get('source_terminal_id')
            dest_terminal_id = options.get('dest_terminal_id')
            self.service_qos = options.get('service_qos')
            
            # 🔥 NEW: Get explicit ground stations from options
            source_ground_station = options.get('source_ground_station')
            dest_ground_station = options.get('dest_ground_station')
            
            self.source_terminal = self._terminal_cache.get(source_terminal_id)
            self.dest_terminal = self._terminal_cache.get(dest_terminal_id)
        else:
            # Random terminals
            if len(self.terminals) < 2:
                raise ValueError("Need at least 2 terminals")
            
            indices = self.np_random.choice(len(self.terminals), size=2, replace=False)
            self.source_terminal = self.terminals[indices[0]]
            self.dest_terminal = self.terminals[indices[1]]
        
        if not self.source_terminal or not self.dest_terminal:
            raise ValueError("Source or destination terminal not found")
        
        # 🔥 FIX: Sử dụng explicit ground stations nếu có, otherwise tìm optimal
        if source_ground_station:
            self.current_node = source_ground_station
            logger.info(f"🛰️ RL starting from explicit source GS: {source_ground_station['nodeId']}")
        else:
            # Tìm initial node thông minh
            self.current_node = self._find_optimal_initial_node(
                self.source_terminal, self.dest_terminal
            )
        
        if not self.current_node:
            operational_nodes = [
                n for n in self.nodes 
                if n.get('isOperational', True) and n.get('position')
            ]
            if operational_nodes:
                # Chọn node gần destination nhất
                dest_pos = self.dest_terminal.get('position')
                self.current_node = min(
                    operational_nodes,
                    key=lambda n: self._calculate_distance(
                        n.get('position'), dest_pos
                    ) if n.get('position') else float('inf')
                )
            else:
                raise ValueError("No operational nodes available")
        
        # 🔥 NEW: Store dest_ground_station for validation
        self.dest_ground_station = dest_ground_station
        
        # Reset episode state
        self.path = [self.source_terminal, self.current_node]
        self.visited_nodes = {self.current_node.get('nodeId')}
        self.step_count = 0
        self.total_distance = 0.0
        self.total_latency = 0.0
        
        # Build state
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
        """Optimized step function với reward engineering tiên tiến"""
        self.step_count += 1
        
        # Lấy available nodes với stress-aware filtering
        filtered_nodes = self.state_builder._smart_node_filtering(
            self.nodes, self.source_terminal, self.dest_terminal, 
            self.current_node, list(self.visited_nodes)
        )
        
        # Filter out problematic nodes in stress scenarios (optional - can be disabled)
        # This helps RL learn to avoid bad nodes
        stress_aware_nodes = self._filter_stress_problematic_nodes(filtered_nodes)
        if len(stress_aware_nodes) > 0:
            filtered_nodes = stress_aware_nodes
        
        # Validate action và chọn next node
        if action < 0 or action >= len(filtered_nodes) or not filtered_nodes:
            # Fallback strategy
            next_node = self._find_fallback_node()
            if not next_node:
                # No valid nodes, end episode
                state = self.state_builder.build_state(
                    self.nodes, self.source_terminal, self.dest_terminal,
                    self.current_node, self.service_qos, list(self.visited_nodes)
                )
                return state, self.failure_penalty, True, False, {'error': 'no_valid_nodes'}
        else:
            next_node = filtered_nodes[action]
        
        # Loop detection
        next_node_id = next_node.get('nodeId')
        if next_node_id in self.visited_nodes:
            # Loop penalty
            reward = -20.0
            terminated = False
            truncated = self.step_count >= self.max_steps
            
            state = self.state_builder.build_state(
                self.nodes, self.source_terminal, self.dest_terminal,
                self.current_node, self.service_qos, list(self.visited_nodes)
            )
            
            info = {
                'path': self.path.copy(),
                'loop_detected': True,
                'current_node': next_node_id,
                'hops': len(self.path) - 1
            }
            return state, reward, terminated, truncated, info
        
        # Thêm node vào path
        self.path.append(next_node)
        self.visited_nodes.add(next_node_id)
        
        # Tính metrics cho hop này
        current_pos = self.current_node.get('position')
        next_pos = next_node.get('position')
        hop_distance = self._calculate_distance(current_pos, next_pos)
        self.total_distance += hop_distance
        
        # Tính latency
        speed_of_light = 299792458
        propagation_delay = (hop_distance / speed_of_light) * 1000
        processing_delay = next_node.get('nodeProcessingDelayMs', 5)
        hop_latency = propagation_delay + processing_delay
        self.total_latency += hop_latency
        
        # Get node types for reward calculation
        current_node_type = self.current_node.get('nodeType', '')
        next_node_type = next_node.get('nodeType', '')
        
        # Get connection counts để tính utilization thực tế
        current_connection_count = get_terminal_connection_count(self.current_node.get('nodeId'))
        next_connection_count = get_terminal_connection_count(next_node.get('nodeId'))
        
        # Tính utilization thực tế (mỗi terminal ~4-10% utilization)
        current_node_utilization = self.current_node.get('resourceUtilization', 0) + (current_connection_count * 7.0)
        next_node_utilization = next_node.get('resourceUtilization', 0) + (next_connection_count * 7.0)
        
        # Penalty đặc biệt cho ground station hops - LUÔN PENALTY trừ khi load balancing
        initial_reward = 0.0
        if current_node_type == 'GROUND_STATION' and next_node_type == 'GROUND_STATION':
            # LUÔN penalty GS→GS, trừ khi current GS quá tải VÀ next GS ít tải hơn
            if current_node_utilization > 80.0 and next_node_utilization < current_node_utilization - 20.0:
                # Load balancing case: current GS quá tải, next GS tốt hơn
                initial_reward = 5.0  # Bonus cho load balancing
                logger.debug(f"✅ Load balancing bonus: current={current_node_utilization:.1f}%, next={next_node_utilization:.1f}%")
            else:
                # Normal case: LUÔN penalty GS→GS
                initial_reward = self.ground_station_hop_penalty  # -15.0 (tăng từ -5.0)
                logger.debug(f"⚠️ GS→GS penalty: {initial_reward}")
        
        # Kiểm tra destination reached
        dest_pos = self.dest_terminal.get('position')
        dist_to_dest = self._calculate_distance(next_pos, dest_pos)
        
        # 🔥 ENHANCED: Check if we reached dest ground station explicitly
        reached_dest_gs = False
        if hasattr(self, 'dest_ground_station') and self.dest_ground_station:
            reached_dest_gs = (next_node_id == self.dest_ground_station['nodeId'])
        
        # Điều kiện success - STRICT: Chỉ accept khi thực sự đến destination GS
        is_ground_station = next_node_type == 'GROUND_STATION'
        is_near_dest = dist_to_dest < 500000  # Tightened: Within 500km only
        
        terminated = False
        reward = initial_reward  # Start with ground station hop penalty if applicable
        
        has_min_hops = len(self.path) >= 3  # Ít nhất 2 hops (source GS + 1 satellite + current)
        
        # 🎯 STRICT SUCCESS: Chỉ accept khi:
        # 1. Reached exact dest GS (best case)
        # 2. GS node AND very close to destination (<500km)
        # 3. Has minimum hops AND close to destination (<1000km)
        if reached_dest_gs or \
           (is_ground_station and is_near_dest and has_min_hops) or \
           (has_min_hops and dist_to_dest < 1000000):  # Tightened từ 2000km xuống 1000km
            # Success!
            self.path.append(self.dest_terminal)
            terminated = True
            self.terminated = True  # Mark as successfully terminated
            
            # Base success reward
            reward = self.success_reward
            
            # 🔥 BONUS: Extra reward if reached exact dest GS
            if reached_dest_gs:
                reward += 50.0
                logger.info(f"🎯 RL reached exact destination GS: {self.dest_ground_station['nodeId']}")
            
            # QoS compliance bonus
            if self.service_qos:
                max_latency = self.service_qos.get('maxLatencyMs', float('inf'))
                if self.total_latency <= max_latency:
                    reward += 30.0  # QoS bonus
                else:
                    reward -= 15.0  # QoS violation penalty
            
            # Path efficiency bonus/penalty - MỤC TIÊU SỐ 1
            num_hops = len(self.path) - 2
            optimal_hops = self._estimate_optimal_hops()
            
            if num_hops <= optimal_hops:
                efficiency_bonus = (optimal_hops - num_hops) * 20.0  # TĂNG MẠNH: 10.0 → 20.0: THƯỞNG CỰC LỚN cho path ngắn
                reward += efficiency_bonus
            else:
                efficiency_penalty = (num_hops - optimal_hops) * 15.0  # TĂNG MẠNH: 5.0 → 15.0: PENALTY CỰC LỚN cho path dài
                reward -= efficiency_penalty
                
            # 🔥 EXTRA PENALTY cho paths quá dài (>5 hops) - GIẢM threshold
            if num_hops > 5:  # Giảm từ 6 xuống 5
                extra_penalty = (num_hops - 5) ** 2 * 30.0  # TĂNG: 20.0 → 30.0
                reward -= extra_penalty
                logger.warning(f"⚠️ Path too long: {num_hops} hops, extra penalty: -{extra_penalty}")
                
            # Distance efficiency - MỤC TIÊU SỐ 1
            direct_distance = self._calculate_distance(
                self.source_terminal.get('position'),
                self.dest_terminal.get('position')
            )
            # 🆕 FIX: Prevent ZeroDivisionError when source = destination
            if direct_distance > 0:
                distance_ratio = self.total_distance / direct_distance
                if distance_ratio < 1.2:  # Rất hiệu quả (<20% detour)
                    reward += 30.0  # Tăng từ 20.0: THƯNG LỚN cho đường thẳng
                elif distance_ratio < 1.5:  # Hiệu quả (<50% detour)
                    reward += 15.0  # Tăng từ 10.0
                elif distance_ratio > 3.0:  # Quá vòng (>200% detour)
                    reward -= 20.0  # Tăng từ 10.0: PENALTY LỚN cho đường dài
            else:
                # Source = Destination (direct_distance = 0), max bonus
                reward += 50.0
                
        else:
            # Still routing - tính progressive reward với proximity bonus
            prev_dist = self._calculate_distance(
                self.current_node.get('position'), dest_pos
            )
            progress = prev_dist - dist_to_dest
            
            # Progressive rewards với detour penalty
            if progress > 0:
                # Progress reward - khuyến khích tiến gần destination
                reward += progress / 100000.0 * self.progress_reward_scale  # Scale đã giảm xuống 80.0
            else:
                # 🔥 DETOUR PENALTY: Đi xa destination = penalty MỰC NẶNG
                detour_penalty = abs(progress) / 50000.0 * 30.0  # Penalty lớn hơn progress reward
                reward -= detour_penalty
                logger.debug(f"⚠️ Detour penalty: -{detour_penalty:.2f} (moved away from dest by {abs(progress)/1000:.1f}km)")
            
            # Distance penalty
            reward -= hop_distance / 10000000.0 * self.distance_reward_scale
            # Step và hop penalties (tăng để tránh quá nhiều hops)
            reward += self.step_penalty  # Full penalty cho mỗi step
            reward += self.hop_penalty  # Full penalty cho mỗi hop
            
            # Satellite bonus GIẢM MẠNH: Ưu tiên satellites nhưng KHÔNG override hop penalty
            # Net effect: satellite hop = -15 (hop) + 5 (satellite) = -10 (vẫn penalty)
            if next_node_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']:
                satellite_bonus = 3.0  # GIẢM từ 15.0 → 3.0 - Chỉ bonus nhỏ
                if next_node_type == 'LEO_SATELLITE':
                    satellite_bonus = 5.0  # GIẢM từ 20.0 → 5.0 (LEO tốt hơn nhưng vẫn bị hop penalty)
                elif next_node_type == 'MEO_SATELLITE':
                    satellite_bonus = 4.0  # GIẢM từ 18.0 → 4.0
                reward += satellite_bonus
                logger.debug(f"✅ Satellite hop bonus: {satellite_bonus} for {next_node_type} (net với hop penalty: {satellite_bonus - 15.0})")
            
            # Penalty tăng dần cho nhiều hops (exponential penalty) - CỰC KỲ NGHIÊM KHẮC
            num_hops = len(self.path) - 1
            if num_hops > 3:  # GIẢM threshold từ 4 xuống 3 - Force RL học đường ngắn
                excess_hops = num_hops - 3
                excess_penalty = excess_hops * excess_hops * 20.0  # TĂNG: 10.0 → 20.0 - PENALTY CỰC LỚN
                reward -= excess_penalty
                logger.debug(f"⚠️ Excess hops penalty: -{excess_penalty} for {num_hops} hops (threshold=3)")
            
            # Proximity bonus - thưởng khi đến gần destination (tăng scale)
            if dist_to_dest < 1000000:  # Trong 1000km
                proximity_bonus = (1000000 - dist_to_dest) / 1000000.0 * self.proximity_bonus_scale * 2.0
                reward += proximity_bonus
            elif dist_to_dest < 2000000:  # Trong 2000km
                proximity_bonus = (2000000 - dist_to_dest) / 2000000.0 * self.proximity_bonus_scale
                reward += proximity_bonus
            
            # Node quality reward - MỤC TIÊU THỨ 2 (sau khi giảm hop/distance)
            node_quality = self.state_builder._compute_node_quality(next_node)
            quality_reward = node_quality * self.quality_reward_scale  # 0-10.0 points
            reward += quality_reward
            
            # Extra bonus for EXCELLENT nodes (quality > 0.8) - Giảm để không override hop penalty
            if node_quality > 0.8:
                excellent_bonus = 5.0  # Giảm từ 15.0 → 5.0
                reward += excellent_bonus
                logger.debug(f"✨ Excellent node bonus: {excellent_bonus} (quality={node_quality:.2f})")
            # Bonus for GOOD nodes (quality > 0.6)
            elif node_quality > 0.6:
                good_bonus = 3.0  # Giảm từ 8.0 → 3.0
                reward += good_bonus
                logger.debug(f"✅ Good node bonus: {good_bonus} (quality={node_quality:.2f})")
            # Penalty for BAD nodes (quality < 0.3) - GIỮNGUYÊN vì tránh node tồi vẫn quan trọng
            elif node_quality < 0.3:
                bad_penalty = -20.0  # Giữ nguyên
                reward += bad_penalty
                logger.debug(f"❌ Bad node penalty: {bad_penalty} (quality={node_quality:.2f})")
            
            # Resource utilization penalty - SỬ DỤNG UTILIZATION THỰC TẾ
            # Nhiều terminals quanh GS → utilization cao → RL nên tìm đường vòng qua GS khác
            # next_node_utilization đã được tính ở trên (bao gồm connection count)
            estimated_utilization = min(100.0, next_node_utilization)
            
            if estimated_utilization > 90:
                reward -= 40.0  # Tăng từ 30 → 40 - RẤT NGUY HIỂM
            elif estimated_utilization > 80:
                reward -= 25.0  # Tăng từ 20 → 25 - Nguy hiểm
            elif estimated_utilization > 70:
                reward -= 15.0  # Tăng từ 12 → 15 - Cảnh báo cao
            elif estimated_utilization > 60:
                reward -= 8.0   # Giữ nguyên - Cảnh báo
            elif estimated_utilization < 30:
                reward += 10.0  # Giữ nguyên: Bonus cho node ít tải
            
            # Bonus/Penalty dựa trên số terminals (cho GS)
            if next_node_type == 'GROUND_STATION':
                if next_connection_count <= 2:
                    reward += 8.0  # Tăng từ 5 → 8: Bonus lớn cho GS ít tải
                elif next_connection_count <= 5:
                    reward += 3.0  # Tăng từ 2 → 3: Bonus cho GS tải vừa
                elif next_connection_count > 15:
                    reward -= 25.0  # Tăng từ 10 → 25: Penalty RẤT LỚN cho GS quá tải
                elif next_connection_count > 10:
                    reward -= 15.0  # Penalty lớn cho GS tải cao
                
            # Battery level penalty - tránh nodes có battery thấp
            battery_level = next_node.get('batteryChargePercent', 100)
            if battery_level < 20:
                reward -= 10.0  # Battery rất thấp - penalty lớn
            elif battery_level < 30:
                reward -= 5.0  # Battery thấp - penalty vừa
            elif battery_level < 50:
                reward -= 2.0  # Battery trung bình - penalty nhỏ
                
            # Loss rate penalty - tăng penalty MẠNh
            loss_rate = next_node.get('packetLossRate', 0)
            if loss_rate > 0.1:
                reward -= loss_rate * 50.0  # Tăng từ 20 → 50: Rất cao loss - penalty rất lớn
            elif loss_rate > 0.05:
                reward -= loss_rate * 30.0  # Tăng từ 10 → 30: Cao loss - penalty lớn
            elif loss_rate > 0:
                reward -= loss_rate * 10.0  # Penalty cho bất kỳ loss rate nào
        
        # Check truncation - giảm penalty và tăng partial success reward
        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            reward += self.failure_penalty
            
            # Partial success reward based on progress - tăng reward
            initial_dist = self._calculate_distance(
                self.source_terminal.get('position'),
                self.dest_terminal.get('position')
            )
            current_dist = dist_to_dest
            if initial_dist > 0:
                progress_made = (initial_dist - current_dist) / initial_dist
                # Thưởng cho bất kỳ progress nào - tăng mạnh
                reward += progress_made * 200.0  # Tăng từ 100.0
                # Bonus nếu đến gần destination - tăng scale
                if dist_to_dest < 500000:  # Trong 500km
                    reward += 100.0  # Bonus lớn cho việc đến gần
                elif dist_to_dest < 1000000:  # Trong 1000km
                    reward += 50.0
                elif dist_to_dest < 2000000:  # Trong 2000km
                    reward += 25.0
        
        # Update current node
        self.current_node = next_node
        
        # Build next state
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
        """Fallback strategy khi không có valid actions"""
        # Ưu tiên ground stations gần destination
        fallback_node = self._find_best_ground_station(self.dest_terminal, self.nodes)
        if fallback_node:
            return fallback_node
        
        # Fallback đến node operational bất kỳ gần destination
        operational_nodes = [
            n for n in self.nodes 
            if n.get('isOperational', True) and n.get('position')
        ]
        
        if not operational_nodes:
            return None
            
        dest_pos = self.dest_terminal.get('position')
        return min(
            operational_nodes,
            key=lambda n: self._calculate_distance(
                n.get('position'), dest_pos
            ) if n.get('position') else float('inf')
        )
    
    def _estimate_optimal_hops(self) -> int:
        """Ước tính số hops tối ưu cho path - STRICT để force shorter paths"""
        direct_dist = self._calculate_distance(
            self.source_terminal.get('position'),
            self.dest_terminal.get('position')
        )
        
        # 🔥 STRICT: Force RL to learn shortest paths
        # Typical optimal: Terminal → GS → LEO → GS → Terminal = 3-4 hops
        avg_hop_dist = 3000000  # 3000km (tăng từ 2500km để giảm estimated hops)
        optimal_hops = max(3, int(direct_dist / avg_hop_dist) + 2)  # +2 for GS hops
        
        return min(optimal_hops, 5)  # Max 5 hops (giảm từ 6, STRICT!)
    
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
        
        # Nếu filter quá nhiều, giữ lại một số nodes tốt nhất
        if len(filtered) < 3 and len(nodes) > 0:
            # Sort by quality và giữ top nodes
            nodes_sorted = sorted(
                nodes,
                key=lambda n: (
                    -n.get('resourceUtilization', 0),  # Lower is better
                    -n.get('batteryChargePercent', 100),  # Higher is better
                    n.get('packetLossRate', 0)  # Lower is better
                )
            )
            filtered = nodes_sorted[:max(3, len(nodes) // 2)]
        
        return filtered
    
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
        
        # Check if path successfully reached destination
        is_success = self.terminated and len(path_segments) >= 4  # At least: source_terminal, source_node, dest_node, dest_terminal
        
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