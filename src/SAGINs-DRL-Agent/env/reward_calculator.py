# env/RewardCalculator.py
from typing import Dict, Any, List

class RewardCalculator:
    
    """
        Improved Reward Calculator với path history tracking
        Giúp RL tránh routing loops và học được behavior tốt hơn Dijkstra
    """
    def __init__(self):
        # Service-specific QoS profiles
        self.service_profiles = {
            "VIDEO_STREAMING": {
                'priorityWeight': 1.0,
                'maxLatencyMs': 50.0,
                'minBandwidthMbps': 500.0,
                'maxLossRate': 0.02
            },
            "VOICE_CALL": {
                'priorityWeight': 1.5,
                'maxLatencyMs': 30.0,
                'minBandwidthMbps': 0.128,
                'maxLossRate': 0.01
            },
            "MESSAGING": {
                'priorityWeight': 0.7,
                'maxLatencyMs': 2000.0,
                'minBandwidthMbps': 0.01,
                'maxLossRate': 0.05
            },
            "FILE_TRANSFER": {
                'priorityWeight': 0.8,
                'maxLatencyMs': 5000.0,
                'minBandwidthMbps': 100.0,
                'maxLossRate': 0.001
            }
        }
        self.training_phase = "exploration"
    
    def calculate_reward(self, qos: Dict, link_metrics: Dict, source_node: Dict, 
                    next_hop_node: Dict, step: int, total_steps: int, 
                    reached_destination: bool = False, path_history: List[str] = None) -> float:
        """Simplified reward function - FOCUS ON SUCCESS"""
        
        if not self._is_link_usable(link_metrics):
            return -5.0  # GIẢM penalty
        
        # 1. 🎯 SUCCESS REWARD - QUAN TRỌNG NHẤT
        if reached_destination:
            step_bonus = (total_steps - step) * 10.0  # TĂNG bonus cho đường ngắn
            return 300.0 + step_bonus  # TĂNG success reward

        # 2. 🔄 LOOP PENALTY - GIẢM
        if path_history and self._is_loop_detected(next_hop_node, path_history):
            return -20.0  # GIẢM penalty
        
        reward = 0.0
        
        # 3. 📈 PROGRESS REWARD - Thưởng tiến độ đơn giản
        reward += 5.0  # Luôn thưởng cho việc chuyển tiếp
        
        # 4. 🔗 LINK QUALITY REWARD - ĐƠN GIẢN HÓA
        latency = link_metrics.get('latencyMs', 1000.0)
        loss_rate = link_metrics.get('packetLossRate', 1.0)
        
        # Latency reward
        if latency < 100.0:  # Chỉ penalty latency rất cao
            reward += 10.0 * (100.0 - latency) / 100.0
        else:
            reward -= 5.0
        
        # Packet loss penalty - GIẢM
        if loss_rate > 0.1:  # Chỉ penalty loss rate cao
            reward -= 10.0 * min(loss_rate, 1.0)
        else:
            reward += 5.0
        
        # 5. 🎪 EXPLORATION BONUS - Thưởng khám phá
        if path_history and next_hop_node.get('nodeId') not in path_history:
            reward += 8.0
        
        # 6. ⏱️ HOP PENALTY - RẤT NHẸ
        reward -= 0.5  # Giảm penalty
        
        return max(-10.0, min(50.0, reward))  # Giới hạn reward nhỏ hơn
    
    def _is_loop_detected(self, next_hop_node: Dict, path_history: List[str]) -> bool:
        """Phát hiện vòng lặp dựa trên path history"""
        if not path_history or not next_hop_node:
            return False
            
        next_node_id = next_hop_node.get('nodeId')
        if not next_node_id:
            return False
            
        # Kiểm tra nếu next node đã xuất hiện trong path history
        return next_node_id in path_history
    
    def _calculate_progress_reward(self, source_node: Dict, next_hop_node: Dict, 
                                 qos: Dict, path_history: List[str]) -> float:
        """Thưởng cho việc tiến gần đến đích - improved version"""
        try:
            dest_id = qos.get('destinationNodeId')
            if not dest_id:
                return 1.0  # Default progress reward
            
            src_pos = source_node.get('position', {})
            next_pos = next_hop_node.get('position', {})
            dest_node = qos.get('destinationNodeInfo', {})
            dest_pos = dest_node.get('position', {})
            
            # Nếu có đủ position data, tính toán khoảng cách
            if src_pos and next_pos and dest_pos:
                # Tính khoảng cách từ source và next_hop đến destination
                src_to_dest = self._calculate_distance_3d(src_pos, dest_pos)
                next_to_dest = self._calculate_distance_3d(next_pos, dest_pos)
                
                if src_to_dest > 0:
                    progress_ratio = (src_to_dest - next_to_dest) / src_to_dest
                    # Thưởng 5-15 points cho việc tiến gần đích
                    return max(0.0, progress_ratio * 15.0)
            
            # Fallback: heuristic progress reward
            if path_history:
                # Nếu path đang dài, thưởng ít hơn để tránh đường vòng
                path_length_penalty = min(len(path_history) * 0.5, 5.0)
                return 5.0 - path_length_penalty
            else:
                return 3.0  # Default progress reward
                
        except Exception as e:
            print(f"⚠️  Progress reward calculation error: {e}")
            return 2.0
    
    def _calculate_distance_3d(self, pos1: Dict, pos2: Dict) -> float:
        """Tính khoảng cách 3D giữa hai positions"""
        try:
            lat1, lon1, alt1 = pos1.get('latitude', 0), pos1.get('longitude', 0), pos1.get('altitude', 0)
            lat2, lon2, alt2 = pos2.get('latitude', 0), pos2.get('longitude', 0), pos2.get('altitude', 0)
            
            # Simple Euclidean distance approximation
            distance = ((lat2 - lat1)**2 + (lon2 - lon1)**2 + (alt2 - alt1)**2) ** 0.5
            return max(0.1, distance)  # Avoid division by zero
        except:
            return 1.0  # Default distance
    
    def _calculate_link_quality_reward(self, link_metrics: Dict, service_profile: Dict) -> float:
        """Tính toán thưởng dựa trên chất lượng liên kết - FIX math errors"""
        reward = 0.0
        
        latency = link_metrics.get('latencyMs', 1000.0)
        bandwidth = link_metrics.get('currentAvailableBandwidthMbps', 0.0)
        loss_rate = link_metrics.get('packetLossRate', 1.0)
        link_score = link_metrics.get('linkScore', 0.0)
        
        max_latency = service_profile['maxLatencyMs']
        min_bandwidth = service_profile['minBandwidthMbps']
        max_loss_rate = service_profile['maxLossRate']
        
        # FIX: Avoid division by zero và negative values
        max_latency = max(1.0, max_latency)
        min_bandwidth = max(0.1, min_bandwidth)
        max_loss_rate = max(0.001, max_loss_rate)
        
        # FIX: Ensure values are reasonable
        latency = max(0.1, latency)
        bandwidth = max(0.0, bandwidth)
        loss_rate = max(0.0, min(1.0, loss_rate))
        
        # 1. 📞 LATENCY REWARD
        if latency <= max_latency:
            latency_quality = (max_latency - latency) / max_latency
            reward += 25.0 * max(0.0, latency_quality)
        else:
            latency_penalty = (latency - max_latency) / max_latency
            reward -= 20.0 * min(max(0.0, latency_penalty), 2.0)
            
        # 2. 📊 BANDWIDTH REWARD
        if bandwidth >= min_bandwidth:
            bandwidth_surplus = (bandwidth - min_bandwidth) / min_bandwidth
            reward += 20.0 * min(max(0.0, bandwidth_surplus), 2.0)
        else:
            bandwidth_deficit = (min_bandwidth - bandwidth) / min_bandwidth
            reward -= 15.0 * min(max(0.0, bandwidth_deficit), 2.0)
        
        # 3. 📉 PACKET LOSS PENALTY
        if loss_rate > max_loss_rate:
            loss_penalty = (loss_rate - max_loss_rate) / max_loss_rate
            reward -= 25.0 * min(max(0.0, loss_penalty), 3.0)
        else:
            loss_quality = (max_loss_rate - loss_rate) / max_loss_rate
            reward += 15.0 * max(0.0, loss_quality)
        
        # 4. 🏆 LINK SCORE BONUS
        if link_score > 50.0:
            reward += min(link_score * 0.1, 10.0)

        return reward
        
    def _is_link_usable(self, link_metrics: Dict) -> bool:
        """Kiểm tra link có usable không - relaxed conditions"""
        if not link_metrics or not link_metrics.get('isLinkActive', True):
            return False
        if link_metrics.get('currentAvailableBandwidthMbps', 0.0) < 0.1:
            return False
        if link_metrics.get('packetLossRate', 1.0) > 0.9:  # Tăng ngưỡng loss
            return False
        return True

    def update_training_phase(self, episode: int, total_episodes: int):
        """Cập nhật phase training để điều chỉnh reward strategy"""
        progress = episode / total_episodes
        
        if progress < 0.3:
            self.training_phase = "exploration"
        elif progress < 0.7:
            self.training_phase = "exploitation" 
        else:
            self.training_phase = "fine_tuning"