# env/RewardCalculator.py
from typing import Dict, Any

class RewardCalculator:
    
    def calculate_reward(self, qos: Dict, link_metric: Dict) -> float:
        """Tính Reward R(s, a) dựa trên việc đáp ứng QoS cho liên kết được chọn."""
        
        # 🚨 PHẠT CỰC ĐỘ: Nếu link hỏng hoặc BW gần như bằng 0 (dưới 1 Mbps)
        if not link_metric or not link_metric.get('isLinkActive', False) or link_metric.get('currentAvailableBandwidthMbps', 0.0) < 1.0:
            return -1500.0 

        reward = 0.0
        
        max_lat = qos.get('maxLatencyMs', 1000.0)
        link_lat = link_metric.get('latencyMs', 1000.0)
        min_bw = qos.get('minBandwidthMbps', 1.0)
        link_bw = link_metric.get('currentAvailableBandwidthMbps', 0.0)
        max_loss = qos.get('maxLossRate', 1.0)
        link_loss = link_metric.get('packetLossRate', 1.0)
        
        # 1. Latency (Tăng phạt)
        reward += 20.0 if link_lat <= max_lat else -250.0 
            
        # 2. Bandwidth (Tăng phạt mạnh mẽ nhất)
        reward += 20.0 if link_bw >= min_bw else -500.0 # Phạt nặng hơn nhiều
            
        # 3. Loss Rate (Tăng phạt)
        reward += 10.0 if link_loss <= max_loss else -150.0 
        
        # 4. LinkScore (Bonus)
        reward += link_metric.get('linkScore', 0.0) * 0.5 

        # Giới hạn Reward
        return max(-2000.0, min(reward, 100.0))