import numpy as np
from scipy.spatial.distance import euclidean

class SAGINsEnv:
    def __init__(self, nodes, packet):
        self.nodes = nodes
        self.packet = packet.copy()
        self.current_node = packet['currentHoldingNodeId']
        self.dest_node = packet['stationDest']
        self.visited = {self.current_node}
        self.ttl = packet['TTL']
        self.node_ids = list(nodes.keys())
        self.service_types = {'VIDEO_STREAM': 0, 'AUDIO_CALL': 1, 'IMAGE_TRANSFER': 2, 'TEXT_MESSAGE': 3}
        self.service_defaults = {
            'VIDEO_STREAM': {'maxAcceptableLatencyMs': 150, 'maxAcceptableLossRate': 0.01, 'latencyFactor': 1.2},
            'AUDIO_CALL': {'maxAcceptableLatencyMs': 100, 'maxAcceptableLossRate': 0.005, 'latencyFactor': 1.0},
            'IMAGE_TRANSFER': {'maxAcceptableLatencyMs': 200, 'maxAcceptableLossRate': 0.02, 'latencyFactor': 1.1},
            'TEXT_MESSAGE': {'maxAcceptableLatencyMs': 300, 'maxAcceptableLossRate': 0.03, 'latencyFactor': 1.0}
        }
        self.hop_count = 0  # Thêm thuộc tính để đếm số hop
        self.reset()
    
    def reset(self):
        self.current_node = self.packet['currentHoldingNodeId']
        self.visited = {self.current_node}
        self.packet['maxAcceptableLatencyMs'] = self.packet.get('maxAcceptableLatencyMs', self.service_defaults.get(self.packet.get('serviceType', 'VIDEO_STREAM'), self.service_defaults['VIDEO_STREAM'])['maxAcceptableLatencyMs'])
        self.packet['maxAcceptableLossRate'] = self.packet.get('maxAcceptableLossRate', self.service_defaults.get(self.packet.get('serviceType', 'VIDEO_STREAM'), self.service_defaults['VIDEO_STREAM'])['maxAcceptableLossRate'])
        self.packet['accumulatedDelayMs'] = self.packet.get('accumulatedDelayMs', 0)
        self.ttl = self.packet['TTL']
        self.hop_count = 0  # Reset hop count khi reset
        return self._get_obs()
    
    def step(self, action_path):
        if not action_path or self.current_node == self.dest_node:
            return self._get_obs(), 0, True
        next_node = action_path[0]
        if next_node not in self.node_ids or next_node in self.visited or not self._is_connected(self.current_node, next_node):
            return self._get_obs(), -10, False
        
        latency, loss_rate, bandwidth_mbps = self._calculate_link_metrics(self.current_node, next_node, self.packet)
        self.packet['accumulatedDelayMs'] += latency
        self.hop_count += 1  # Tăng số hop mỗi lần chuyển
        reward = 0
        
        # Reward based on resource optimization
        if self.packet['accumulatedDelayMs'] <= self.packet['maxAcceptableLatencyMs']:
            reward += 5 - (latency / 100)  # Reward for low latency
        else:
            reward -= 5
        
        if loss_rate <= self.packet['maxAcceptableLossRate']:
            reward += 5 - (loss_rate * 500)  # Reward for low loss rate
        else:
            reward -= 5
        
        min_bandwidth = self.packet.get('minRequiredBandwidthMbps', 1)
        if bandwidth_mbps >= min_bandwidth:
            reward += 5 + (bandwidth_mbps / 100)  # Reward for high bandwidth
        else:
            reward -= 3
        
        self.current_node = next_node
        self.visited.add(next_node)
        self.ttl -= 1
        
        if self.ttl <= 0:
            reward -= 10
        if self.current_node == self.dest_node:
            reward += 100 - self.packet['accumulatedDelayMs'] * 0.1  # Phần thưởng khi đến đích nhanh
            return self._get_obs(), reward, True
        reward -= 1  # Step penalty
        
        return self._get_obs(), reward, False
    
    def _get_obs(self):
        node = self.nodes.get(self.current_node, {})
        pos = [node.get('position', {}).get(k, 0) for k in ['latitude', 'longitude', 'altitude']]
        vel = [node.get('velocity', {}).get(k, 0) for k in ['velocityX', 'velocityY', 'velocityZ']]
        comm = [node.get('communication', {}).get(k, 0) for k in ['frequencyGHz', 'bandwidthMHz', 'maxRangeKm']]
        
        service_idx = self.service_types.get(self.packet.get('serviceType', 'VIDEO_STREAM'), 0)
        service_onehot = [1 if i == service_idx else 0 for i in range(4)]
        packet_feats = [self.packet.get(k, 0) for k in ['payloadSizeByte', 'priorityLevel', 'accumulatedDelayMs']] + service_onehot
        
        return np.concatenate((pos + vel + comm, packet_feats))
    
    def _is_connected(self, node1, node2):
        pos1 = np.array([self.nodes.get(node1, {}).get('position', {}).get(k, 0) for k in ['latitude', 'longitude', 'altitude']])
        pos2 = np.array([self.nodes.get(node2, {}).get('position', {}).get(k, 0) for k in ['latitude', 'longitude', 'altitude']])
        dist = euclidean(pos1, pos2)
        max_range1 = self.nodes.get(node1, {}).get('communication', {}).get('maxRangeKm', float('inf'))  # No limit for GS
        max_range2 = self.nodes.get(node2, {}).get('communication', {}).get('maxRangeKm', float('inf'))  # No limit for GS
        return dist <= min(max_range1, max_range2)
    
    def _calculate_link_metrics(self, node1, node2, packet):
        pos1 = np.array([self.nodes.get(node1, {}).get('position', {}).get(k, 0) for k in ['latitude', 'longitude', 'altitude']])
        pos2 = np.array([self.nodes.get(node2, {}).get('position', {}).get(k, 0) for k in ['latitude', 'longitude', 'altitude']])
        dist_km = euclidean(pos1, pos2)
        speed_light_km_ms = 300
        latency = (dist_km / speed_light_km_ms) * 2
        service_type = packet.get('serviceType', 'VIDEO_STREAM')
        latency_factor = self.service_defaults.get(service_type, self.service_defaults['VIDEO_STREAM'])['latencyFactor']
        latency *= latency_factor
        
        # New loss rate calculation based on hop count
        base_loss_rate = 0.001  # Base loss rate per hop (0.1%)
        additional_loss_per_hop = 0.0005  # Additional 0.05% per hop
        total_loss_rate = base_loss_rate + (self.hop_count * additional_loss_per_hop)
        
        # Adjust based on distance (optional multiplier)
        if dist_km > 1000:
            total_loss_rate *= 1.5  # Increase by 50% for long distances
                
        bandwidth_mbps = min(self.nodes.get(node1, {}).get('communication', {}).get('bandwidthMHz', 1), self.nodes.get(node2, {}).get('communication', {}).get('bandwidthMHz', 1)) * 0.1
        return latency, total_loss_rate, bandwidth_mbps