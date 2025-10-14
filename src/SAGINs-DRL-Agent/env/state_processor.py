# env/state_processor.py
import numpy as np
from typing import Dict, Any

class StateProcessor:
    """Xử lý state từ node info thành vector cho DQN"""
    
    def __init__(self, max_neighbors: int = 10):
        # XÓA dòng này: super().__init__(max_neighbors)
        self.MAX_NEIGHBORS = max_neighbors
        self.SERVICE_FEATURES = 4  # Sửa lỗi chính tả: SEVICE -> SERVICE
        self.QOS_FEATURES = 3      # Số lượng đặc trưng QoS trên link
        self.SOURCE_FEATURES = 9   # Số lượng đặc trưng node nguồn
        self.LINK_FEATURES_PER_NEIGHBOR = 6
        
        # Tính state size
        self.STATE_SIZE = (
            self.SERVICE_FEATURES + 
            self.QOS_FEATURES + 
            self.SOURCE_FEATURES + 
            self.MAX_NEIGHBORS * self.LINK_FEATURES_PER_NEIGHBOR
        )
        
        print(f"🔧 StateProcessor initialized: state_size={self.STATE_SIZE}, max_neighbors={self.MAX_NEIGHBORS}")
        
    def json_to_state_vector(self, data: Dict[str, Any]) -> np.ndarray:
        """Chuyển JSON state thành vector cho DQN"""
        try:
            # 1. Service Type Feature
            service_vector = self._encode_service_type(data)
            
            # 2. QoS Features
            qos_vector = self._extract_qos_features(data)
            
            # 3. Source Node Features
            source_vector = self._extract_source_features(data)
            
            # 4. Neighbor Link Features
            link_vector = self._extract_link_features(data)
            
            # Kết hợp tất cả thành state vector
            state_vector = np.concatenate([service_vector, qos_vector, source_vector, link_vector])
            
            # Đảm bảo state vector có đúng kích thước
            if len(state_vector) > self.STATE_SIZE:
                state_vector = state_vector[:self.STATE_SIZE]
            elif len(state_vector) < self.STATE_SIZE:
                # Padding nếu cần
                padding = np.zeros(self.STATE_SIZE - len(state_vector))
                state_vector = np.concatenate([state_vector, padding])
            
            return state_vector.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error in state processing: {e}")
            # Return zero vector as fallback
            return np.zeros(self.STATE_SIZE, dtype=np.float32)
    
    def _encode_service_type(self, data: Dict) -> np.ndarray:
        """Mã hóa service type thành one-hot vector"""
        service_type = data.get('targetQoS', {}).get('serviceType', 'VIDEO_STREAMING')
        
        service_types = ['VIDEO_STREAMING', 'VOICE_CALL', 'MESSAGING', 'FILE_TRANSFER']
        encoding = [1.0 if service_type == st else 0.0 for st in service_types]
        
        return np.array(encoding, dtype=np.float32)
    
    def _extract_qos_features(self, data: Dict) -> np.ndarray:
        """Trích xuất các đặc trưng QoS từ dữ liệu"""
        qos = data.get('targetQoS', {})
        return np.array([
            qos.get('maxLatencyMs', 100.0) / 100.0,           # Normalize
            qos.get('minBandwidthMbps', 500.0) / 1000.0,      # Normalize  
            qos.get('maxLossRate', 0.02) * 50.0               # Scale
        ], dtype=np.float32)
        
    def _extract_source_features(self, data: Dict) -> np.ndarray:
        """Trích xuất source node features"""
        src_info = data.get('sourceNodeInfo', {})
        if not src_info:
            return np.zeros(self.SOURCE_FEATURES, dtype=np.float32)
        
        # Buffer load ratio
        current_packets = src_info.get('currentPacketCount', 0)
        buffer_capacity = src_info.get('packetBufferCapacity', 1)
        buffer_ratio = current_packets / buffer_capacity if buffer_capacity > 0 else 0.0
        
        # Node type encoding
        node_type = src_info.get('nodeType', 'GROUND_STATION')
        is_ground = 1.0 if node_type == 'GROUND_STATION' else 0.0
        is_sea = 1.0 if node_type == 'SEA_STATION' else 0.0
        is_satellite = 1.0 if node_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] else 0.0
        
        return np.array([
            src_info.get('resourceUtilization', 0.0),          # 0-1
            src_info.get('batteryChargePercent', 100.0) / 100.0, # 0-1
            buffer_ratio,                                      # 0-1
            src_info.get('nodeProcessingDelayMs', 5.0) / 10.0, # Normalize
            src_info.get('packetLossRate', 0.0) * 100.0,       # Scale
            is_ground,
            is_sea,
            is_satellite,
            self._encode_weather(src_info.get('weather', 'CLEAR'))
        ], dtype=np.float32)
    
    def _extract_link_features(self, data: Dict) -> np.ndarray:
        """Trích xuất neighbor link features"""
        neighbors_links = data.get('neighborLinkMetrics', {})
        if not neighbors_links:
            # Return zeros nếu không có neighbors
            return np.zeros(self.MAX_NEIGHBORS * self.LINK_FEATURES_PER_NEIGHBOR, dtype=np.float32)
        
        link_features = []
        
        # Sắp xếp neighbors để đảm bảo tính nhất quán
        sorted_neighbors = sorted(neighbors_links.keys())
        
        for i, neighbor_id in enumerate(sorted_neighbors):
            if i >= self.MAX_NEIGHBORS:
                break
            
            link = neighbors_links[neighbor_id]
            link_features.extend([
                link.get('latencyMs', 1000.0) / 100.0,                    # Normalize
                link.get('currentAvailableBandwidthMbps', 0.0) / 1000.0, # Normalize
                link.get('packetLossRate', 1.0),                          # 0-1
                link.get('linkAttenuationDb', 0.0) / 50.0,               # Normalize
                link.get('linkScore', 0.0) / 100.0,                      # Normalize
                1.0 if link.get('isLinkActive', True) else 0.0           # Binary
            ])
        
        # Padding nếu cần
        expected_size = self.MAX_NEIGHBORS * self.LINK_FEATURES_PER_NEIGHBOR
        padding_needed = expected_size - len(link_features)
        if padding_needed > 0:
            link_features.extend([0.0] * padding_needed)
        
        return np.array(link_features, dtype=np.float32)

    def _encode_weather(self, weather: str) -> float:
        """Mã hóa thời tiết"""
        encoding = {
            'CLEAR': 0.0,
            'LIGHT_RAIN': 0.25,
            'MODERATE_RAIN': 0.5,
            'HEAVY_RAIN': 0.75,
            'SEVERE_STORM': 1.0
        }
        return encoding.get(weather, 0.0)

    def get_state_size(self) -> int:
        """Trả về kích thước state vector"""
        return self.STATE_SIZE