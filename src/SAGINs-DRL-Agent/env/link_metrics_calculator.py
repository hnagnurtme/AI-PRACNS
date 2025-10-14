# env/link_metrics_calculator.py
import math
import time
import random
from typing import Dict, List

class LinkMetricsCalculator:
    """Tính toán link metrics real-time từ thông tin nodes"""
    
    def calculate_all_possible_links(self, node: Dict[str, Dict]) -> List[Dict]:
        """Tính toán tất cả possible links giữa các nodes"""
        links = []
        node_ids = list(node.keys())
        
        # Tạo links giữa tất cả các node pairs 
        for i, source_id in enumerate(node_ids):
            for j, dest_id in enumerate(node_ids):
                if i != j:
                    source_node = node[source_id]
                    dest_node = node[dest_id]
                    
                    link_metrics = self.calculate_link_metrics(source_node, dest_node)
                    links.append(link_metrics)
        
        return links

    def calculate_link_metrics(self, source_node: Dict, dest_node: Dict) -> Dict:
        """Tính toán metrics cho link giữa source và destination"""
        
        # 1. Tính khoảng cách 
        distance_km = self._calculate_distance(source_node, dest_node)
        
        # 2. Base metrics từ distance và node types
        base_latency = self._calculate_bse_latency(distance_km, source_node, dest_node)
        max_bandwidth = self._get_max_bandwidth(source_node, dest_node)
        
        # 3. Môi trường và trạng thái node ảnh hưởng
        weather_factor = self._get_weather_factor(source_node, dest_node)
        mobility_factor = self._get_mobility_factor(source_node, dest_node)
        
        # 4. Available bandwidth (có xét đến resource utilization)
        available_bandwidth = self._calculate_available_bandwidth(max_bandwidth, source_node, dest_node)
        
        # 5. Packet loss rate
        packet_loss_rate = self._calculate_packet_loss_rate(distance_km, weather_factor, source_node, dest_node)
        
        # 6. Xác định link có active hay không
        is_link_active = self._is_link_available(source_node, dest_node, distance_km)
        
        link_metrics = {
            "sourceNodeId": source_node['nodeId'],
            "destinationNodeId": dest_node['nodeId'],
            "distanceKm": distance_km,
            "maxBandwidthMbps": max_bandwidth,
            "currentAvailableBandwidthMbps": available_bandwidth,
            "latencyMs": base_latency * weather_factor * mobility_factor,
            "packetLossRate": packet_loss_rate,
            "linkAttenuationDb": self._calculate_attenuation_db(distance_km, weather_factor),
            "isLinkActive": is_link_active,
            "lastUpdated": int(time.time() * 1000)
        }
        
        # Tính link score
        link_metrics["linkScore"] = self._calculate_link_score(link_metrics)
        
        return link_metrics

    def _is_link_available(self, source_node: Dict, dest_node: Dict, distance_km: float) -> bool:
        """Xác định link có available dựa trên trạng thái node và khoảng cách"""
        
        # 1. Cả hai node phải operational
        if not source_node.get('isOperational', True) or not dest_node.get('isOperational', True):
            return False
        
        # 2. Cả hai node phải healthy 
        if not source_node.get('healthy', True) or not dest_node.get('healthy', True):
            return False

        # 3. Khoảng cách phải trong phạm vi tối đa
        max_distance = self._get_max_link_distance(source_node, dest_node)
        if distance_km > max_distance:
            return False

        # # 4. 
        # if not self._is_connection_allowed(source_node, dest_node):
        #     return False
        
        return True

    def _get_max_link_distance(self, source_node: Dict, dest_node: Dict) -> float:
        """Xác định khoảng cách tối đa cho phép link"""
        source_type = source_node.get('nodeType')
        dest_type = dest_node.get('nodeType')
        
        # Ground/Sea Station to Ground/Sea Station
        if (source_type in ['GROUND_STATION', 'SEA_STATION'] and 
            dest_type in ['GROUND_STATION', 'SEA_STATION']):
            return 50.0  # km (terrestrial/maritime links)
        
        # Ground/Sea Station to Satellite
        elif (source_type in ['GROUND_STATION', 'SEA_STATION'] and 
              dest_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']):
            return 2000.0  # km
        
        # Satellite to Ground/Sea Station  
        elif (source_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] and 
              dest_type in ['GROUND_STATION', 'SEA_STATION']):
            return 2000.0  # km
        
        # Satellite to Satellite
        elif (source_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] and 
              dest_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']):
            return 5000.0  # km
        
        return 1000.0  # km (default)
    
    # def _is_connection_allowed(self, source_node: Dict, dest_node: Dict) -> bool:
    #     """Kiểm tra xem hai node có thể kết nối trực tiếp với nhau hay không"""
    
    def _calculate_distance(self, source_node: Dict, dest_node: Dict) -> float:
        """Tính khoảng cách giữa hai nodes (km)"""
        pos_src = source_node.get('position', {})
        pos_dst = dest_node.get('position', {})
        
        lat1, lon1, alt1 = pos_src.get('latitude', 0.0), pos_src.get('longitude', 0.0), pos_src.get('altitude', 0.0)
        lat2, lon2, alt2 = pos_dst.get('latitude', 0.0), pos_dst.get('longitude', 0.0), pos_dst.get('altitude', 0.0)
        
        # Harvesine formula for ground stations
        if source_node.get('nodeType') == 'GROUND_STATION' and dest_node.get('nodeType') == 'GROUND_STATION':
            return self._haversine_distance(lat1, lon1, lat2, lon2)
        else:
            # Tính khoảng cách Euclidean 3D cho các trường hợp khác
            return math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2 + (alt2-alt1)**2)
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Tính khoảng cách Haversine giữa hai điểm trên bề mặt Trái Đất (km)"""
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _calculate_bse_latency(self, distance_km: float, source_node: Dict, dest_node: Dict) -> float:
        """Tính base latency từ khoảng cách và node types"""
        # Speed of light latency (ms)
        light_speed_latency = (distance_km / 300000) * 1000  # 300000 km/s
        
        # Processing delays
        processing_delay = (source_node.get('nodeProcessingDelayMs', 1.0) + dest_node.get('nodeProcessingDelayMs', 1.0))
        
        # Properties delay dựa trên node types
        propagation_delay = 0
        if (source_node.get('nodeType') in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] or 
            dest_node.get('nodeType') in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']):
            propagation_delay = 2.0  # ms for satellite links
            
        return light_speed_latency + processing_delay + propagation_delay
    
    def _get_max_bandwidth(self, source_node: Dict, dest_node: Dict) -> float:
        """Xác định băng thông tối đa dựa trên loại node"""
        source_type = source_node.get('nodeType')
        dest_type = source_node.get('nodeType')
        
        # Ground/Sea to Ground/Sea
        if (source_type in ['GROUND_STATION', 'SEA_STATION'] and 
            dest_type in ['GROUND_STATION', 'SEA_STATION']):
            return random.uniform(800.0, 1200.0)  # Mbps
        
        # Ground/Sea to Satellite
        elif ((source_type in ['GROUND_STATION', 'SEA_STATION'] and 
               dest_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']) or
              (source_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] and 
               dest_type in ['GROUND_STATION', 'SEA_STATION'])):
            return random.uniform(400.0, 800.0)  # Mbps
        
        # Satellite to Satellite
        elif (source_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] and 
              dest_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']):
            return random.uniform(200.0, 500.0)  # Mbps
        
        return 100.0  # Mbps (default)
    
    def _get_weather_factor(self, source_node: Dict, dest_node: Dict) -> float:
        """Tính toán hệ số ảnh hưởng của thời tiết"""
        weather_penalties = {
            'CLEAR': 1.0,
            'LIGHT_RAIN': 1.05,
            'MODERATE_RAIN': 1.15,
            'HEAVY_RAIN': 1.3,
            'SEVERE_STORM': 1.5
        }
        
        # Weather ảnh hưởng Ground Station và Sea Station
        weather1 = source_node.get('weather', 'CLEAR') if source_node.get('nodeType') in ['GROUND_STATION', 'SEA_STATION'] else 'CLEAR'
        weather2 = dest_node.get('weather', 'CLEAR') if dest_node.get('nodeType') in ['GROUND_STATION', 'SEA_STATION'] else 'CLEAR'
        
        factor_1 = weather_penalties.get(weather1, 1.0)
        factor_2 = weather_penalties.get(weather2, 1.0)
        
        return max(factor_1, factor_2)
    
    def _get_mobility_factor(self, source_node: Dict, dest_node: Dict) -> float:
        """Tính mobility impact factor"""
        # Ground stations 
        if (source_node.get('nodeType') == 'GROUND_STATION' and dest_node.get('nodeType') == 'GROUND_STATION'):
            return 1.0
        
        # Satellites mobility 
        velocity_1 = source_node.get('velocity', {})
        velocity_2 = dest_node.get('velocity', {})
        
        speed_1 = velocity_1.get('speed', 0.0) if source_node.get('nodeType') in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] else 0.0
        speed_2 = velocity_2.get('speed', 0.0) if dest_node.get('nodeType') in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] else 0.0
        
        avg_speed = (speed_1 + speed_2) / 2.0
        
        return 1.0 + (avg_speed / 50.0) # Giả sử mỗi 50 km/s tăng thêm 1% latency
    
    def _calculate_available_bandwidth(self, max_bandwidth: float, source_node: Dict, dest_node: Dict) -> float:
        """Tính bandwidth khả dụng thực tế"""
        source_util = source_node.get('resourceUtilization', 0.0)
        dest_util = dest_node.get('resourceUtilization', 0.0)
        
        # Utilization penalty
        avg_util = (source_util + dest_util) / 2.0
        utilization_penalty = avg_util * 0.5 # Giảm tối đa 50%
        
        available_bandwidth = max_bandwidth * (1.0 - utilization_penalty)
        
        return max(available_bandwidth, max_bandwidth * 0.1)  # Đảm bảo ít nhất 10% của max bandwidth
    
    def _calculate_packet_loss_rate(self, distance_km: float, weather_factor: float, source_node: Dict, dest_node: Dict) -> float:
        """Tính packet loss rate tổng hợp"""
        # base loss từ khoảng cách
        distance_loss = min(0.1, distance_km / 10000)
        
        # weather impact
        weather_loss = (weather_factor - 1.0) * 0.02
        
        # node reliability
        source_reliability = source_node.get('packetLossRate', 0.0)
        dest_node_reliability = dest_node.get('packetLossRate', 0.0)
        node_loss = (source_reliability + dest_node_reliability) / 2.0
        
        # connection type loss - BỔ SUNG SEA STATION
        connection_loss = 0.0
        source_type = source_node.get('nodeType')
        dest_type = dest_node.get('nodeType')
        
        # Ground/Sea to Ground/Sea
        if (source_type in ['GROUND_STATION', 'SEA_STATION'] and 
            dest_type in ['GROUND_STATION', 'SEA_STATION']):
            connection_loss = 0.001  # 0.1% loss
        
        # Ground/Sea to Satellite
        elif ((source_type in ['GROUND_STATION', 'SEA_STATION'] and 
               dest_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']) or
              (source_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] and 
               dest_type in ['GROUND_STATION', 'SEA_STATION'])):
            connection_loss = 0.005  # 0.5% loss
        
        # Satellite to Satellite
        elif (source_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'] and 
              dest_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']):
            connection_loss = 0.03  # 3% loss
        
        total_loss = distance_loss + weather_loss + node_loss + connection_loss
        return min(total_loss, 0.3)  # Giới hạn tối đa 30% loss
    
    def _calculate_attenuation_db(self, distance_km: float, weather_factor: float) -> float:
        """Tính toán attenuation (dB) dựa trên khoảng cách và thời tiết"""
        # Free space path loss 
        fsl = 20 * math.log10(distance_km) + 20 * math.log10(2000) + 32.44 # 2GHz frequency
        weather_factor = (weather_factor - 1.0) * 10.0  # Chuyển hệ số thời tiết thành dB

        return fsl + weather_factor
    
    def _calculate_link_score(self, link_metrics: Dict) -> float:
        """Tính toán link score tổng hợp từ các metrics"""
        if not link_metrics.get('isLinkActive', True) or link_metrics.get('currentAvailableBandwidthMbps', 0.0) < 1.0:
            return 0.0
        
        latency_cost = 1.0 + math.log(1.0 + link_metrics.get('latencyMs', 1.0))
        loss_factor = 1.0 - link_metrics.get('packetLossRate', 0.0)
        attenuation_factor = 1.0 / (1.0 + 0.05 * link_metrics.get('linkAttenuationDb', 0.0))

        score = (link_metrics['currentAvailableBandwidthMbps'] / latency_cost) * loss_factor * attenuation_factor 

        return max(0.001, score)
