import numpy as np
import random
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from utils.state_builder import StateBuilder
from utils.constants import (
    MAX_NEIGHBORS, 
    NEIGHBOR_FEAT_SIZE,
    MAX_PROCESSING_DELAY_MS,
    MAX_PROPAGATION_DELAY_MS
)
from simulation.core.packet import Packet, HopRecord, RoutingAlgorithm, RoutingDecisionInfo
from simulation.core.network import SAGINNetwork
from simulation.dynamics.mobility import MobilityManager
from simulation.dynamics.weather import WeatherModel
from simulation.dynamics.traffic import TrafficModel
MAX_SYSTEM_LATENCY_MS = 5000.0

DEFAULT_WEIGHTS = {
    'goal': 200.0,
    'drop': 300.0,
    'hop_cost': -150.0,
    'progress_penalty': -250.0,
    'latency': -25.0,
    'latency_violation': -200.0,
    'snr_reward': 5.0,
    'utilization': 8.0,
    'reliability': 5.0,
    'operational': 10.0,
    'node_load': -20.0,
    'resource_balance': 5.0,
    'weather_penalty': -50.0,
    'traffic_bonus': 20.0,
}

class SatelliteEnv:
    """
    Môi trường SAGIN với tính động - ĐÃ SỬA
    """
    
    def __init__(self, state_builder: StateBuilder, network: SAGINNetwork, 
                 weights: Optional[Dict[str, float]] = None):
        self.state_builder = state_builder
        self.network = network
        self.weights = DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        # Khởi tạo các mô hình động
        self.mobility_manager = MobilityManager()
        self.weather_model = WeatherModel()
        self.traffic_model = TrafficModel()
        
        self.current_packet: Optional[Packet] = None
        self.simulation_time = 0.0
        self.time_step = 1.0  # 1 giây mỗi bước
        
        self.START_INDEX_NEIGHBORS = 14 + 8
        self.NEIGHBOR_FEAT_SIZE = NEIGHBOR_FEAT_SIZE

    def reset(self, packet: Packet) -> np.ndarray:
        """Reset environment với packet mới - ĐÃ SỬA"""
        self.current_packet = packet
        self.simulation_time = 0.0
        
        # Reset các mô hình động
        self.mobility_manager.reset()
        self.weather_model.reset()
        self.traffic_model.reset()
        
        return self.state_builder.build(packet)

    def step_dynamics(self):
        """Cập nhật động lực học mạng trước mỗi bước"""
        self.simulation_time += self.time_step
        
        # Cập nhật tất cả mô hình động
        weather_impact = self.weather_model.update(self.time_step)
        traffic_load = self.traffic_model.update(self.simulation_time)
        self.mobility_manager.update_nodes(self.network, self.time_step, self.simulation_time)
        
        return {
            'weather_impact': weather_impact,
            'traffic_load': traffic_load,
            'simulation_time': self.simulation_time
        }

    def simulate_episode(self, agent, packet: Packet, max_hops: int = 15) -> Dict[str, Any]:
        """Chạy episode với tính động - ĐÃ SỬA"""
        state = self.reset(packet)
        total_reward = 0.0
        hop = 0
        transitions = []
        visited_nodes = set()
        current_node_id = packet.current_holding_node_id
        visited_nodes.add(current_node_id)

        episode_metrics = {
            'start_time': datetime.now(),
            'packet_id': packet.packet_id,
            'source': packet.station_source,
            'destination': packet.station_dest
        }

        while hop < max_hops:
            # QUAN TRỌNG: Cập nhật động lực học trước mỗi bước
            dynamic_state = self.step_dynamics()
            
            current_node_id = self.current_packet.current_holding_node_id
            current_node = self.network.get_node(current_node_id)
            
            # Kiểm tra node hiện tại có tồn tại và hoạt động
            if not current_node or not current_node.isOperational:
                reward = -self.weights['drop']
                next_packet = self._simulate_hop(self.current_packet, None, dropped=True)
                next_state = self.state_builder.build(next_packet)
                transitions.append((state, None, reward, next_state, True))
                total_reward += reward
                break

            # Lấy danh sách neighbor ĐÃ SORT theo StateBuilder logic
            neighbor_ids = self._get_sorted_neighbors(current_node_id, dynamic_state)
            valid_neighbor_count = min(len(neighbor_ids), MAX_NEIGHBORS)

            if valid_neighbor_count == 0:
                reward = -self.weights['drop']
                next_packet = self._simulate_hop(self.current_packet, None, dropped=True)
                next_state = self.state_builder.build(next_packet)
                transitions.append((state, None, reward, next_state, True))
                total_reward += reward
                break

            # Agent chọn action với masking
            action_index = agent.select_action(state, num_valid_actions=valid_neighbor_count)

            if not (0 <= action_index < valid_neighbor_count):
                reward = -self.weights['drop']
                transitions.append((state, action_index, reward, state, True))
                total_reward += reward
                break

            # Lấy neighbor ID ĐÚNG THỨ TỰ
            selected_neighbor_id = neighbor_ids[action_index]
            selected_neighbor = self.network.get_node(selected_neighbor_id)
            
            # Kiểm tra neighbor có hợp lệ
            if not selected_neighbor or not selected_neighbor.isOperational:
                reward = -self.weights['drop']
                next_packet = self._simulate_hop(self.current_packet, None, dropped=True)
                next_state = self.state_builder.build(next_packet)
                transitions.append((state, action_index, reward, next_state, True))
                total_reward += reward
                break

            # Loop detection
            if selected_neighbor_id in visited_nodes:
                reward = -self.weights['hop_cost'] * 3.0
                new_packet = self._simulate_hop(self.current_packet, selected_neighbor_id)
                next_state = self.state_builder.build(new_packet)
                done = False
            else:
                visited_nodes.add(selected_neighbor_id)
                new_packet = self._simulate_hop(self.current_packet, selected_neighbor_id)
                next_state = self.state_builder.build(new_packet)
                done = self._is_terminal(new_packet)
                reward = self._calculate_dynamic_reward(state, action_index, new_packet, dynamic_state)

            transitions.append((state, action_index, reward, next_state, done))
            total_reward += reward

            state = next_state
            self.current_packet = new_packet
            hop += 1

            if done:
                break

        # Experience replay
        for s, a, r, s_next, d in transitions:
            if a is not None:
                agent.memory.push(s, a, r, s_next, d)

        agent.optimize_model()

        # Return comprehensive episode metrics
        return self._compile_episode_metrics(episode_metrics, total_reward, hop)

    def _get_sorted_neighbors(self, node_id: str, dynamic_state: Dict) -> List[str]:
        """Lấy danh sách neighbor ĐÃ SORT theo StateBuilder logic"""
        node = self.network.get_node(node_id)
        if not node:
            return []
        
        # Sử dụng mobility manager để lấy neighbor có tính đến vị trí hiện tại
        return self.mobility_manager.get_current_neighbors(node_id, self.network, dynamic_state)

    def _is_terminal(self, packet: Packet) -> bool:
        """Kiểm tra điều kiện kết thúc"""
        current_node = self.network.get_node(packet.current_holding_node_id)
        if not current_node or not current_node.isOperational:
            return True
            
        is_dest = packet.current_holding_node_id == packet.station_dest
        is_dead = packet.dropped or packet.ttl <= 0
        return is_dest or is_dead

    def _simulate_hop(self, packet: Packet, next_node_id: Optional[str], dropped: bool = False) -> Packet:
        """Mô phỏng bước nhảy packet - ĐÃ SỬA"""
        # Tạo bản sao của packet để tránh thay đổi trạng thái gốc
        new_packet = Packet(
            packet_id=packet.packet_id,
            source_user_id=packet.source_user_id,
            destination_user_id=packet.destination_user_id,
            station_source=packet.station_source,
            station_dest=packet.station_dest,
            type=packet.type,
            time_sent_from_source_ms=packet.time_sent_from_source_ms,
            payload_data_base64=packet.payload_data_base64,
            payload_size_byte=packet.payload_size_byte,
            service_qos=packet.service_qos,
            current_holding_node_id=next_node_id if next_node_id else packet.current_holding_node_id,
            next_hop_node_id="",
            priority_level=packet.priority_level,
            max_acceptable_latency_ms=packet.max_acceptable_latency_ms,
            max_acceptable_loss_rate=packet.max_acceptable_loss_rate,
            analysis_data=packet.analysis_data,
            use_rl=packet.use_rl,
            ttl=packet.ttl - 1,
            acknowledged_packet_id=packet.acknowledged_packet_id,
            path_history=packet.path_history + [packet.current_holding_node_id],
            hop_records=packet.hop_records.copy(),
            accumulated_delay_ms=packet.accumulated_delay_ms,
            dropped=dropped,
            drop_reason=packet.drop_reason
        )
        
        # Tính toán delay thực tế dựa trên loại node và khoảng cách
        if next_node_id and not dropped:
            current_node = self.network.get_node(packet.current_holding_node_id)
            next_node = self.network.get_node(next_node_id)
            
            if current_node and next_node:
                # Tính delay dựa trên loại liên kết và khoảng cách
                hop_delay = self._calculate_hop_delay(current_node, next_node)
                new_packet.accumulated_delay_ms += hop_delay
                
                # Thêm hop record
                hop_record = HopRecord(
                    from_node_id=packet.current_holding_node_id,
                    to_node_id=next_node_id,
                    latency_ms=hop_delay,
                    timestamp_ms=self.simulation_time * 1000,
                    distance_km=self._calculate_distance(current_node, next_node),
                    packet_loss_rate=0.01,  # Có thể tính dựa trên link quality
                    from_node_position=current_node.position,
                    to_node_position=next_node.position,
                    routing_decision_info=RoutingDecisionInfo(
                        algorithm=RoutingAlgorithm.REINFORCEMENT_LEARNING
                    )
                )
                new_packet.hop_records.append(hop_record)
        
        return new_packet

    def _calculate_hop_delay(self, from_node, to_node) -> float:
        """Tính delay cho bước nhảy dựa trên loại node và khoảng cách"""
        # Processing delay
        processing_delay = random.uniform(1.0, MAX_PROCESSING_DELAY_MS / 10.0)
        
        # Propagation delay dựa trên khoảng cách
        distance = self._calculate_distance(from_node, to_node)
        propagation_delay = (distance / 300000.0) * 1000  # km to ms (speed of light)
        
        # Additional delay dựa trên loại liên kết
        link_type_delay = self._get_link_type_delay(from_node.nodeType, to_node.nodeType)
        
        return processing_delay + propagation_delay + link_type_delay

    def _calculate_distance(self, node1, node2) -> float:
        """Tính khoảng cách giữa hai node"""
        pos1 = node1.position.to_cartesian()
        pos2 = node2.position.to_cartesian()
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _get_link_type_delay(self, type1: str, type2: str) -> float:
        """Delay bổ sung dựa trên loại liên kết"""
        delays = {
            ('GROUND_STATION', 'LEO_SATELLITE'): 10.0,
            ('LEO_SATELLITE', 'MEO_SATELLITE'): 15.0,
            ('MEO_SATELLITE', 'GEO_SATELLITE'): 20.0,
            ('GROUND_STATION', 'GROUND_STATION'): 5.0,
        }
        return delays.get((type1, type2), delays.get((type2, type1), 10.0))

    def _calculate_dynamic_reward(self, prev_state: np.ndarray, action_idx: int, 
                                new_packet: Packet, dynamic_state: Dict) -> float:
        """Reward function với yếu tố động - ĐÃ SỬA"""
        w = self.weights

        # Terminal rewards
        if new_packet.current_holding_node_id == new_packet.station_dest:
            return w['goal']
        if new_packet.dropped or new_packet.ttl <= 0:
            return -w['drop']

        # Lấy neighbor features từ state trước đó
        start_idx = self.START_INDEX_NEIGHBORS + (action_idx * self.NEIGHBOR_FEAT_SIZE)
        
        if start_idx + self.NEIGHBOR_FEAT_SIZE > len(prev_state):
            return -w['drop']

        feats = prev_state[start_idx : start_idx + self.NEIGHBOR_FEAT_SIZE]
        
        # Mapping features
        dist_score = feats[1]       # Khoảng cách tới đích
        snr_score = feats[3]        # Chất lượng tín hiệu
        queue_score = feats[5]      # Queue load
        cpu_score = feats[6]        # CPU utilization
        loss_score = feats[7]       # Packet loss rate
        delay_score = feats[8]      # Node processing delay
        is_op = feats[9]            # Operational status

        reward = 0.0

        # Base routing rewards
        reward += w['hop_cost']
        reward += w['progress_penalty'] * dist_score
        reward += w['snr_reward'] * snr_score
        reward += w['node_load'] * queue_score
        reward += w['utilization'] * (1.0 - cpu_score)
        reward += w['reliability'] * (1.0 - loss_score)
        reward += w['operational'] * is_op

        # Dynamic penalties/bonuses
        weather_penalty = dynamic_state['weather_impact'] * w['weather_penalty']
        traffic_bonus = 0.0
        
        # Thưởng cho routing hiệu quả trong điều kiện traffic cao
        if dynamic_state['traffic_load'] > 2.0 and reward > 0:
            traffic_bonus = w['traffic_bonus']
            
        reward += weather_penalty + traffic_bonus

        # QoS violation check
        curr_delay = new_packet.accumulated_delay_ms
        max_lat = new_packet.service_qos.max_latency_ms
        
        if max_lat > 0:
            ratio = curr_delay / max_lat
            if ratio > 1.0:
                reward += w['latency_violation']
            elif ratio > 0.8:
                reward += w['latency_violation'] * 0.5
        
        reward += w['latency'] * delay_score

        return float(reward)

    def get_operational_nodes(self) -> List:
        """Lấy danh sách node đang hoạt động"""
        return self.network.get_operational_nodes()

    def _compile_episode_metrics(self, episode_metrics: Dict, total_reward: float, hops: int) -> Dict[str, Any]:
        """Tổng hợp metrics cho episode"""
        delivered = (self.current_packet.current_holding_node_id == 
                    self.current_packet.station_dest if self.current_packet else False)
        
        return {
            **episode_metrics,
            'total_reward': total_reward,
            'delivered': delivered,
            'latency': self.current_packet.accumulated_delay_ms if self.current_packet else 0.0,
            'hops': hops,
            'end_time': datetime.now(),
            'success': delivered,
            'packet_dropped': self.current_packet.dropped if self.current_packet else True,
            'final_ttl': self.current_packet.ttl if self.current_packet else 0,
            'path_length': len(self.current_packet.path_history) if self.current_packet else 0
        }