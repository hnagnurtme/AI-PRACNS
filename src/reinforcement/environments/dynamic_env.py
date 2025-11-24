import numpy as np
from typing import Dict, Any, List, Optional
from .satellite_env import SatelliteEnv
from simulation.dynamics.mobility import MobilityManager
from simulation.dynamics.weather import WeatherModel
from simulation.dynamics.traffic import TrafficModel
from simulation.dynamics.failures import FailureModel
from simulation.core.packet import Packet
import random
from utils.constants import MAX_PROCESSING_DELAY_MS

# Congestion and load balancing thresholds
HIGH_CONGESTION_THRESHOLD = 0.8
MODERATE_CONGESTION_THRESHOLD = 0.6
MODERATE_CONGESTION_MULTIPLIER = 0.5
SEVERE_IMBALANCE_THRESHOLD = 0.9

class DynamicSatelliteEnv(SatelliteEnv):
    def __init__(self, state_builder, nodes: Dict, weights: Optional[Dict] = None, dynamic_config: Optional[Dict] = None):
        super().__init__(state_builder, weights)
        
        self.dynamic_config = dynamic_config or {}
        
        self.mobility_manager = MobilityManager(self.dynamic_config.get('mobility'))
        self.weather_model = WeatherModel(self.dynamic_config.get('weather') or {})
        self.traffic_model = TrafficModel(self.dynamic_config.get('traffic'))
        self.failure_model = FailureModel(self.dynamic_config.get('failures'))
        
        self.mobility_manager.set_nodes(nodes)
        self.failure_model.set_nodes(nodes)
        
        self.simulation_time = 0.0
        self.time_step = self.dynamic_config.get('time_step', 1.0)
        self.current_packet_state: Optional[Packet] = None
        
    def reset(self, initial_packet: Packet) -> np.ndarray:
        self.simulation_time = 0.0
        self.current_packet_state = initial_packet
        
        self.mobility_manager.reset()
        self.weather_model.reset()
        self.traffic_model.reset()
        self.failure_model.reset()
        
        dynamic_state = self.step_dynamics()
        current_node_id = initial_packet.current_holding_node_id
        neighbor_ids = self._get_dynamic_sorted_neighbors(current_node_id, dynamic_state)
        return self.state_builder.build(initial_packet, dynamic_neighbors=neighbor_ids)
    
    def step_dynamics(self):
        """
        Update all dynamic factors and ensure node neighbors are updated for baseline algorithms.
        """
        self.simulation_time += self.time_step
        
        weather_impact = self.weather_model.update(self.time_step)
        traffic_load = self.traffic_model.update(self.simulation_time)
        
        self.mobility_manager.update_nodes(self.time_step)
        self.failure_model.update_failures()
        
        dynamic_state = {
            'weather_impact': weather_impact,
            'traffic_load': traffic_load,
            'simulation_time': self.simulation_time
        }
        
        # CRITICAL FIX: Update node neighbors so baseline algorithms have current neighbors
        self.mobility_manager.update_node_neighbors(dynamic_state)
        
        return dynamic_state
    
    def _simulate_hop(self, packet: Packet, next_node_id: Optional[str], dropped: bool = False) -> Packet:
        new_p = packet
        if next_node_id:
            new_p.current_holding_node_id = next_node_id
        
        new_p.ttl -= 1
        
        step_delay = random.uniform(1.0, MAX_PROCESSING_DELAY_MS/10.0) + random.uniform(5.0, 50.0)
        new_p.accumulated_delay_ms += step_delay

        if dropped:
            new_p.dropped = True
        
        return new_p

    def simulate_episode(self, agent, initial_packet: Packet, max_hops: int = 10, is_training: bool = True) -> Dict[str, Any]:
        state = self.reset(initial_packet)
        total_reward = 0.0
        hop = 0
        transitions = []
        visited_nodes = set()
        current_node_id = initial_packet.current_holding_node_id
        visited_nodes.add(current_node_id)

        while hop < max_hops:
            dynamic_state = self.step_dynamics()
            
            neighbor_ids = self._get_dynamic_sorted_neighbors(current_node_id, dynamic_state)
            state = self.state_builder.build(self.current_packet_state, dynamic_neighbors=neighbor_ids)
            valid_neighbor_count = min(len(neighbor_ids), self.state_builder.MAX_NEIGHBORS)

            if valid_neighbor_count == 0:
                reward = -self.weights['drop']
                next_packet = self._simulate_hop(self.current_packet_state, None, dropped=True)
                next_state = self.state_builder.build(next_packet, dynamic_neighbors=[])
                if is_training:
                    transitions.append((state, None, reward, next_state, True))
                total_reward += reward
                break 

            action_index = agent.select_action(state, num_valid_actions=valid_neighbor_count, is_training=is_training)

            if not (0 <= action_index < valid_neighbor_count):
                reward = -self.weights['drop']
                if is_training:
                    transitions.append((state, action_index, reward, state, True))
                total_reward += reward
                break

            selected_neighbor_id = neighbor_ids[action_index]
            
            new_packet_data = self._simulate_hop(self.current_packet_state, selected_neighbor_id)
            next_node_id = new_packet_data.current_holding_node_id
            next_neighbor_ids = self._get_dynamic_sorted_neighbors(next_node_id, dynamic_state)
            next_state = self.state_builder.build(new_packet_data, dynamic_neighbors=next_neighbor_ids)
            
            if selected_neighbor_id in visited_nodes:
                reward = -self.weights['hop_cost'] * 3.0
                done = False
            else:
                visited_nodes.add(selected_neighbor_id)
                done = self._is_terminal(new_packet_data)
                reward = self._calculate_dynamic_reward(state, action_index, new_packet_data, dynamic_state)

            if is_training:
                transitions.append((state, action_index, reward, next_state, done))
            total_reward += reward

            self.current_packet_state = new_packet_data
            current_node_id = self.current_packet_state.current_holding_node_id
            hop += 1

            if done:
                break

        if is_training:
            for s, a, r, s_next, d in transitions:
                if a is not None:
                    agent.memory.push(s, a, r, s_next, d)
            agent.optimize_model()

        # Return episode metrics dictionary
        delivered = self.current_packet_state.current_holding_node_id == initial_packet.station_dest
        latency = self.current_packet_state.accumulated_delay_ms

        return {
            'total_reward': total_reward,
            'delivered': delivered,
            'latency': latency,
            'hops': hop
        }
    
    def _get_dynamic_sorted_neighbors(self, node_id: str, dynamic_state: Dict) -> List[str]:
        # Giả sử mobility_manager có phương thức này
        return self.mobility_manager.get_current_neighbors(node_id, dynamic_state)
    
    def get_operational_nodes(self) -> List[Dict[str, Any]]:
        """Get all operational nodes from the network"""
        operational_nodes = []
        for node_id, node in self.mobility_manager.nodes.items():
            if node.isOperational and not self.failure_model.is_node_failed(node_id):
                operational_nodes.append({
                    'nodeId': node_id,
                    'position': node.position,
                    'node_type': node.nodeType
                })
        return operational_nodes
    
    def get_network_metrics(self) -> Dict[str, float]:
        """
        Calculate network-wide metrics for analysis.
        Useful for comparing RL vs baseline performance.
        
        Returns:
            Dictionary with metrics like average congestion, packet loss, etc.
        """
        total_utilization = 0.0
        total_queue_occupancy = 0.0
        total_packet_loss = 0.0
        operational_count = 0
        
        for node_id, node in self.mobility_manager.nodes.items():
            if node.isOperational and not self.failure_model.is_node_failed(node_id):
                total_utilization += node.resourceUtilization
                total_queue_occupancy += node.currentPacketCount / max(node.packetBufferCapacity, 1)
                total_packet_loss += node.packetLossRate
                operational_count += 1
        
        if operational_count == 0:
            return {
                'avg_utilization': 0.0,
                'avg_queue_occupancy': 0.0,
                'avg_packet_loss': 0.0,
                'operational_nodes': 0
            }
        
        return {
            'avg_utilization': total_utilization / operational_count,
            'avg_queue_occupancy': total_queue_occupancy / operational_count,
            'avg_packet_loss': total_packet_loss / operational_count,
            'operational_nodes': operational_count,
            'max_utilization': max((node.resourceUtilization 
                                   for node_id, node in self.mobility_manager.nodes.items() 
                                   if node.isOperational), default=0.0),
            'utilization_variance': np.var([node.resourceUtilization 
                                           for node_id, node in self.mobility_manager.nodes.items() 
                                           if node.isOperational]) if operational_count > 0 else 0.0
        }

    def _is_terminal(self, packet: Packet) -> bool:
        is_dest = packet.current_holding_node_id == packet.station_dest
        is_dead = packet.dropped or packet.ttl <= 0
        return is_dest or is_dead

    def _calculate_reward(self, prev_state: np.ndarray, action_idx: int, new_packet: Packet) -> float:
        w = self.weights

        if new_packet.current_holding_node_id == new_packet.station_dest:
            return w['goal']
        if new_packet.dropped or new_packet.ttl <= 0:
            return -w['drop']

        start_idx = self.START_INDEX_NEIGHBORS + (action_idx * self.NEIGHBOR_FEAT_SIZE)
        
        if start_idx + self.NEIGHBOR_FEAT_SIZE > len(prev_state):
            return -w['drop'] 

        feats = prev_state[start_idx : start_idx + self.NEIGHBOR_FEAT_SIZE]
        
        dist_score = feats[1]
        snr_score = feats[3]
        queue_score = feats[5]
        cpu_score = feats[6]
        loss_score = feats[7]
        delay_score = feats[8]
        is_op = feats[9]

        reward = 0.0

        reward += w['hop_cost']
        reward += w['progress_penalty'] * dist_score 

        reward += w['snr_reward'] * snr_score

        reward += w['node_load'] * queue_score 
        reward += w['utilization'] * (1.0 - cpu_score)

        reward += w['reliability'] * (1.0 - loss_score)
        reward += w['operational'] * is_op

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

    def _calculate_dynamic_reward(self, prev_state: np.ndarray, action_idx: int,
                                new_packet: Packet, dynamic_state: Dict) -> float:
        """
        Enhanced reward function that considers:
        1. Basic routing metrics (base_reward)
        2. Dynamic environmental factors (weather, traffic)
        3. Resource balancing and congestion avoidance
        4. Proactive optimization (avoiding overloaded nodes)
        """
        base_reward = self._calculate_reward(prev_state, action_idx, new_packet)

        # Dynamic environmental penalties
        dynamic_penalty = 0.0
        
        if dynamic_state['weather_impact'] > 0.7:
            dynamic_penalty -= self.weights.get('weather_penalty', 50.0)

        if dynamic_state['traffic_load'] > 2.0:
            dynamic_penalty -= self.weights.get('traffic_penalty', 20.0)
        
        # Extract neighbor features for resource balancing analysis
        start_idx = self.START_INDEX_NEIGHBORS + (action_idx * self.NEIGHBOR_FEAT_SIZE)
        
        if start_idx + self.NEIGHBOR_FEAT_SIZE <= len(prev_state):
            feats = prev_state[start_idx : start_idx + self.NEIGHBOR_FEAT_SIZE]
            
            # Resource utilization features
            queue_score = feats[5]  # Queue occupancy (0-1, higher = more congested)
            cpu_score = feats[6]    # CPU utilization (0-1, higher = more utilized)
            
            # Reward for selecting less congested nodes (load balancing)
            # This helps RL learn to avoid overloaded nodes proactively
            congestion_level = (queue_score + cpu_score) / 2.0
            
            # Strong reward for avoiding congested nodes
            if congestion_level > HIGH_CONGESTION_THRESHOLD:
                # Heavily penalize selecting highly congested nodes
                dynamic_penalty -= self.weights.get('congestion_penalty', 100.0)
            elif congestion_level > MODERATE_CONGESTION_THRESHOLD:
                # Moderate penalty for moderately congested nodes
                dynamic_penalty -= self.weights.get('congestion_penalty', 100.0) * MODERATE_CONGESTION_MULTIPLIER
            else:
                # Reward for selecting underutilized nodes (load balancing)
                dynamic_penalty += self.weights.get('load_balance_reward', 20.0) * (1.0 - congestion_level)
            
            # Reward fairness - penalize extreme resource imbalances
            # This encourages spreading load across the network
            resource_variance_penalty = 0.0
            if cpu_score > SEVERE_IMBALANCE_THRESHOLD or queue_score > SEVERE_IMBALANCE_THRESHOLD:
                # Severe imbalance detected
                resource_variance_penalty = -self.weights.get('resource_imbalance_penalty', 75.0)
            
            dynamic_penalty += resource_variance_penalty

        return base_reward + dynamic_penalty