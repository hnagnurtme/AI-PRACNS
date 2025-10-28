# simulator/network_simulator.py
import time
import math
from data.mongo_manager import MongoManager
from env.link_metrics_calculator import LinkMetricsCalculator
from env.state_processor import StateProcessor
from agents.dqn_agent import DqnAgent
from agents.replay_buffer import ReplayBuffer  # Assume this exists; if not, add a simple one below
from env.packet import Packet
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, mongo_uri: str):
        self.mongo = MongoManager(mongo_uri)
        self.link_calc = LinkMetricsCalculator(self.mongo)
        self.state_proc = StateProcessor(self.mongo)
        self.agent = DqnAgent(self.state_proc.state_size, 100)  # action_size=100, adjust if needed
        self.buffer = ReplayBuffer()
        self.batch_size = 64
        self.target_update_freq = 100
        self.step_count = 0
        self.last_train_time = time.time()
    
    async def route_packet(self, packet: Packet):
        experiences = []
        while not packet.is_at_dest() and not packet.dropped:
            nodes = {n['nodeId']: n for n in self.mongo.get_all_nodes()}
            current_node = nodes.get(packet.current_holding_node_id)
            
            if not current_node:
                packet.dropped = True
                packet.drop_reason = "Current node not found"
                break
            
            valid_next = self.get_valid_next_hops(current_node, nodes, packet)
            
            if not valid_next:
                packet.dropped = True
                packet.drop_reason = "No valid next hops"
                break
            
            # Get state and select action
            state = self.state_proc.get_state(current_node, packet, [nodes[n] for n in valid_next])
            next_hop_id = self.agent.select_action(state, valid_next)
            next_hop_node = nodes[next_hop_id]
            
            # Calculate link metrics
            metrics = self.link_calc.calculate_link_metrics(current_node, next_hop_node, packet.service_qos)
            
            latency_ms = metrics.get('latencyMs', 0.0)
            bandwidth_mbps = metrics.get('bandwidthMbps', 0.0)
            loss_rate = metrics.get('lossRate', 0.0)
            timestamp_ms = int(time.time() * 1000)
            
            # Update packet with link metrics
            packet.update_hop(
                from_node=packet.current_holding_node_id,
                to_node=next_hop_id,
                latency_ms=latency_ms,
                bandwidth_mbps=bandwidth_mbps,
                loss_rate=loss_rate,
                timestamp_ms=timestamp_ms
            )
            
            # Calculate reward for RL
            reward = self.calculate_reward(metrics, packet, current_node)
            
            # Get next state
            next_valid = self.get_valid_next_hops(next_hop_node, nodes, packet)
            next_state = self.state_proc.get_state(next_hop_node, packet, [nodes[n] for n in next_valid])
            
            # Store experience
            action_index = self.agent.action_mapper.get_action_index(next_hop_id)
            if action_index is not None:
                experiences.append((state, action_index, reward, next_state))
            
            logger.debug(
                f"Hop: {current_node['nodeId']} -> {next_hop_id} | "
                f"Latency: {latency_ms:.2f}ms, BW: {bandwidth_mbps:.2f}Mbps, Loss: {loss_rate:.4f}"
            )
        
        # Train if packet reached destination
        if packet.is_at_dest():
            final_reward = 100.0 - packet.accumulated_delay_ms / 10.0
            if experiences:
                experiences[-1] = (experiences[-1][0], experiences[-1][1], final_reward, experiences[-1][3])
            
            for exp in experiences:
                self.buffer.add(exp)
            
            await self.train_step()
        
        logger.info(
            f"📦 Packet {packet.packet_id} routed: "
            f"path={packet.path_history}, "
            f"latency={packet.accumulated_delay_ms:.2f}ms, "
            f"bandwidth={packet.min_bandwidth_mbps:.2f}Mbps, "
            f"loss_rate={packet.accumulated_loss_rate:.4f}"
        )
        
        return packet
    
    def get_valid_next_hops(self, current_node: Dict, nodes: Dict, packet: Packet) -> List[str]:
        valid_next = []
        for nid, node in nodes.items():
            # Skip if already visited or is current node
            if nid in packet.path_history or nid == packet.current_holding_node_id:
                continue
            
            # Calculate link metrics
            metrics = self.link_calc.calculate_link_metrics(current_node, node, packet.service_qos)
            
            # Check constraints
            los = self.check_line_of_sight(current_node, node)
            elevation_ok = self.check_elevation(current_node, node)
            is_active = metrics.get('isActive', False)
            
            logger.debug(
                f"Checking {current_node['nodeId']} -> {nid}: "
                f"distance={metrics.get('distanceKm', 0):.1f}km, "
                f"LOS={los} (maxRange={current_node['communication'].get('maxRangeKm', 0)}km), "
                f"elevation={elevation_ok}, "
                f"active={is_active}"
            )
            
            if is_active and los and elevation_ok:
                valid_next.append(nid)
        
        logger.info(f"Valid next hops for {current_node['nodeId']} ({current_node['nodeType']}): {valid_next}")
        
        if not valid_next:
            logger.warning(
                f"⚠️ No valid next hops from {current_node['nodeId']} "
                f"(type: {current_node['nodeType']}, pos: {current_node['position']}). "
                f"Available nodes: {list(nodes.keys())}"
            )
        
        return valid_next
    
    def check_line_of_sight(self, from_node: Dict, to_node: Dict) -> bool:
        """Check if distance is within communication range"""
        dist = self.mongo.calculate_distance(from_node['position'], to_node['position'])
        max_range = from_node['communication'].get('maxRangeKm', 2000.0)
        return dist <= max_range
    
    def check_elevation(self, from_node: Dict, to_node: Dict) -> bool:
        """Check if elevation angle meets minimum requirement"""
        # For ground stations communicating with satellites
        if from_node.get('nodeType') == 'GROUND_STATION' and to_node.get('nodeType') in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']:
            from_pos = from_node['position']
            to_pos = to_node['position']
            
            # Calculate elevation angle
            lat1, lon1 = math.radians(from_pos['latitude']), math.radians(from_pos['longitude'])
            lat2, lon2 = math.radians(to_pos['latitude']), math.radians(to_pos['longitude'])
            
            # Ground distance
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            ground_dist_km = 6371 * c
            
            # Altitude difference
            alt_diff = to_pos.get('altitude', 0) - from_pos.get('altitude', 0)
            
            # Elevation angle in degrees
            if ground_dist_km > 0:
                elevation_deg = math.degrees(math.atan(alt_diff / ground_dist_km))
            else:
                elevation_deg = 90.0
            
            min_elevation = from_node['communication'].get('minElevationDeg', 10.0)
            
            logger.debug(f"Elevation from {from_node['nodeId']} to {to_node['nodeId']}: {elevation_deg:.2f}° (min: {min_elevation}°)")
            return elevation_deg >= min_elevation
        
        # For satellite-to-satellite or other cases, assume True if in range
        return True

    def calculate_reward(self, metrics: Dict, packet: Packet, current_node: Dict) -> float:
        reward = 0
        latency_ratio = metrics['latencyMs'] / packet.max_acceptable_latency_ms if packet.max_acceptable_latency_ms > 0 else 0
        bw_ratio = packet.service_qos['minBandwidthMbps'] / metrics['availableBandwidthMbps'] if metrics['availableBandwidthMbps'] > 0 else 0
        loss_ratio = metrics['packetLossRate'] / packet.max_acceptable_loss_rate if packet.max_acceptable_loss_rate > 0 else 0
        if latency_ratio > 1:
            reward -= 10 * (latency_ratio - 1) * (2 if packet.service_type in ["VIDEO_STREAM", "AUDIO_CALL"] else 1)
        if bw_ratio > 1:
            reward -= 10 * (bw_ratio - 1) * (2 if packet.service_type == "IMAGE_TRANSFER" else 1)
        if loss_ratio > 1:
            reward -= 10 * (loss_ratio - 1) * (2 if packet.service_type == "AUDIO_CALL" else 1)
        reward += 5  # Progress bonus
        reward -= 0.5 * (1 + current_node['position']['altitude'] / 1000)  # Hop cost, penalty for high altitude
        if packet.path_history.count(packet.current_holding_node_id) > 1:
            reward -= 20  # Loop penalty
        if len(packet.path_history) > 2:  # Bonus for multi-hop
            reward += 10
        return max(min(reward, 50), -50)
    
    async def train_step(self):
        if self.buffer.size() >= self.batch_size or (time.time() - self.last_train_time > 5):
            batch = self.buffer.sample(self.batch_size)
            loss = self.agent.learn(batch)
            self.agent.soft_update_target()
            self.agent.decay_epsilon()
            self.step_count += 1
            self.last_train_time = time.time()
            logger.info(f"Training step {self.step_count}, loss: {loss:.4f}, epsilon: {self.agent.epsilon:.4f}")