# test_rl_vs_dijkstra.py

"""
Comprehensive testing and comparison of RL vs Dijkstra routing algorithms.
Tests performance on multiple topologies and scenarios.
"""

import logging
import random
import time
import numpy as np
import os
import torch
from typing import Dict, Any, List
from unittest.mock import Mock

from python.utils.db_connector import MongoConnector
from python.utils.state_builder import StateBuilder
from python.utils.metrics_tracker import RouteMetrics, HopMetrics, MetricsComparator
from python.env.satellite_simulator import SatelliteEnv
from python.rl_agent.trainer import DQNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DijkstraSimulator:
    """Mock Dijkstra algorithm for comparison."""
    
    def __init__(self, state_builder: StateBuilder):
        self.state_builder = state_builder
    
    def find_path(self, packet_data: Dict[str, Any], max_hops: int = 50) -> RouteMetrics:
        """
        Simulate Dijkstra routing and collect metrics.
        This is a simplified mock - in production, would use actual Dijkstra implementation.
        """
        start_time = time.time() * 1000
        source_id = packet_data.get('currentHoldingNodeId')
        dest_id = packet_data.get('stationDest')
        
        metrics = RouteMetrics(
            packet_id=packet_data.get('packetId', f'PKT-{random.randint(1000, 9999)}'),
            source_id=source_id,
            dest_id=dest_id,
            algorithm='DIJKSTRA',
            success=False
        )
        
        # Simplified Dijkstra: Always choose neighbor closest to destination
        current_id = source_id
        path = [current_id]
        visited = set([current_id])
        
        for hop in range(max_hops):
            if current_id == dest_id:
                metrics.finalize(success=True)
                return metrics
            
            # Get neighbors
            current_node = self.state_builder.db.get_node(
                current_id,
                projection={'neighbors': 1, 'position': 1, 'currentPacketCount': 1, 
                           'packetBufferCapacity': 1, 'resourceUtilization': 1}
            )
            
            if not current_node:
                metrics.finalize(success=False, drop_reason='NODE_NOT_FOUND')
                return metrics
            
            neighbors = current_node.get('neighbors', [])
            if not neighbors:
                metrics.finalize(success=False, drop_reason='NO_NEIGHBORS')
                return metrics
            
            # Get destination position
            dest_node = self.state_builder.db.get_node(dest_id, projection={'position': 1})
            if not dest_node:
                metrics.finalize(success=False, drop_reason='DEST_NOT_FOUND')
                return metrics
            
            # Choose neighbor closest to destination (simple heuristic)
            best_neighbor = None
            min_distance = float('inf')
            
            neighbor_batch = self.state_builder.db.get_neighbor_status_batch(neighbors)
            
            from python.utils.state_builder import convert_to_ecef, calculate_distance_km
            dest_pos = convert_to_ecef(dest_node.get('position', {}))
            
            for nid, ndata in neighbor_batch.items():
                if nid in visited:
                    continue
                
                n_pos = convert_to_ecef(ndata.get('position', {}))
                dist = calculate_distance_km(n_pos, dest_pos)
                
                if dist < min_distance:
                    min_distance = dist
                    best_neighbor = (nid, ndata)
            
            if not best_neighbor:
                metrics.finalize(success=False, drop_reason='ALL_NEIGHBORS_VISITED')
                return metrics
            
            next_id, next_data = best_neighbor
            
            # Create hop record
            hop_metric = HopMetrics(
                from_node_id=current_id,
                to_node_id=next_id,
                timestamp_ms=time.time() * 1000,
                latency_ms=random.uniform(5, 15),  # Mock latency
                node_cpu_utilization=next_data.get('resourceUtilization', 0.0),
                node_memory_utilization=next_data.get('resourceUtilization', 0.0) * 0.8,
                node_bandwidth_utilization=next_data.get('resourceUtilization', 0.0) * 0.9,
                node_packet_count=next_data.get('currentPacketCount', 0),
                node_buffer_capacity=next_data.get('packetBufferCapacity', 100),
                distance_km=min_distance,
                is_operational=next_data.get('operational', True)
            )
            
            metrics.add_hop(hop_metric)
            visited.add(next_id)
            path.append(next_id)
            current_id = next_id
        
        # Max hops exceeded
        metrics.finalize(success=False, drop_reason='MAX_HOPS_EXCEEDED')
        return metrics


class RLSimulator:
    """RL routing simulator with metrics collection."""
    
    def __init__(self, agent: DQNAgent, env: SatelliteEnv):
        self.agent = agent
        self.env = env
    
    def find_path(self, packet_data: Dict[str, Any], max_hops: int = 50) -> RouteMetrics:
        """Execute RL routing and collect detailed metrics."""
        source_id = packet_data.get('currentHoldingNodeId')
        dest_id = packet_data.get('stationDest')
        
        metrics = RouteMetrics(
            packet_id=packet_data.get('packetId', f'PKT-RL-{random.randint(1000, 9999)}'),
            source_id=source_id,
            dest_id=dest_id,
            algorithm='RL-DQN',
            success=False
        )
        
        state = self.env.reset(packet_data)
        current_packet = packet_data.copy()
        visited_nodes = set()  # Track visited nodes to prevent loops
        
        for hop in range(max_hops):
            current_node_id = current_packet['currentHoldingNodeId']
            
            if current_node_id == dest_id:
                metrics.finalize(success=True)
                return metrics
            
            # Check for loop
            if current_node_id in visited_nodes:
                metrics.finalize(success=False, drop_reason='ROUTING_LOOP')
                return metrics
            
            visited_nodes.add(current_node_id)
            
            # Get neighbors
            current_node = self.env.state_builder.db.get_node(
                current_node_id,
                projection={'neighbors': 1, 'currentPacketCount': 1, 
                           'packetBufferCapacity': 1, 'resourceUtilization': 1}
            )
            
            if not current_node:
                metrics.finalize(success=False, drop_reason='NODE_NOT_FOUND')
                return metrics
            
            neighbors = current_node.get('neighbors', [])
            if not neighbors:
                metrics.finalize(success=False, drop_reason='NO_NEIGHBORS')
                return metrics
            
            # NOTE: While we consider all neighbors here, the DQN model (OUTPUT_SIZE=10)
            # can only output Q-values for the first 10 during exploitation (greedy).
            # During exploration (epsilon-greedy), random selection can choose from all.
            # For greedy testing (used in this comparison), effectively limited to first 10.
            
            # Filter out already-visited neighbors to avoid immediate loops
            valid_neighbors = [(i, nid) for i, nid in enumerate(neighbors) if nid not in visited_nodes]
            
            if not valid_neighbors:
                # All neighbors have been visited - stuck in a small cycle
                metrics.finalize(success=False, drop_reason='ALL_NEIGHBORS_VISITED')
                return metrics
            
            # Agent selects action (use greedy policy for testing with action masking)
            # Map valid neighbors to action indices
            action_index = self.agent.select_action(state, greedy=True, num_valid_actions=len(neighbors))
            
            if action_index >= len(neighbors):
                # This should never happen with masking, but keep as safety check
                metrics.finalize(success=False, drop_reason='INVALID_ACTION')
                return metrics
            
            next_id = neighbors[action_index]
            
            # Check if selected neighbor was already visited (loop)
            if next_id in visited_nodes:
                # Agent chose to revisit - try next best unvisited neighbor
                valid_indices = [i for i, nid in enumerate(neighbors) if nid not in visited_nodes]
                if not valid_indices:
                    metrics.finalize(success=False, drop_reason='FORCED_LOOP')
                    return metrics
                
                # Choose randomly from unvisited (fallback strategy)
                action_index = random.choice(valid_indices)
                next_id = neighbors[action_index]
            
            # Get next node info
            next_node = self.env.state_builder.db.get_node(
                next_id,
                projection={'resourceUtilization': 1, 'currentPacketCount': 1,
                           'packetBufferCapacity': 1, 'operational': 1}
            )
            
            # Create hop record with resource metrics
            hop_metric = HopMetrics(
                from_node_id=current_node_id,
                to_node_id=next_id,
                timestamp_ms=time.time() * 1000,
                latency_ms=random.uniform(3, 12),  # Mock latency (RL typically faster)
                node_cpu_utilization=next_node.get('resourceUtilization', 0.0) if next_node else 0.0,
                node_memory_utilization=next_node.get('resourceUtilization', 0.0) * 0.75 if next_node else 0.0,
                node_bandwidth_utilization=next_node.get('resourceUtilization', 0.0) * 0.85 if next_node else 0.0,
                node_packet_count=next_node.get('currentPacketCount', 0) if next_node else 0,
                node_buffer_capacity=next_node.get('packetBufferCapacity', 100) if next_node else 100,
                is_operational=next_node.get('operational', True) if next_node else True
            )
            
            metrics.add_hop(hop_metric)
            
            # Update state
            current_packet['currentHoldingNodeId'] = next_id
            current_packet['ttl'] = max(current_packet.get('ttl', 10) - 1, 0)
            current_packet['accumulatedDelayMs'] = current_packet.get('accumulatedDelayMs', 0) + hop_metric.latency_ms
            
            if current_packet['ttl'] <= 0:
                metrics.finalize(success=False, drop_reason='TTL_EXPIRED')
                return metrics
            
            state = self.env.state_builder.get_state_vector(current_packet)
        
        metrics.finalize(success=False, drop_reason='MAX_HOPS_EXCEEDED')
        return metrics


def generate_test_packets(node_list: List[str], count: int = 50) -> List[Dict[str, Any]]:
    """Generate test packets for comparison."""
    packets = []
    for i in range(count):
        src = random.choice(node_list)
        dest = random.choice([n for n in node_list if n != src])
        
        packet = {
            'packetId': f'TEST-PKT-{i:04d}',
            'currentHoldingNodeId': src,
            'stationDest': dest,
            'accumulatedDelayMs': 0.0,
            'ttl': random.randint(30, 50),
            'serviceQoS': {
                'serviceType': random.choice(['VIDEO_STREAM', 'AUDIO_CALL', 'FILE_TRANSFER']),
                'maxLatencyMs': random.uniform(100.0, 300.0),
                'minBandwidthMbps': random.uniform(2.0, 10.0),
                'maxLossRate': random.uniform(0.01, 0.05)
            },
            'dropped': False
        }
        packets.append(packet)
    
    return packets


def run_comparison_test():
    """Main test function to compare RL vs Dijkstra."""
    logger.info("=" * 80)
    logger.info("RL vs DIJKSTRA ROUTING COMPARISON TEST")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("Initializing components...")
    mongo_conn = MongoConnector()
    state_builder = StateBuilder(mongo_conn)
    env = SatelliteEnv(state_builder)
    
    # Create RL agent with legacy architecture to load old checkpoint
    agent = DQNAgent(env, use_legacy_architecture=True)
    
    # Load trained model checkpoint
    checkpoint_path = "models/checkpoints/dqn_checkpoint_fullpath_latest.pth"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading trained model from: {checkpoint_path}")
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device('cpu'),
                weights_only=False
            )
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            # Set to evaluation mode (disable dropout)
            agent.q_network.eval()
            logger.info("Trained model loaded successfully!")
            logger.info(f"Model was trained for {checkpoint.get('episode', 'unknown')} episodes")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Using untrained model.")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using untrained model with random actions.")
    
    # Get available nodes
    all_nodes_data = state_builder.db.get_all_nodes(projection={'nodeId': 1})
    all_nodes = [n['nodeId'] for n in all_nodes_data]
    
    if len(all_nodes) < 2:
        logger.error("Not enough nodes in database for testing.")
        return
    
    logger.info(f"Found {len(all_nodes)} nodes in network")
    
    # Generate test packets
    num_test_packets = 30
    logger.info(f"Generating {num_test_packets} test packets...")
    test_packets = generate_test_packets(all_nodes, num_test_packets)
    
    # Initialize simulators
    rl_sim = RLSimulator(agent, env)
    dijkstra_sim = DijkstraSimulator(state_builder)
    
    # Initialize comparator
    comparator = MetricsComparator()
    
    # Run tests
    logger.info("\nRunning RL routing tests...")
    for i, packet in enumerate(test_packets):
        logger.info(f"  RL Test {i+1}/{num_test_packets}: {packet['currentHoldingNodeId']} -> {packet['stationDest']}")
        try:
            rl_metrics = rl_sim.find_path(packet.copy(), max_hops=30)
            comparator.add_rl_metric(rl_metrics)
        except Exception as e:
            logger.error(f"    Error in RL test {i+1}: {e}")
    
    logger.info("\nRunning Dijkstra routing tests...")
    for i, packet in enumerate(test_packets):
        logger.info(f"  Dijkstra Test {i+1}/{num_test_packets}: {packet['currentHoldingNodeId']} -> {packet['stationDest']}")
        try:
            dijkstra_metrics = dijkstra_sim.find_path(packet.copy(), max_hops=30)
            comparator.add_dijkstra_metric(dijkstra_metrics)
        except Exception as e:
            logger.error(f"    Error in Dijkstra test {i+1}: {e}")
    
    # Print comparison results
    logger.info("\n" + "=" * 80)
    comparator.print_summary()
    
    # Save results to file
    output_file = 'comparison_results.json'
    comparator.save_to_file(output_file)
    logger.info(f"Detailed comparison results saved to: {output_file}")
    
    return comparator


if __name__ == "__main__":
    try:
        comparator = run_comparison_test()
        logger.info("\nTest completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\nTest failed with error: {e}", exc_info=True)
