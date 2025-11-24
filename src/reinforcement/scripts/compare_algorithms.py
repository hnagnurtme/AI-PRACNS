#!/usr/bin/env python
"""
Algorithm Comparison Script - Compares RL agent with baseline routing algorithms.

Usage:
    python scripts/compare_algorithms.py --config configs/base_config.yaml --model checkpoints/models/best_agent.pth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import numpy as np
from typing import Dict, List, Any
import time
from dataclasses import asdict

from data.mongodb.connection import MongoDBManager
from agents.rl_agent import DQNAgent
from algorithms.dijkstra import DijkstraRouter
from algorithms.baseline import GreedyRouter, RandomRouter, QualityAwareRouter
from utils.state_builder import StateBuilder
from environments.dynamic_env import DynamicSatelliteEnv
from simulation.core.packet import Packet, QoS
from utils.node_converter import nodes_dict_to_objects


class AlgorithmComparator:
    """
    Compares different routing algorithms on the same test scenarios.
    """

    def __init__(self, config: Dict, model_path: str):
        """
        Initialize comparator.

        Args:
            config: Configuration dictionary
            model_path: Path to trained RL model
        """
        self.config = config
        self.model_path = model_path

        # Setup database connection
        db_config = config['database']
        if 'connection_string' in db_config:
            connection_string = db_config['connection_string']
        else:
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 27017)
            connection_string = f"mongodb://{host}:{port}/"

        self.db_manager = MongoDBManager(
            connection_string=connection_string,
            db_name=db_config.get('db_name', 'sagin_simulation'),
            username=db_config.get('username'),
            password=db_config.get('password'),
            auth_source=db_config.get('auth_source', 'admin')
        )

        # Get nodes from database and convert to Node objects
        nodes_data = {node['node_id']: node for node in self.db_manager.get_all_nodes()}
        self.nodes = nodes_dict_to_objects(nodes_data)
        print(f"Loaded {len(self.nodes)} nodes from database")

        # Initialize RL agent using DynamicSatelliteEnv for realistic conditions
        state_builder = StateBuilder(self.db_manager)
        env = DynamicSatelliteEnv(
            state_builder=state_builder,
            nodes=self.nodes,
            weights=config.get('reward_weights', {}),
            dynamic_config=config.get('dynamics', {})
        )
        self.rl_agent = DQNAgent(env, config['rl_agent'], use_legacy_architecture=False)
        if os.path.exists(model_path):
            self.rl_agent.load_checkpoint(model_path)
            print(f"Loaded RL model from {model_path}")
        else:
            print(f"WARNING: Model file not found at {model_path}")

        self.env = env
        self.state_builder = state_builder

        # Initialize baseline algorithms
        self.dijkstra = DijkstraRouter(self.db_manager)
        self.greedy = GreedyRouter(self.db_manager)
        self.random = RandomRouter(self.db_manager, seed=42)
        self.quality_aware = QualityAwareRouter(self.db_manager)

    def generate_test_scenarios(self, num_scenarios: int = 100) -> List[Packet]:
        """
        Generate test routing scenarios.

        Args:
            num_scenarios: Number of test scenarios to generate

        Returns:
            List of test packet objects
        """
        scenarios = []
        operational_nodes = [
            node_id for node_id, node in self.nodes.items()
            if getattr(node, 'isOperational', True)
        ]

        if len(operational_nodes) < 2:
            print("ERROR: Not enough operational nodes")
            return scenarios

        np.random.seed(42)  # For reproducibility

        for i in range(num_scenarios):
            # Randomly select source and destination
            source, dest = np.random.choice(operational_nodes, size=2, replace=False)

            qos = QoS(
                service_type='default',
                default_priority=0,
                max_latency_ms=np.random.uniform(500, 2000),
                max_jitter_ms=0,
                min_bandwidth_mbps=0,
                max_loss_rate=0
            )

            packet = Packet(
                packet_id=f'pkt-eval-{i}',
                source_user_id='user1',
                destination_user_id='user2',
                station_source=source,
                station_dest=dest,
                type='data',
                time_sent_from_source_ms=0,
                payload_data_base64='',
                payload_size_byte=np.random.randint(100, 1500),
                service_qos=qos,
                current_holding_node_id=source,
                next_hop_node_id='',
                priority_level=0,
                max_acceptable_latency_ms=qos.max_latency_ms,
                max_acceptable_loss_rate=0.1,
                analysis_data=None,
                use_rl=True,
                ttl=50
            )
            scenarios.append(packet)

        return scenarios

    def evaluate_rl_agent(self, scenarios: List[Packet], max_hops: int = 15) -> Dict:
        """
        Evaluate RL agent on test scenarios.

        Args:
            scenarios: List of test packets
            max_hops: Maximum hops allowed

        Returns:
            Evaluation metrics
        """
        print("\nEvaluating RL Agent...")
        results = {
            'successful': 0,
            'total_delay': 0.0,
            'total_hops': 0,
            'total_time': 0.0,
            'paths': []
        }

        for packet in scenarios:
            start_time = time.time()

            # Run RL agent
            episode_metrics = self.env.simulate_episode(
                self.rl_agent,
                packet,
                max_hops=max_hops,
                is_training=False  # Evaluation mode
            )

            elapsed = time.time() - start_time

            if episode_metrics['delivered']:
                results['successful'] += 1
                results['total_delay'] += episode_metrics['latency']
                results['total_hops'] += episode_metrics['hops']

            results['total_time'] += elapsed

        # Calculate averages
        num_scenarios = len(scenarios)
        num_successful = results['successful']

        return {
            'algorithm': 'RL',
            'delivery_rate': num_successful / num_scenarios if num_scenarios > 0 else 0,
            'avg_delay': results['total_delay'] / num_successful if num_successful > 0 else 0,
            'avg_hops': results['total_hops'] / num_successful if num_successful > 0 else 0,
            'avg_computation_time': results['total_time'] / num_scenarios if num_scenarios > 0 else 0,
            'total_successful': num_successful,
            'total_scenarios': num_scenarios
        }

    def evaluate_baseline(self, algorithm_name: str, router, scenarios: List[Packet],
                         max_hops: int = 15) -> Dict:
        """
        Evaluate a baseline algorithm with dynamic conditions.

        Args:
            algorithm_name: Name of algorithm
            router: Router instance
            scenarios: Test scenarios
            max_hops: Maximum hops

        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating {algorithm_name}...")
        results = {
            'successful': 0,
            'total_delay': 0.0,
            'total_hops': 0,
            'total_time': 0.0
        }

        for packet in scenarios:
            start_time = time.time()

            # Manually reset environment for fair comparison
            self.env.simulation_time = 0.0
            self.env.mobility_manager.reset()
            self.env.weather_model.reset()
            self.env.traffic_model.reset()
            self.env.failure_model.reset()
            dynamic_state = self.env.step_dynamics()

            # Update communication quality for all nodes
            time_of_day = (self.env.simulation_time % 86400) / 86400.0
            for node_id, node_obj in self.nodes.items():
                # A simplified distance calculation for the update
                distance = 6371 + node_obj.position.altitude
                node_obj.communication.update_communication_quality(
                    weather_impact=dynamic_state['weather_impact'],
                    distance=distance,
                    traffic_load=dynamic_state['traffic_load'],
                    time_of_day=time_of_day
                )

            # Get current node states after dynamics
            current_nodes = {}
            for node_id, node_obj in self.nodes.items():
                # Convert Node object to dict for baseline algorithms
                current_nodes[node_id] = {
                    'node_id': node_id,
                    'nodeType': node_obj.nodeType,
                    'position': {
                        'lat': node_obj.position.latitude,
                        'lon': node_obj.position.longitude,
                        'alt': node_obj.position.altitude
                    },
                    'isOperational': node_obj.isOperational,
                    'battery': node_obj.batteryChargePercent,
                    'congestion': node_obj.resourceUtilization,
                    'link_quality': node_obj.communication.link_quality,
                    'neighbors': self.env.mobility_manager.get_current_neighbors(node_id, dynamic_state),
                    'delay': node_obj.nodeProcessingDelayMs
                }

            # Route packet with dynamic network state
            result = router.route_packet(asdict(packet), current_nodes, max_hops=max_hops)

            elapsed = time.time() - start_time

            if result['success']:
                results['successful'] += 1
                results['total_delay'] += result['total_delay']
                results['total_hops'] += result['hops']

            results['total_time'] += elapsed

        num_scenarios = len(scenarios)
        num_successful = results['successful']

        return {
            'algorithm': algorithm_name,
            'delivery_rate': num_successful / num_scenarios if num_scenarios > 0 else 0,
            'avg_delay': results['total_delay'] / num_successful if num_successful > 0 else 0,
            'avg_hops': results['total_hops'] / num_successful if num_successful > 0 else 0,
            'avg_computation_time': results['total_time'] / num_scenarios if num_scenarios > 0 else 0,
            'total_successful': num_successful,
            'total_scenarios': num_scenarios
        }

    def run_comparison(self, num_scenarios: int = 100, max_hops: int = 15):
        """
        Run full comparison of all algorithms.

        Args:
            num_scenarios: Number of test scenarios
            max_hops: Maximum hops allowed
        """
        print("="*70)
        print("ALGORITHM COMPARISON")
        print("="*70)
        print(f"Test Scenarios: {num_scenarios}")
        print(f"Max Hops: {max_hops}")
        print(f"Number of Nodes: {len(self.nodes)}")
        print("="*70)

        # Generate test scenarios
        scenarios = self.generate_test_scenarios(num_scenarios)
        print(f"\nGenerated {len(scenarios)} test scenarios")

        # Evaluate all algorithms
        results = []

        # RL Agent
        rl_results = self.evaluate_rl_agent(scenarios, max_hops)
        results.append(rl_results)

        # Dijkstra
        dijkstra_results = self.evaluate_baseline('Dijkstra', self.dijkstra, scenarios, max_hops)
        results.append(dijkstra_results)

        # Greedy
        greedy_results = self.evaluate_baseline('Greedy', self.greedy, scenarios, max_hops)
        results.append(greedy_results)

        # Random
        random_results = self.evaluate_baseline('Random', self.random, scenarios, max_hops)
        results.append(random_results)

        # Quality Aware
        quality_results = self.evaluate_baseline('QualityAware', self.quality_aware, scenarios, max_hops)
        results.append(quality_results)

        # Print results
        self.print_results(results)

        return results

    def print_results(self, results: List[Dict]):
        """
        Print comparison results in a formatted table.

        Args:
            results: List of result dictionaries
        """
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"{'Algorithm':<15} {'Delivery':<10} {'Avg Delay':<12} {'Avg Hops':<10} {'Comp Time':<12}")
        print(f"{'':15} {'Rate':<10} {'(ms)':<12} {'':10} {'(ms)':<12}")
        print("-"*70)

        for result in results:
            print(f"{result['algorithm']:<15} "
                  f"{result['delivery_rate']*100:>8.2f}%  "
                  f"{result['avg_delay']:>10.2f}  "
                  f"{result['avg_hops']:>8.2f}  "
                  f"{result['avg_computation_time']*1000:>10.4f}")

        print("="*70)

        # Print relative performance
        if len(results) > 1:
            rl_result = results[0]
            print("\nRL Agent Performance vs Baselines:")
            print("-"*70)

            for i in range(1, len(results)):
                baseline = results[i]
                delivery_improvement = (rl_result['delivery_rate'] - baseline['delivery_rate']) * 100
                delay_improvement = ((baseline['avg_delay'] - rl_result['avg_delay']) /
                                   baseline['avg_delay'] * 100) if baseline['avg_delay'] > 0 else 0

                print(f"vs {baseline['algorithm']:12}: "
                      f"Delivery: {delivery_improvement:+.2f}%, "
                      f"Delay: {delay_improvement:+.2f}%")

        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare Routing Algorithms')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/dynamic_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/models/dynamic_agent_final.pth',
        help='Path to trained RL model'
    )

    parser.add_argument(
        '--scenarios',
        type=int,
        default=100,
        help='Number of test scenarios'
    )

    parser.add_argument(
        '--max-hops',
        type=int,
        default=15,
        help='Maximum number of hops'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run comparison
    comparator = AlgorithmComparator(config, args.model)
    comparator.run_comparison(num_scenarios=args.scenarios, max_hops=args.max_hops)


if __name__ == "__main__":
    main()
