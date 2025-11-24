#!/usr/bin/env python
"""
Testing/Evaluation script for trained SAGIN RL routing agent.

Usage:
    python scripts/test.py --checkpoint checkpoints/models/agent_episode_1000.pth --episodes 100
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any, List

from agents.rl_agent import DQNAgent
from environments.dynamic_env import DynamicSatelliteEnv
from utils.state_builder import StateBuilder
from data.mongodb.connection import MongoDBManager


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_agent(agent: DQNAgent, env: DynamicSatelliteEnv,
                   num_episodes: int = 100) -> Dict[str, Any]:
    """
    Evaluate agent performance.

    Args:
        agent: Trained DQN agent
        env: Environment
        num_episodes: Number of episodes to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating agent for {num_episodes} episodes...")

    total_rewards = []
    delivery_count = 0
    latencies = []
    hops_list = []
    failed_episodes = 0

    for episode in range(num_episodes):
        # Generate random test packet
        packet_data = {
            "currentHoldingNodeId": np.random.choice(['gs1', 'gs2', 'gs3']),
            "stationDest": np.random.choice(['gs1', 'gs2', 'gs3', 'leo1', 'leo2']),
            "ttl": 50,
            "serviceQoS": {
                "maxLatencyMs": np.random.uniform(500.0, 2000.0)
            },
            "accumulatedDelayMs": 0.0,
            "packetSize": np.random.randint(512, 1500)
        }

        # Ensure source != destination
        while packet_data["currentHoldingNodeId"] == packet_data["stationDest"]:
            packet_data["stationDest"] = np.random.choice(['gs1', 'gs2', 'gs3', 'leo1', 'leo2'])

        # Run episode
        metrics = env.simulate_episode(
            agent=agent,
            initial_packet_data=packet_data,
            max_hops=12,
            is_training=False  # Evaluation mode (no exploration)
        )

        if isinstance(metrics, dict) and 'total_reward' in metrics:
            total_rewards.append(metrics['total_reward'])
        else:
            print(f"Warning: Unexpected metrics format: {metrics}")

        if isinstance(metrics, dict) and metrics.get('delivered', False):
            delivery_count += 1
            latencies.append(metrics['latency'])
            hops_list.append(metrics['hops'])
        else:
            failed_episodes += 1

        # Progress indicator
        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode + 1}/{num_episodes} episodes completed")

    # Calculate statistics
    results = {
        'num_episodes': num_episodes,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'delivery_rate': (delivery_count / num_episodes) * 100,
        'failed_episodes': failed_episodes,
        'avg_latency': np.mean(latencies) if latencies else 0.0,
        'std_latency': np.std(latencies) if latencies else 0.0,
        'avg_hops': np.mean(hops_list) if hops_list else 0.0,
        'std_hops': np.std(hops_list) if hops_list else 0.0,
    }

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted table"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total Episodes:            {results['num_episodes']}")
    print(f"")
    print(f"Average Reward:            {results['avg_reward']:.2f} � {results['std_reward']:.2f}")
    print(f"")
    print(f"Packet Delivery Rate:      {results['delivery_rate']:.2f}%")
    print(f"Failed Deliveries:         {results['failed_episodes']}")
    print(f"")
    print(f"Average Latency:           {results['avg_latency']:.2f} � {results['std_latency']:.2f} ms")
    print(f"Average Hops:              {results['avg_hops']:.2f} � {results['std_hops']:.2f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test/Evaluate SAGIN RL Routing Agent')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pth)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/dynamic_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes (default: 100)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results (JSON format)'
    )

    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic (greedy) policy'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Print test configuration
    print("\n" + "="*70)
    print("TEST CONFIGURATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config File: {args.config}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("="*70)

    # Initialize components
    print("\nInitializing environment and agent...")

    try:
        # Database connection
        db_config = config['database']

        # Support both connection_string and host/port formats
        if 'connection_string' in db_config:
            connection_string = db_config['connection_string']
        else:
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 27017)
            connection_string = f"mongodb://{host}:{port}/"

        db_manager = MongoDBManager(
            connection_string=connection_string,
            db_name=db_config.get('db_name', 'sagin_simulation'),
            username=db_config.get('username'),
            password=db_config.get('password'),
            auth_source=db_config.get('auth_source', 'admin')
        )

        # Get nodes from database
        nodes = {node['id']: node for node in db_manager.get_all_nodes()}
        print(f"Loaded {len(nodes)} nodes from database")

        if len(nodes) == 0:
            print("\nWARNING: No nodes found in database!")
            print("Please ensure MongoDB is running and contains network data.")
            return 1

        # State builder
        state_builder = StateBuilder(db_manager)

        # Environment
        env = DynamicSatelliteEnv(
            state_builder=state_builder,
            nodes=nodes,
            weights=config.get('reward_weights', {}),
            dynamic_config=config.get('dynamics', {})
        )

        # Agent
        agent = DQNAgent(env, use_legacy_architecture=False)

        # Load checkpoint
        print(f"Loading model from checkpoint: {args.checkpoint}")
        agent.q_network.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        agent.q_network.eval()  # Set to evaluation mode
        print("Checkpoint loaded successfully!")

    except Exception as e:
        print(f"\nERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run evaluation
    try:
        results = evaluate_agent(agent, env, num_episodes=args.episodes)

        # Print results
        print_results(results)

        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
