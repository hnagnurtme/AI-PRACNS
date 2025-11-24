#!/usr/bin/env python3
"""
Dynamic training script with weather, mobility, traffic, and failures.
"""

import sys
import yaml
import numpy as np

from data.mongodb.connection import MongoDBManager
from agents.rl_agent import DQNAgent
from environments.satellite_env import SatelliteEnv
from simulation.core.network import SAGINNetwork
from utils.state_builder import StateBuilder

# Load config
with open('configs/dynamic_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup database
db_config = config['database']
if 'connection_string' in db_config:
    connection_string = db_config['connection_string']
else:
    connection_string = f"mongodb://{db_config['host']}:{db_config['port']}/"
db_manager = MongoDBManager(
    connection_string=connection_string,
    db_name=db_config['db_name'],
    username=db_config.get('username'),
    password=db_config.get('password'),
    auth_source=db_config.get('auth_source', 'admin')
)

print("="*70)
print("DYNAMIC RL TRAINING (Weather + Mobility + Traffic + Failures)")
print("="*70)

# Setup network and environment
state_builder = StateBuilder(db_manager)
network = SAGINNetwork(db_manager)
network.initialize_network(config.get('network', {}))


env = SatelliteEnv(
    state_builder=state_builder,
    network=network,
    weights=config.get('reward_weights', {})
)

# Create agent
agent = DQNAgent(env, config['rl_agent'], use_legacy_architecture=False)

from simulation.core.packet import Packet, QoS
# ... (other imports)

# ... (config loading and db setup)

# ... (node loading)

# ... (environment setup)

# ... (agent creation)

# Training parameters
num_episodes = config['training'].get('num_episodes', 3000)
max_hops = config['training'].get('max_hops', 15)

print(f"Training for {num_episodes} episodes with FULL dynamics...")
print(f"Dynamics enabled: Weather, Mobility, Traffic, Failures")
print("="*70)

# Training loop
successful_deliveries = 0
for episode in range(num_episodes):
    # Generate random packet
    operational_nodes = [node.nodeId for node in env.get_operational_nodes()]
    if len(operational_nodes) < 2:
        continue

    src, dest = np.random.choice(operational_nodes, size=2, replace=False)

    qos = QoS(
        service_type='default',
        default_priority=0,
        max_latency_ms=np.random.uniform(500, 2000),
        max_jitter_ms=0,
        min_bandwidth_mbps=0,
        max_loss_rate=0
    )

    packet = Packet(
        packet_id=f'pkt-{episode}',
        source_user_id='user1',
        destination_user_id='user2',
        station_source=src,
        station_dest=dest,
        type='data',
        time_sent_from_source_ms=0,
        payload_data_base64='',
        payload_size_byte=np.random.randint(100, 1500),
        service_qos=qos,
        current_holding_node_id=src,
        next_hop_node_id='',
        priority_level=0,
        max_acceptable_latency_ms=qos.max_latency_ms,
        max_acceptable_loss_rate=0.1,
        analysis_data=AnalysisData(),  # Replace None with a default AnalysisData instance
        use_rl=True,
        ttl=50
    )

    # Run episode
    metrics = env.simulate_episode(agent, packet, max_hops=max_hops)

    if metrics['delivered']:
        successful_deliveries += 1

    # Log progress
    if episode % 10 == 0:
        delivery_rate = successful_deliveries / (episode + 1) * 100
        print(f"Ep {episode:4d}: R={metrics['total_reward']:7.2f}, "
              f"Del={'✓' if metrics['delivered'] else '✗'}, "
              f"Hops={metrics['hops']:2d}, Lat={metrics['latency']:5.1f}ms, "
              f"DelivRate={delivery_rate:5.2f}%")

    # Save checkpoint
    if episode % 200 == 0 and episode > 0:
        checkpoint_path = f"checkpoints/models/dynamic_agent_ep{episode}.pth"
        agent.save_checkpoint(checkpoint_path)
        delivery_rate = successful_deliveries / episode * 100
        print(f"  → Checkpoint saved: {checkpoint_path} (Delivery: {delivery_rate:.1f}%)")


print("="*70)
print("TRAINING COMPLETE!")
print(f"Final delivery rate: {successful_deliveries / num_episodes * 100:.2f}%")
print("="*70)

# Save final model
final_path = "checkpoints/models/dynamic_agent_final.pth"
agent.save_checkpoint(final_path)
print(f"Final model saved to: {final_path}")

db_manager.close_connection()
