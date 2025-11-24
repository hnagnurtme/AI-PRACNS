import torch
import numpy as np
from typing import Dict, Any
from agents.rl_agent import DQNAgent
from environments.dynamic_env import DynamicSatelliteEnv
from utils.state_builder import StateBuilder
from data.mongodb.connection import MongoDBManager
from simulation.core.node import Node
from simulation.core.packet import Packet, QoS

class DynamicTrainingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_components()
        
    def setup_components(self):
        db_config = self.config['database']

        # Support both connection_string and host/port formats
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
        nodes_dict = {node['node_id']: node for node in self.db_manager.get_all_nodes()}
        nodes = {node_id: Node.from_dict(node_data) for node_id, node_data in nodes_dict.items()}
        self.state_builder = StateBuilder(self.db_manager)
        self.env = DynamicSatelliteEnv(
            state_builder=self.state_builder,
            nodes=nodes,
            weights=self.config['reward_weights'],
            dynamic_config=self.config['dynamics']
        )
        self.agent = DQNAgent(self.env, self.config['rl_agent'], use_legacy_architecture=False)
        
    def train(self, num_episodes: int = 1000):
        print("Starting dynamic SAGIN RL training...")
        
        for episode in range(num_episodes):
            packet = self._generate_dynamic_packet(episode)
            episode_metrics = self.env.simulate_episode(
                agent=self.agent,
                initial_packet=packet,
                max_hops=self.config['training']['max_hops'],
                is_training=True
            )
            if not isinstance(episode_metrics, dict):
                raise TypeError("simulate_episode must return a dictionary with episode metrics.")
            
            if episode % 10 == 0:
                self._log_training_progress(episode, episode_metrics)
                
            if episode % 100 == 0:
                self._evaluate_agent(episode)
                self.agent.save_checkpoint(f"checkpoints/models/agent_episode_{episode}.pth")
                
        print("Training completed!")
    
    def _generate_dynamic_packet(self, episode: int) -> Packet:
        operational_nodes = self.env.get_operational_nodes()
        
        if len(operational_nodes) < 2:
            src_id = "gs1"
            dest_id = "gs2"
        else:
            src_node = operational_nodes[np.random.choice(len(operational_nodes))]
            src_id = src_node['nodeId']
            dest_ids = [n['nodeId'] for n in operational_nodes if n['nodeId'] != src_id]
            if not dest_ids:
                dest_id = src_id # Should not happen with >1 nodes
            else:
                dest_id = np.random.choice(dest_ids)

        qos = QoS(
            service_type='default',
            default_priority=0,
            max_latency_ms=np.random.uniform(500, 2000),
            max_jitter_ms=0,
            min_bandwidth_mbps=0,
            max_loss_rate=0.1
        )

        return Packet(
            packet_id=f'pkt-{episode}',
            source_user_id='user1',
            destination_user_id='user2',
            station_source=src_id,
            station_dest=dest_id,
            type='data',
            time_sent_from_source_ms=0,
            payload_data_base64='',
            payload_size_byte=np.random.randint(100, 1500),
            service_qos=qos,
            current_holding_node_id=src_id,
            next_hop_node_id='',
            priority_level=0,
            max_acceptable_latency_ms=qos.max_latency_ms,
            max_acceptable_loss_rate=0.1,
            analysis_data=None,
            use_rl=True,
            ttl=50
        )
    
    def _log_training_progress(self, episode: int, metrics: Dict[str, Any]):
        reward = metrics['total_reward']
        delivered = "Yes" if metrics['delivered'] else "No"
        latency = metrics['latency']
        hops = metrics['hops']
        print(f"[Episode {episode:03d}] Reward: {reward:.2f} | Delivered: {delivered} | Latency: {latency:.2f}ms | Hops: {hops}")

    def _evaluate_agent(self, episode: int):
        print(f"\n--- Evaluating agent at episode {episode} ---")
        num_eval_episodes = self.config['evaluation'].get('num_episodes', 50)
        
        total_rewards = []
        successful_deliveries = 0
        total_latency = 0
        total_hops = 0
        
        for i in range(num_eval_episodes):
            packet_data = self._generate_dynamic_packet(i)
            episode_metrics = self.env.simulate_episode(
                agent=self.agent,
                initial_packet=packet_data,
                max_hops=self.config['training']['max_hops'],
                is_training=False  # Evaluation mode
            )
            
            if isinstance(episode_metrics, dict) and 'total_reward' in episode_metrics:
                total_rewards.append(episode_metrics['total_reward'])
            else:
                raise TypeError("simulate_episode must return a dictionary with a 'total_reward' key.")
            if episode_metrics['delivered']:
                successful_deliveries += 1
                total_latency += episode_metrics['latency']
                total_hops += episode_metrics['hops']

        avg_reward = np.mean(total_rewards)
        delivery_rate = (successful_deliveries / num_eval_episodes) * 100
        avg_latency = total_latency / successful_deliveries if successful_deliveries > 0 else 0
        avg_hops = total_hops / successful_deliveries if successful_deliveries > 0 else 0

        print(f"Evaluation Results:")
        print(f"  - Average Reward: {avg_reward:.2f}")
        print(f"  - Packet Delivery Rate: {delivery_rate:.2f}%")
        print(f"  - Average Latency (for delivered packets): {avg_latency:.2f}ms")
        print(f"  - Average Hops (for delivered packets): {avg_hops:.2f}")
        print("-------------------------------------------------\n")