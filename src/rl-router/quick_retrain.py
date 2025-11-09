#!/usr/bin/env python3
"""
Quick retraining script to demonstrate the action masking and loop detection fixes.
This fine-tunes the existing model with the corrected training loop.
"""

import logging
import random
import os
import torch
from tqdm import tqdm
import numpy as np

from python.utils.db_connector import MongoConnector
from python.utils.state_builder import StateBuilder
from python.env.satellite_simulator import SatelliteEnv
from python.rl_agent.trainer import DQNAgent
from python.rl_agent.policy import get_epsilon

# Configuration
NUM_EPISODES = 1000  # Quick training for demonstration
MAX_HOPS_PER_EPISODE = 30
SAVE_INTERVAL = 100

BASE_CHECKPOINT_PATH = "models/checkpoints/dqn_checkpoint_fullpath_latest.pth"
NEW_CHECKPOINT_PATH = "models/checkpoints/dqn_checkpoint_retrained.pth"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_packet(node_list):
    """Generate a random packet between two different nodes."""
    if len(node_list) < 2:
        raise ValueError("Not enough nodes to create packet.")
    
    src = random.choice(node_list)
    dest = random.choice([n for n in node_list if n != src])
    
    return {
        "currentHoldingNodeId": src,
        "stationDest": dest,
        "accumulatedDelayMs": 0.0,
        "ttl": random.randint(35, 45),
        "serviceQoS": {
            "serviceType": random.choice(["VIDEO_STREAM", "AUDIO_CALL", "FILE_TRANSFER"]),
            "maxLatencyMs": random.uniform(100.0, 300.0),
            "minBandwidthMbps": random.uniform(2.0, 10.0),
            "maxLossRate": random.uniform(0.01, 0.05)
        },
        "dropped": False
    }


def quick_retrain():
    logger.info("=== QUICK RETRAINING WITH FIXES ===")
    logger.info("This demonstrates action masking and loop detection")
    
    # Initialize system
    mongo_conn = MongoConnector(uri="mongodb://user:password123@localhost:27017/")
    state_builder = StateBuilder(mongo_conn)
    env = SatelliteEnv(state_builder)
    
    # Create agent with legacy architecture to match existing checkpoint
    agent = DQNAgent(env, use_legacy_architecture=True)
    
    # Load existing checkpoint to fine-tune
    if os.path.exists(BASE_CHECKPOINT_PATH):
        logger.info(f"Loading base checkpoint: {BASE_CHECKPOINT_PATH}")
        checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location=torch.device('cpu'), weights_only=False)
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        agent.target_network.load_state_dict(checkpoint.get('target_network_state_dict', checkpoint['model_state_dict']))
        if 'optimizer_state_dict' in checkpoint:
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.steps_done = checkpoint.get('steps_done', 0)
        logger.info(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    else:
        logger.info("No base checkpoint found. Training from scratch.")
    
    # Get all nodes
    all_nodes_data = state_builder.db.get_all_nodes(projection={"nodeId": 1})
    all_nodes = [n["nodeId"] for n in all_nodes_data]
    logger.info(f"Training with {len(all_nodes)} nodes")
    
    # Training loop
    rewards = []
    success_count = 0
    
    pbar = tqdm(range(NUM_EPISODES), desc="Retraining")
    
    for episode in pbar:
        packet = generate_packet(all_nodes)
        
        # Track if packet reaches destination for statistics
        initial_src = packet["currentHoldingNodeId"]
        dest = packet["stationDest"]
        
        # Run episode with fixed training (action masking + loop detection)
        episode_reward = env.simulate_episode(agent, packet, max_hops=MAX_HOPS_PER_EPISODE)
        
        # Simple success tracking (positive reward typically means success)
        if episode_reward > 0:
            success_count += 1
        
        rewards.append(episode_reward)
        
        # Update progress bar
        avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        success_rate = (success_count / (episode + 1)) * 100
        epsilon = get_epsilon(agent.steps_done)
        
        pbar.set_postfix({
            'Reward': f"{episode_reward:.2f}",
            'Avg50': f"{avg_reward:.2f}",
            'Success%': f"{success_rate:.1f}",
            'Eps': f"{epsilon:.3f}"
        })
        
        # Save checkpoint periodically
        if (episode + 1) % SAVE_INTERVAL == 0:
            checkpoint_data = {
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'steps_done': agent.steps_done,
            }
            save_path = f"models/checkpoints/dqn_checkpoint_retrained_ep{episode+1}.pth"
            torch.save(checkpoint_data, save_path)
            logger.info(f"Saved checkpoint: {save_path}")
    
    # Save final checkpoint
    final_checkpoint = {
        'episode': NUM_EPISODES - 1,
        'model_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'steps_done': agent.steps_done,
    }
    torch.save(final_checkpoint, NEW_CHECKPOINT_PATH)
    logger.info(f"=== RETRAINING COMPLETE ===")
    logger.info(f"Final checkpoint saved: {NEW_CHECKPOINT_PATH}")
    logger.info(f"Average reward (last 50): {np.mean(rewards[-50:]):.2f}")
    logger.info(f"Success rate: {(success_count / NUM_EPISODES) * 100:.1f}%")


if __name__ == "__main__":
    quick_retrain()
