# python/main_train.py

import logging
import random
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from torch.utils.tensorboard import SummaryWriter 

# === Imports ===
# Đảm bảo đường dẫn module đúng
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Giả định các module utils của bạn đã hoạt động tốt với MockDB hoặc RealDB
try:
    from python.utils.db_connector import MongoConnector, LOCAL_MONGO_URI
except ImportError:
    # Fallback cho trường hợp test không có MongoDriver
    from python.utils.mock_db import MockDBConnector as MongoConnector
    LOCAL_MONGO_URI = "mongodb://localhost:27017"

from python.utils.state_builder import StateBuilder
from python.env.satellite_simulator import SatelliteEnv
from python.rl_agent.trainer import DQNAgent
from python.rl_agent.policy import get_epsilon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === CONFIG ===
NUM_EPISODES = 50000
MAX_HOPS_PER_EPISODE = 50
SAVE_INTERVAL = 500             # Save checkpoint mỗi 500 episodes
LOG_INTERVAL = 10               # Update TensorBoard/Tqdm mỗi 10 episodes

BASE_DIR = "models/checkpoints"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "dqn_checkpoint.pth")
LATEST_PATH = os.path.join(BASE_DIR, "dqn_latest.pth")
CHART_PATH = "training_charts.png"
LOG_DIR = "runs/dqn_experiment_v1"

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------- PACKET GENERATOR -----------------
def generate_packet(node_ids: List[str]) -> Dict[str, Any]:
    """Tạo packet với Source/Dest khác nhau."""
    if len(node_ids) < 2:
        raise ValueError("Network too small (<2 nodes)")
        
    src = random.choice(node_ids)
    # Cách lấy dest nhanh hơn list comprehension nếu danh sách node lớn
    while True:
        dest = random.choice(node_ids)
        if dest != src:
            break
            
    return {
        "currentHoldingNodeId": src,
        "stationDest": dest,
        "accumulatedDelayMs": 0.0,
        "ttl": random.randint(30, 50),
        "serviceQoS": {
            "maxLatencyMs": 500.0,
            "minBandwidthMbps": 5.0
        },
        "dropped": False,
        "path": [src] # Tracking path for debug
    }

# ----------------- VISUALIZATION -----------------
def save_charts(rewards, avg_rewards, coverage, path):
    """Vẽ biểu đồ tĩnh mà không block main thread."""
    try:
        plt.figure(figsize=(10, 8))
        
        # Subplot 1: Reward
        plt.subplot(2, 1, 1)
        plt.plot(rewards, alpha=0.3, color='gray', label='Raw')
        if len(avg_rewards) > 0:
            # Scale x-axis cho avg_rewards (vì nó ngắn hơn rewards 50 đơn vị)
            x_avg = np.arange(len(rewards) - len(avg_rewards), len(rewards))
            plt.plot(x_avg, avg_rewards, color='orange', linewidth=2, label='Avg (50)')
        plt.title('Training Reward')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Coverage
        plt.subplot(2, 1, 2)
        plt.plot(coverage, color='green')
        plt.title('Source-Dest Pair Coverage (%)')
        plt.ylabel('% Covered')
        plt.xlabel('Episode')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close() # Quan trọng: đóng figure để giải phóng RAM
    except Exception as e:
        logger.warning(f"Chart plotting failed: {e}")

# ----------------- MAIN TRAINING -----------------
def main():
    # 1. Init System
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    from python.utils.mock_db import MockDBConnector
    mongo = MockDBConnector()

    state_builder = StateBuilder(mongo)
    env = SatelliteEnv(state_builder)
    agent = DQNAgent(env, use_legacy_architecture=False) # Dùng Dueling DQN mới
    
    writer = SummaryWriter(log_dir=LOG_DIR)

    # 2. Get Node List
    # Giả sử db.get_all_nodes trả về list dict
    try:
        nodes_raw = state_builder.db.get_nodes([]) # Lấy sample hoặc tất cả
        if not nodes_raw and hasattr(state_builder.db, 'sample_node'):
             # Mock fallback
             nodes_raw = [state_builder.db.get_node(f"SAT_{i}") for i in range(10)]
        
        node_ids = [n.get('nodeId', 'UNKNOWN') for n in nodes_raw if n is not None]
        node_ids = [nid for nid in node_ids if nid != 'UNKNOWN']
    except Exception as e:
        logger.error(f"Failed to fetch nodes: {e}")
        return

    if len(node_ids) < 2:
        # Fallback cho Mock nếu DB rỗng
        logger.warning("DB empty. Using Mock Nodes.")
        node_ids = [f"SAT_{i}" for i in range(20)]

    total_pairs = len(node_ids) * (len(node_ids) - 1)
    logger.info(f"Nodes: {len(node_ids)} | Pairs: {total_pairs}")

    # 3. Variables
    rewards_history = []
    avg_rewards = []
    coverage_history = []
    trained_pairs = set()
    start_ep = 0

    # 4. Resume Checkpoint
    if os.path.exists(LATEST_PATH):
        logger.info(f"Resuming from {LATEST_PATH}...")
        try:
            # weights_only=False để load dict cấu trúc phức tạp
            ckpt = torch.load(
                LATEST_PATH, 
                map_location= device,
                weights_only=False  
            )
            
            agent.q_network.load_state_dict(ckpt['model'])
            agent.target_network.load_state_dict(ckpt['target'])
            agent.optimizer.load_state_dict(ckpt['optim'])
            agent.steps_done = ckpt['steps']
            
            start_ep = ckpt['episode'] + 1
            rewards_history = ckpt.get('rewards', [])
            avg_rewards = ckpt.get('avg_rewards', [])
            trained_pairs = ckpt.get('pairs', set())
            coverage_history = ckpt.get('coverage', [])
        except Exception as e:
            logger.error(f"Resume failed: {e}. Starting fresh.")

    # 5. Training Loop
    logger.info(">>> START TRAINING <<<")
    try:
        pbar = tqdm(range(start_ep, NUM_EPISODES), initial=start_ep, total=NUM_EPISODES)
        
        for ep in pbar:
            # A. Run Episode
            packet = generate_packet(node_ids)
            reward = env.simulate_episode(agent, packet, max_hops=MAX_HOPS_PER_EPISODE)
            
            # B. Metrics
            rewards_history.append(reward)
            trained_pairs.add((packet['currentHoldingNodeId'], packet['stationDest']))
            
            # C. Stats Calculation (Đừng tính quá nặng mỗi step)
            if len(rewards_history) >= 50:
                avg = np.mean(rewards_history[-50:])
                avg_rewards.append(avg)
            else:
                avg = np.mean(rewards_history)

            cov_pct = (len(trained_pairs) / max(1, total_pairs)) * 100
            coverage_history.append(cov_pct)
            epsilon = get_epsilon(agent.steps_done)

            # D. Log & Save
            if ep % LOG_INTERVAL == 0:
                writer.add_scalar('Train/Reward', reward, ep)
                writer.add_scalar('Train/AvgReward_50', avg, ep)
                writer.add_scalar('Train/Epsilon', epsilon, ep)
                writer.add_scalar('Train/Coverage', cov_pct, ep)
                
                pbar.set_postfix({'Avg': f"{avg:.1f}", 'Cov': f"{cov_pct:.1f}%", 'Eps': f"{epsilon:.2f}"})

            if (ep + 1) % SAVE_INTERVAL == 0:
                # Save Checkpoint
                state = {
                    'episode': ep,
                    'model': agent.q_network.state_dict(),
                    'target': agent.target_network.state_dict(),
                    'optim': agent.optimizer.state_dict(),
                    'steps': agent.steps_done,
                    'rewards': rewards_history,
                    'avg_rewards': avg_rewards,
                    'pairs': trained_pairs,
                    'coverage': coverage_history
                }
                torch.save(state, LATEST_PATH)
                
                # Save Chart
                save_charts(rewards_history, avg_rewards, coverage_history, CHART_PATH)

    except KeyboardInterrupt:
        logger.warning("\n!!! Training interrupted by User !!!")
    except Exception as e:
        logger.error(f"\n!!! CRASH: {e} !!!")
        raise e
    finally:
        # Emergency Save
        if len(rewards_history) > 0:
            logger.info("Saving Emergency Checkpoint...")
            state = {
                'episode': len(rewards_history), # Approximate
                'model': agent.q_network.state_dict(),
                'target': agent.target_network.state_dict(),
                'optim': agent.optimizer.state_dict(),
                'steps': agent.steps_done,
                'rewards': rewards_history,
                'avg_rewards': avg_rewards,
                'pairs': trained_pairs,
                'coverage': coverage_history
            }
            torch.save(state, LATEST_PATH)
        
        writer.close()
        logger.info("Done.")

if __name__ == "__main__":
    main()