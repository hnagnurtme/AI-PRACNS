# python/main_train.py

import logging
import random
from tqdm import tqdm
from python.utils.db_connector import MongoConnector
from python.utils.state_builder import StateBuilder
from python.env.satellite_simulator import SatelliteEnv
from python.rl_agent.trainer import DQNAgent, TARGET_UPDATE
import numpy as np

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 200
CHECKPOINT_PATH = "models/checkpoints/dqn_checkpoint.pth"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mock_initial_packet():
    return {
        "currentHoldingNodeId": "SAT-LEO-2",
        "stationDest": "N-TOKYO",
        "accumulatedDelayMs": 0.0,
        "ttl": 20,
        "serviceQoS": {"serviceType": "VIDEO_STREAM", "maxLatencyMs": 150.0, "minBandwidthMbps": 5.0, "maxLossRate": 0.02},
        "dropped": False
    }

def mock_next_step_packet(prev_packet):
    next_packet = prev_packet.copy()
    next_packet['ttl'] -= 1
    next_packet['accumulatedDelayMs'] += 15.0
    if next_packet['ttl'] <= 0:
        next_packet['dropped'] = True
    elif random.random() < 0.05:
        next_packet['currentHoldingNodeId'] = next_packet['stationDest']
    else:
        next_packet['currentHoldingNodeId'] = random.choice(["SAT-LEO-2", "N-HANOI", "N-TOKYO", "SAT-MEO-5"])
    return next_packet

def train_agent():
    logger.info("--- KHỞI TẠO HỆ THỐNG DQN ROUTER ---")
    mongo_conn = MongoConnector(uri="mongodb://user:password123@localhost:27017/?authSource=admin")
    state_builder = StateBuilder(mongo_conn)
    reward_weights = {'goal': 200.0, 'drop': 100.0, 'latency': -10.0}

    env = SatelliteEnv(state_builder, weights=reward_weights)
    agent = DQNAgent(env)

    pbar = tqdm(range(NUM_EPISODES), desc="DQN Training Progress")
    for episode in pbar:
        state = env.reset(mock_initial_packet())
        episode_reward = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            action_index = agent.select_action(state)
            neighbor_id_mock = f"SAT-LEO-{random.randint(10, 30)}"
            new_packet_data = mock_next_step_packet(env.current_packet_state)
            next_state, reward, done = env.step(action_index, neighbor_id_mock, new_packet_data)
            episode_reward += reward
            agent.memory.push(state, action_index, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            if done:
                break

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        pbar.set_postfix({'Reward': f"{episode_reward:.2f}", 'Steps': t+1})
        if (episode + 1) % 100 == 0:
            agent.save_checkpoint(CHECKPOINT_PATH.replace(".pth", f"_ep{episode+1}.pth"))

    agent.save_checkpoint(CHECKPOINT_PATH.replace(".pth", "_final.pth"))
    logger.info("--- HUẤN LUYỆN DQN HOÀN TẤT ---")

if __name__ == "__main__":
    train_agent()
