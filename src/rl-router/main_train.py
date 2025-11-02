# python/main_train.py

import logging
import random
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, List, Tuple

# Imports từ các module đã hoàn thiện
from python.utils.db_connector import MongoConnector
from python.utils.state_builder import StateBuilder
from python.env.satellite_simulator import SatelliteEnv
from python.rl_agent.trainer import DQNAgent, TARGET_UPDATE_INTERVAL
from python.rl_agent.policy import get_epsilon # Cần cho logging

# --- CẤU HÌNH VÀ HẰNG SỐ ---
NUM_EPISODES = 1000
MAX_HOPS_PER_EPISODE = 50 # Giới hạn vòng lặp mô phỏng
CHECKPOINT_PATH = "models/checkpoints/dqn_checkpoint_fullpath.pth"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------- MOCK PACKET GENERATOR -----------------

def generate_packet(node_list: List[str]) -> Dict[str, Any]:
    """Tạo packet từ 1 node ngẫu nhiên đến 1 destination khác."""
    
    if len(node_list) < 2:
        raise ValueError("Cannot generate packet: Need at least two nodes.")
        
    src = random.choice(node_list)
    dest = random.choice([n for n in node_list if n != src])
    
    packet = {
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
        "dropped": False,
        "path": [src]
    }
    return packet

def simulate_full_path(
    env: SatelliteEnv,
    agent: DQNAgent,
    state_builder: StateBuilder,
    packet: Dict[str, Any],
    max_hops: int = MAX_HOPS_PER_EPISODE
) -> Tuple[List[Tuple], float]:
    """
    Mô phỏng hành trình của 1 packet qua nhiều hop đến đích.
    Trả về (transitions, total_reward)
    """
    state = env.reset(packet)
    transitions = []
    total_reward = 0.0
    hops = 0

    current_packet = packet.copy()

    while True:
        current_node_id = current_packet["currentHoldingNodeId"]
        dest_node_id = current_packet["stationDest"]

        # ---- Điều kiện kết thúc ----
        if (
            current_packet.get("dropped")
            or current_packet.get("ttl", 0) <= 0
            or current_node_id == dest_node_id
            or hops >= max_hops
        ):
            done = True
            # Phạt nhẹ nếu TTL cạn hoặc bị drop
            if current_packet.get("dropped", False):
                total_reward += -150.0
            elif current_packet.get("ttl", 0) <= 0:
                total_reward += -50.0
            elif current_node_id == dest_node_id:
                total_reward += 200.0  # Thưởng đến đích
            break

        # ---- Lấy neighbors hiện tại ----
        node_data = state_builder.db.get_node(current_node_id, projection={"neighbors": 1})
        neighbor_ids = node_data.get("neighbors", []) if node_data else []

        if not neighbor_ids:
            current_packet["dropped"] = True
            continue

        # ---- Agent chọn hành động ----
        action_index = agent.select_action(state)

        # Xử lý nếu action_index vượt quá số neighbor thực tế
        if action_index < len(neighbor_ids):
            next_hop_id = neighbor_ids[action_index]
        else:
            next_hop_id = random.choice(neighbor_ids)

        # ---- Mô phỏng chuyển tiếp ----
        next_packet = current_packet.copy()
        next_packet["currentHoldingNodeId"] = next_hop_id
        next_packet["ttl"] = max(current_packet.get("ttl", 10) - 1, 0)
        next_packet["accumulatedDelayMs"] += random.uniform(5.0, 20.0)
        next_packet["path"] = current_packet["path"] + [next_hop_id]

        # ---- Step trong môi trường ----
        next_state, reward, done = env.step(action_index, next_hop_id, next_packet)

        # ---- Ghi nhận ----
        total_reward += reward
        transitions.append((state, action_index, reward, next_state, done))

        # ---- Cập nhật cho vòng tiếp theo ----
        state = next_state
        current_packet = next_packet
        hops += 1

        if done:
            break

    # ---- Logging chi tiết ----
    logger.info(
        f"[Episode Path] {packet['path'][0]} → {current_packet['currentHoldingNodeId']} "
        f"| TotalReward={total_reward:.2f} | Hops={hops} | TTL={current_packet.get('ttl',0)}"
    )

    return transitions, total_reward

# ----------------- TRAINING LOOP -----------------

def train_agent():
    logger.info("=== KHỞI TẠO HỆ THỐNG DQN ROUTER FULLPATH ===")
    mongo_conn = MongoConnector(uri="mongodb://user:password123@localhost:27017/sagsin_network?authSource=admin")
    state_builder = StateBuilder(mongo_conn)

    # Weights cho Reward (Đã thêm hop_cost để giải quyết lỗi lang thang)
    reward_weights = {
        'goal': 200.0,
        'drop': -150.0,
        'latency': -10.0,
        'latency_violation': -50.0,
        'utilization': 2.0,
        'bandwidth': 1.0,
        'reliability': 3.0,
        'fspl': -0.1,
        'hop_cost': -1.0 # 💡 PHẠT MỚI
    }
    env = SatelliteEnv(state_builder, weights=reward_weights)
    agent = DQNAgent(env)
    
    # SỬA LỖI: Lấy tất cả Node ID cho generator
    all_nodes_data = state_builder.db.get_all_nodes(projection={"nodeId": 1})
    all_nodes = [n["nodeId"] for n in all_nodes_data]

    if len(all_nodes) < 2:
        logger.error("Không đủ Node để huấn luyện. Vui lòng kiểm tra MongoDB.")
        return

    pbar = tqdm(range(NUM_EPISODES), desc="DQN Fullpath Training")
    for episode in pbar:
        packet = generate_packet(all_nodes)
        
        # Simulate full path and get all transitions
        transitions, episode_reward = simulate_full_path(env, agent, state_builder, packet)

        # Lưu transitions vào replay buffer và tối ưu hóa
        for s, a, r, s_next, done in transitions:
            if a is not None:
                agent.memory.push(s, a, r, s_next, done)
                agent.optimize_model()

        # Cập nhật Target Network
        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_network()

        # Logging và Checkpoint
        epsilon = get_epsilon(agent.steps_done)
        pbar.set_postfix({'Reward': f"{episode_reward:.2f}", 'Hops': len(transitions), 'Epsilon': f"{epsilon:.4f}"})

        if (episode + 1) % 100 == 0:
            agent.save_checkpoint(CHECKPOINT_PATH.replace(".pth", f"_ep{episode+1}.pth"))

    agent.save_checkpoint(CHECKPOINT_PATH.replace(".pth", "_final.pth"))
    logger.info("=== HUẤN LUYỆN DQN FULLPATH HOÀN TẤT ===")

if __name__ == "__main__":
    train_agent()