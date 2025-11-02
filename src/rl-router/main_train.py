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
from python.rl_agent.trainer import DQNAgent, TARGET_UPDATE
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
        "ttl": random.randint(15, 25),
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

def simulate_full_path(env: SatelliteEnv, agent: DQNAgent, state_builder: StateBuilder, packet: Dict[str, Any]) -> Tuple[List[Tuple], float]:
    """
    Mô phỏng hành trình packet đầy đủ (nhiều hop) cho một Episode.
    Trả về danh sách transitions và tổng reward.
    """
    state = env.reset(packet)
    transitions = []
    total_reward = 0
    hops = 0
    current_packet = packet.copy()

    while hops < MAX_HOPS_PER_EPISODE:
        current_node_id = current_packet["currentHoldingNodeId"]
        
        # Lấy thông tin neighbors cho Agent (dù Agent dùng Q-Network) và cho mô phỏng
        current_node_data = state_builder.db.get_node(current_node_id, projection={"neighbors": 1})
        neighbor_ids = current_node_data.get("neighbors", []) if current_node_data else []

        # Kiểm tra điều kiện kết thúc sớm
        if current_packet.get("dropped") or current_packet.get("ttl", 0) <= 0 or current_node_id == current_packet.get("stationDest"):
            # Lưu transition cuối (nếu có) và kết thúc
            if hops > 0:
                 transitions.append((state, None, 0.0, state, True)) # Transition cuối giả định
            break

        # 1. Chọn Hành động (Action Selection)
        action_index = agent.select_action(state)
        
        # Xử lý trường hợp không có neighbors
        if not neighbor_ids:
            current_packet["dropped"] = True
            continue # Lặp lại và sẽ bị bắt ở kiểm tra điều kiện kết thúc

        # 2. Ánh xạ Hành động sang Node ID
        if action_index < len(neighbor_ids):
            next_hop_id = neighbor_ids[action_index]
        else:
            # Nếu Agent chọn index ngoài phạm vi (padding), chọn ngẫu nhiên neighbor hợp lệ
            next_hop_id = random.choice(neighbor_ids) 
            
        # 3. Cập nhật packet mô phỏng hop tiếp theo (Môi trường)
        next_packet = current_packet.copy()
        next_packet["currentHoldingNodeId"] = next_hop_id
        next_packet["ttl"] = max(current_packet.get("ttl", 10) - 1, 0)
        
        # Giả lập tăng độ trễ (sử dụng random)
        next_packet["accumulatedDelayMs"] += random.uniform(5.0, 20.0) 
        next_packet["path"] = current_packet["path"] + [next_hop_id]

        # 4. Step trong môi trường
        # Hàm env.step sẽ tính Reward dựa trên trạng thái cũ (current_packet) và trạng thái mới (next_packet)
        next_state, reward, done = env.step(action_index, next_hop_id, next_packet)
        total_reward += reward
        
        # 5. Lưu trữ Transition
        transitions.append((state, action_index, reward, next_state, done))

        current_packet = next_packet
        state = next_state
        hops += 1

        if done:
            break

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
        if episode % TARGET_UPDATE == 0:
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