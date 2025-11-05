# python/main_train.py

import logging
import random
import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# === Imports từ các module của bạn ===
from python.utils.db_connector import MongoConnector
from python.utils.state_builder import StateBuilder
from python.env.satellite_simulator import SatelliteEnv
from python.rl_agent.trainer import DQNAgent
from python.rl_agent.policy import get_epsilon

# --- CẤU HÌNH VÀ HẰNG SỐ ---
# (NOTE) Tăng số episode để agent có thời gian học
NUM_EPISODES = 20000
MAX_HOPS_PER_EPISODE = 50
SAVE_INTERVAL = 500

CHECKPOINT_BASE_PATH = "models/checkpoints/dqn_checkpoint_fullpath"
RESUME_FILE_PATH = f"{CHECKPOINT_BASE_PATH}_latest.pth"
CHART_SAVE_PATH = "training_charts.png"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------- MOCK PACKET GENERATOR -----------------
def generate_packet(node_list: List[str]) -> Dict[str, Any]:
    """Tạo packet ngẫu nhiên từ 1 node đến 1 destination khác."""
    if len(node_list) < 2:
        raise ValueError("Không đủ node để tạo packet.")

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


# ----------------- TRAINING LOOP -----------------
def train_agent():
    logger.info("=== KHỞI TẠO HỆ THỐNG DQN ROUTER ===")
    mongo_conn = MongoConnector()
    state_builder = StateBuilder(mongo_conn)
    env = SatelliteEnv(state_builder)
    agent = DQNAgent(env)

    all_nodes_data = state_builder.db.get_all_nodes(projection={"nodeId": 1})
    all_nodes = [n["nodeId"] for n in all_nodes_data]

    total_possible_pairs = 0
    if len(all_nodes) >= 2:
        total_possible_pairs = len(all_nodes) * (len(all_nodes) - 1)
    else:
        logger.error("Không đủ Node để huấn luyện. Kiểm tra MongoDB.")
        return

    logger.info(f"Đang huấn luyện với {len(all_nodes)} nodes ({total_possible_pairs} cặp src-dest khả thi)...")

    # --- Thống kê ---
    rewards = []
    avg_rewards = []
    trained_pairs = set()
    coverage_percent_history = [] 

    # --- (NOTE) LOGIC RESUME TRAINING ---
    start_episode = 0
    if os.path.exists(RESUME_FILE_PATH):
        try:
            logger.info(f"Phát hiện checkpoint. Đang tải từ: {RESUME_FILE_PATH}")
            
            # (SỬA) Thêm `weights_only=False` để cho phép tải file
            # checkpoint chứa dữ liệu (pickle) không phải trọng số.
            checkpoint = torch.load(
                RESUME_FILE_PATH, 
                map_location=torch.device('cpu'),
                weights_only=False 
            )
            
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.steps_done = checkpoint['steps_done']
            start_episode = checkpoint['episode'] + 1
            
            # Khôi phục lịch sử thống kê
            rewards = checkpoint.get('rewards_history', [])
            avg_rewards = checkpoint.get('avg_rewards_history', [])
            trained_pairs = checkpoint.get('trained_pairs_set', set())
            coverage_percent_history = checkpoint.get('coverage_history', [])
            
            logger.info(f"Tải thành công. Huấn luyện tiếp từ Episode {start_episode}")
        except Exception as e:
            logger.error(f"Lỗi khi tải checkpoint: {e}. Bắt đầu lại từ đầu.")
            start_episode = 0
            rewards, avg_rewards, trained_pairs, coverage_percent_history = [], [], set(), [] # Reset
    else:
        logger.info("Không tìm thấy checkpoint. Bắt đầu huấn luyện mới.")


    pbar = tqdm(range(start_episode, NUM_EPISODES), desc="DQN Training", initial=start_episode, total=NUM_EPISODES)

    try: # (NOTE) Thêm try...except để bắt lỗi runtime
        for episode in pbar:
            packet = generate_packet(all_nodes)
            trained_pairs.add((packet["currentHoldingNodeId"], packet["stationDest"]))

            episode_reward = env.simulate_episode(agent, packet, max_hops=MAX_HOPS_PER_EPISODE)

            # --- Cập nhật thống kê ---
            rewards.append(episode_reward)
            current_coverage_pct = (len(trained_pairs) / total_possible_pairs) * 100
            coverage_percent_history.append(current_coverage_pct)

            current_avg_50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards) if rewards else 0
            if len(rewards) >= 50:
                avg_rewards.append(current_avg_50)

            epsilon = get_epsilon(agent.steps_done)
            pbar.set_postfix({
                'Reward': f"{episode_reward:.2f}",
                'Avg50': f"{current_avg_50:.2f}",
                'Steps': agent.steps_done,
                'Eps': f"{epsilon:.3f}",
                'Coverage': f"{current_coverage_pct:.2f}%"
            })

            # --- LOGIC LƯU CHECKPOINT ---
            if (episode + 1) % SAVE_INTERVAL == 0 or (episode + 1) == NUM_EPISODES:
                checkpoint_data = {
                    'episode': episode,
                    'model_state_dict': agent.q_network.state_dict(),
                    'target_network_state_dict': agent.target_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'steps_done': agent.steps_done,
                    'rewards_history': rewards,
                    'avg_rewards_history': avg_rewards,
                    'trained_pairs_set': trained_pairs,
                    'coverage_history': coverage_percent_history
                }
                
                if (episode + 1) % SAVE_INTERVAL == 0:
                    save_path_milestone = f"{CHECKPOINT_BASE_PATH}_ep{episode+1}.pth"
                    torch.save(checkpoint_data, save_path_milestone)
                    logger.info(f"Đã lưu checkpoint mốc tại {save_path_milestone}")
                
                torch.save(checkpoint_data, RESUME_FILE_PATH)
                if (episode + 1) == NUM_EPISODES:
                    logger.info("=== HUẤN LUYỆN HOÀN TẤT ===")
                    final_path = f"{CHECKPOINT_BASE_PATH}_final.pth"
                    torch.save(checkpoint_data, final_path)
                    logger.info(f"Model cuối cùng: {final_path}")

    except KeyboardInterrupt:
        logger.warning("\nPhát hiện (Ctrl+C). Đang dừng và lưu checkpoint...")
        # (NOTE) Vẫn lưu checkpoint cuối cùng khi bị ngắt
        checkpoint_data = {
            'episode': episode, # Lưu episode hiện tại
            'model_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'steps_done': agent.steps_done,
            'rewards_history': rewards,
            'avg_rewards_history': avg_rewards,
            'trained_pairs_set': trained_pairs,
            'coverage_history': coverage_percent_history
        }
        torch.save(checkpoint_data, RESUME_FILE_PATH)
        logger.info(f"Đã lưu tiến trình resume tại {RESUME_FILE_PATH}. Chạy lại để tiếp tục.")
    
    except Exception as e:
        # (NOTE) Đây là nơi sẽ bắt lỗi runtime (lỗi 198)
        logger.error(f"LỖI NGHIÊM TRỌNG ở episode {episode}: {e}", exc_info=True)
        # Vẫn cố gắng lưu checkpoint
        checkpoint_data = {
            'episode': episode,
            'model_state_dict': agent.q_network.state_dict(),
            # ... (lưu tương tự như trên)
        }
        torch.save(checkpoint_data, f"{CHECKPOINT_BASE_PATH}_CRASH.pth")
        logger.info(f"Đã lưu checkpoint CRASH tại {CHECKPOINT_BASE_PATH}_CRASH.pth")
        # Ném lỗi ra ngoài
        raise e

    
    # --- In thống kê cuối ---
    final_coverage_pct = (len(trained_pairs) / total_possible_pairs) * 100
    logger.info(f"Tổng số cặp (src,dest) đã train: {len(trained_pairs)} (coverage ≈ {final_coverage_pct:.2f}%)")

    # --- VẼ BIỂU ĐỒ ---
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Kết quả Huấn luyện DQN', fontsize=16)

        ax1.plot(rewards, label='Reward mỗi episode', alpha=0.3)
        if avg_rewards:
            # (SỬA) Tính toán trục X cho avg_rewards
            avg_reward_x_axis = np.linspace(50, len(rewards), len(avg_rewards))
            ax1.plot(avg_reward_x_axis, avg_rewards, label='Reward TB (50 ep)', linewidth=2, color='orange')
        ax1.set_ylabel('Tổng Reward')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Biểu đồ Reward')

        ax2.plot(coverage_percent_history, label='Độ bao phủ (Src-Dest)', color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Độ bao phủ (%)')
        ax2.set_ylim(0, 100) 
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Biểu đồ Độ bao phủ (Src-Dest)')
        
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(CHART_SAVE_PATH)
        logger.info(f"Đã lưu biểu đồ (Reward & Coverage) tại {CHART_SAVE_PATH}")
    except Exception as e:
        logger.warning(f"Không thể vẽ biểu đồ: {e}")


if __name__ == "__main__":
    train_agent()