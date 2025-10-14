# main.py
import torch
import os
from typing import Dict, Any, Tuple

# Import toàn bộ logic đã xây dựng
from env.StateProcessor import StateProcessor
from env.ActionMapper import ActionMapper
from agents.DqnAgent import DqnAgent
from agents.InMemoryReplayBuffer import InMemoryReplayBuffer 
from simulator.NetworkSimulator import NetworkSimulator
from training.trainer import train_agent_batch
from scripts.scenario_manager import run_scenario 
from utils.static_data import BASE_NODE_INFO, calculate_link_score_simplified 
import random
import numpy as np

# --- 1. HÀM MÔ PHỎNG DIJKSTRA (Greedy) ---
def find_best_dijkstra_next_hop(raw_state_s: Dict[str, Any]) -> str:
    """Mô phỏng quyết định tham lam dựa trên LinkScore cao nhất."""
    neighbor_links = raw_state_s.get('neighborLinkMetrics', {})
    if not neighbor_links:
        return raw_state_s.get('sourceNodeId', 'NONE') 

    best_node_id = None
    max_score = -float('inf')
    
    for node_id, link in neighbor_links.items():
        score = link.get('linkScore', 0.0)
        if score > max_score:
            max_score = score
            best_node_id = node_id
            
    return best_node_id if best_node_id is not None else raw_state_s.get('sourceNodeId', 'NONE')

# --- 2. HÀM CHẠY SO SÁNH (INFERENCE LOOP) ---
def run_comparison_loop(agent: DqnAgent, simulator: NetworkSimulator, source_id: str, dest_id: str, num_queries: int, current_scenario_name: str):
    """Chạy vòng lặp giả định, so sánh quyết định của RL vs Dijkstra."""
    
    agent.epsilon = 0.0 
    drl_wins = 0
    dijkstra_wins = 0
    total_reward_drl = 0.0
    total_reward_dijkstra = 0.0

    print("\n" + "="*80)
    print(f"| BẮT ĐẦU SO SÁNH INFERENCE ({num_queries} QUERY) - KỊCH BẢN: {current_scenario_name} |")
    print("="*80 + "\n")
    
    for i in range(num_queries):
        raw_state_s: Dict[str, Any] = simulator._collect_current_state_data(source_id, dest_id)
        
        # 1. Quyết định của DRL
        state_vector_s = simulator.processor.json_to_state_vector(raw_state_s)
        drl_action_id = agent.select_action(state_vector_s)
        drl_link = raw_state_s['neighborLinkMetrics'].get(drl_action_id, {})
        drl_reward = simulator.reward_calc.calculate_reward(simulator.default_qos, drl_link)

        # 2. Quyết định của Dijkstra (Greedy)
        dijkstra_action_id = find_best_dijkstra_next_hop(raw_state_s)
        dijkstra_link = raw_state_s['neighborLinkMetrics'].get(dijkstra_action_id, {})
        dijkstra_reward = simulator.reward_calc.calculate_reward(simulator.default_qos, dijkstra_link)

        # 3. Log và Đếm
        total_reward_drl += drl_reward
        total_reward_dijkstra += dijkstra_reward
        
        if drl_reward > dijkstra_reward:
            drl_wins += 1
        elif dijkstra_reward > drl_reward:
            dijkstra_wins += 1

    # Log kết quả cuối cùng
    print("\n| KẾT QUẢ TỔNG HỢP:")
    print("-" * 35)
    print(f"| Tổng Reward DRL: {total_reward_drl:.2f}")
    print(f"| Tổng Reward Dijkstra: {total_reward_dijkstra:.2f}")
    print(f"| DRL Tốt hơn: {drl_wins} lần")
    print(f"| Dijkstra Tốt hơn: {dijkstra_wins} lần")
    
    if total_reward_drl > total_reward_dijkstra:
        print("\n🏆 KẾT LUẬN: DRL Agent đã vượt trội trong việc đáp ứng QoS.")
    else:
        print("\n⚠️ KẾT LUẬN: Dijkstra (Greedy) vẫn tốt hơn. Cần huấn luyện thêm!")
    print("="*80)


# --- CẤU HÌNH VÀ CHẠY CHÍNH ---

MAX_NEIGHBORS = 10 
STATE_SIZE = 6 + (4 * MAX_NEIGHBORS) 

if __name__ == "__main__":
    
    print("--- Khởi động DRL Agent ---")
    
    # 1. Khởi tạo Thành phần Cốt lõi
    processor = StateProcessor(max_neighbors=MAX_NEIGHBORS)
    mapper = ActionMapper() 
    mapper.sync_node_list()
    ACTION_SIZE = mapper.get_action_size()
    
    if ACTION_SIZE == 0:
        print("🛑 Lỗi: Không tìm thấy Node nào trong dữ liệu tĩnh.")
        exit()
        
    dqn_agent = DqnAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    dqn_agent.action_mapper = mapper
    buffer = InMemoryReplayBuffer()
    simulator = NetworkSimulator(dqn_agent, buffer) 

    # 2. Huấn luyện (Phase 1)
    TOTAL_TRAINING_STEPS_BASELINE = 10000 
    TOTAL_TRAINING_STEPS_FOCUSED = 10000 # Tăng lên 10k bước
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 200 
    SOURCE_NODE = "NodeA"
    DEST_NODE = "NodeD"
    
    # [BƯỚC 1/3] HUẤN LUYỆN BASELINE
    print("\n[BƯỚC 1/3] Bắt đầu Huấn luyện BASELINE (10k bước)...")
    run_scenario('BASELINE') 
    for step in range(1, TOTAL_TRAINING_STEPS_BASELINE + 1):
        simulator.simulate_one_step(source_id=SOURCE_NODE, dest_id=DEST_NODE)
        train_agent_batch(dqn_agent, buffer, BATCH_SIZE, TARGET_UPDATE_FREQ)
    print(f"HUẤN LUYỆN BASELINE HOÀN TẤT ({TOTAL_TRAINING_STEPS_BASELINE} bước).")

    # [BƯỚC 2/3] HUẤN LUYỆN CHUYÊN SÂU (Focused Training)
    print("\n[BƯỚC 2/3] Huấn luyện Chuyên sâu trên kịch bản TẮC NGHẼN để tối ưu hóa né tránh...")
    run_scenario('CONGESTION_LINK_AB') # Đặt dữ liệu tĩnh về trạng thái TẮC NGHẼN
    dqn_agent.epsilon = 0.8 # Tăng epsilon để khám phá lại các hành động rủi ro
    for step in range(1, TOTAL_TRAINING_STEPS_FOCUSED + 1):
        simulator.simulate_one_step(source_id=SOURCE_NODE, dest_id=DEST_NODE)
        train_agent_batch(dqn_agent, buffer, BATCH_SIZE, TARGET_UPDATE_FREQ)
    print(f"HUẤN LUYỆN CHUYÊN SÂU HOÀN TẤT ({TOTAL_TRAINING_STEPS_FOCUSED} bước).")
    dqn_agent.epsilon = 0.01 # Khôi phục epsilon thấp cho inference

    
    # 3. BẮT ĐẦU SO SÁNH HIỆU SUẤT TRONG CÁC KỊCH BẢN
    
    # --- S-1: MẠNG BÌNH THƯỜNG ---
    run_comparison_loop(dqn_agent, simulator, SOURCE_NODE, DEST_NODE, num_queries=200, current_scenario_name='BASELINE')

    # --- S-2: TẮC NGHẼN (Kiểm tra xem Agent có né tránh được không) ---
    run_scenario('CONGESTION_LINK_AB') 
    run_comparison_loop(dqn_agent, simulator, SOURCE_NODE, DEST_NODE, num_queries=200, current_scenario_name='CONGESTION_LINK_AB')
    
    # --- S-3: NODE QUÁ TẢI ---
    run_scenario('NODE_C_OVERLOAD') 
    run_comparison_loop(dqn_agent, simulator, SOURCE_NODE, DEST_NODE, num_queries=200, current_scenario_name='NODE_C_OVERLOAD')
    
    # --- S-4: MẤT LINK HOÀN TOÀN ---
    run_scenario('LINK_FAIL_AB') 
    run_comparison_loop(dqn_agent, simulator, SOURCE_NODE, DEST_NODE, num_queries=200, current_scenario_name='LINK_FAIL_AB')
    
    # --- S-5: SUY HAO CỰC ĐỘ (Môi trường) ---
    run_scenario('EXTREME_ATTENUATION_AC')
    run_comparison_loop(dqn_agent, simulator, SOURCE_NODE, DEST_NODE, num_queries=200, current_scenario_name='EXTREME_ATTENUATION_AC')