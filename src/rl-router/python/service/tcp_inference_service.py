# python/service/tcp_inference_service.py

import socket
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple

# Imports từ các module RL đã xây dựng
from ..utils.db_connector import MongoConnector
from ..utils.state_builder import StateBuilder
from ..rl_agent.dqn_model import DQN, INPUT_SIZE, OUTPUT_SIZE
from helper.neighbors_update import NodeService

# ===================== CẤU HÌNH DỊCH VỤ =====================
HOST = '0.0.0.0'
PORT = 65000
MODEL_PATH = "models/checkpoints/dqn_checkpoint_fullpath_final.pth"
BUFFER_TIMEOUT = 1.0  # giây

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== 1. LOAD COMPONENTS =====================
def load_components() -> Tuple[DQN, StateBuilder]:
    """Tải mô hình DQN, kết nối MongoDB, khởi tạo StateBuilder."""
    logger.info("1. Đang tải mô hình DQN...")
    model = DQN(INPUT_SIZE, OUTPUT_SIZE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        logger.error(f"❌ LỖI: Không tìm thấy mô hình tại {MODEL_PATH}. Vui lòng chạy main_train.py trước.")
        raise
    model.eval()
    
    logger.info("2. Đang kết nối MongoDB...")
    mongo_conn = MongoConnector(uri="mongodb://user:password123@localhost:27017/?authSource=admin")
    state_builder = StateBuilder(mongo_conn)
    
    logger.info("✅ DQN Agent đã sẵn sàng.")
    return model, state_builder

# ===================== 2. PATH PREDICTION =====================
def get_optimal_path(model: DQN, state_builder: StateBuilder, packet_data: Dict[str, Any],
                     max_hops: int = 50, top_k_neighbors: int = 5) -> Tuple[List[str], str]:
    """
    Dự đoán toàn bộ path từ currentHoldingNodeId đến đích dựa trên DQN.
    Sửa để tránh loop, treo, TTL=0 quá nhiều.
    :param max_hops: giới hạn vòng lặp để tránh infinite loop
    :param top_k_neighbors: số neighbor gần nhất để RL lựa chọn
    :return: (path_list, status)
    """
    start_node = packet_data.get('currentHoldingNodeId')
    if not start_node:
        return ["ERROR_START_NODE"], "ERROR:missing_start_node"

    path: List[str] = [str(start_node)]
    current_packet = packet_data.copy()
    current_packet['path'] = path
    hops = 0
    visited_nodes = set([start_node])

    while hops < max_hops:
        current_node_id = current_packet.get('currentHoldingNodeId')
        dest_node_id = current_packet.get('stationDest')

        if current_node_id == dest_node_id:
            return path, "SUCCESS"

        if current_packet.get('dropped', False) or current_packet.get('ttl', 0) <= 0:
            return path, "DROP_TTL"

        try:
            print('Dang xu li ')
            # 1. Lấy trạng thái vector S
            state_vector = state_builder.get_state_vector(current_packet)
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)

            # 2. Lấy neighbor thực tế từ DB
            current_node = state_builder.db.get_node(str(current_node_id), projection={'neighbors': 1, 'position':1})
            neighbor_ids = current_node.get('neighbors', []) if current_node else []

            if not neighbor_ids:
                return path, "NO_NEIGHBORS"

            # --- Heuristic: lọc top-k neighbor gần nhất đến dest ---
            neighbor_distances = []
            if not dest_node_id:
                return path, "ERROR:missing_dest_node"
            dest_node = state_builder.db.get_node(str(dest_node_id), projection={"position":1})
            if dest_node:
                dest_pos = NodeService.geo_to_xyz(dest_node)
                for n_id in neighbor_ids:
                    n_node = state_builder.db.get_node(n_id, projection={"position":1})
                    if not n_node:
                        continue
                    n_pos = NodeService.geo_to_xyz(n_node)
                    dist = NodeService.distance_3d(n_pos, dest_pos)
                    neighbor_distances.append((n_id, dist))
                neighbor_ids = [nid for nid, _ in sorted(neighbor_distances, key=lambda x: x[1])[:top_k_neighbors]]

            neighbor_count = len(neighbor_ids)

            # 3. DQN inference
            with torch.no_grad():
                q_values_tensor = model(state_tensor)
            q_values = q_values_tensor.cpu().numpy().flatten()
            valid_q_values = q_values[:neighbor_count]
            action_index = int(np.argmax(valid_q_values))
            next_hop = neighbor_ids[action_index]

            # 4. Loop prevention
            attempted_indices = set()
            last_node_id = path[-2] if len(path) >= 2 else None
            for _ in range(neighbor_count):
                if next_hop != last_node_id and next_hop not in visited_nodes:
                    break
                attempted_indices.add(action_index)
                valid_q_values[action_index] = -1e9
                action_index = int(np.argmax(valid_q_values))
                next_hop = neighbor_ids[action_index]
            else:
                # fallback: random neighbor hợp lệ
                possible_neighbors = [n for n in neighbor_ids if n != last_node_id and n not in visited_nodes]
                if possible_neighbors:
                    next_hop = np.random.choice(possible_neighbors)
                else:
                    return path, "LOOP_STUCK"

            # 5. Cập nhật path và packet state
            path.append(next_hop)
            visited_nodes.add(next_hop)
            current_packet['currentHoldingNodeId'] = next_hop
            current_packet['path'] = path
            current_packet['ttl'] = max(current_packet.get('ttl', 10) - 1, 0)
            hops += 1

        except Exception as e:
            logger.error(f"Error in path prediction: {e}", exc_info=True)
            return path, f"ERROR:{e}"

    return path, "MAX_HOPS_EXCEEDED"

# ===================== 3. TCP SERVER =====================
def start_tcp_server():
    model, state_builder = load_components()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        logger.info(f"Binding to {HOST}:{PORT}...")
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f"🚀 RL Router listening on {HOST}:{PORT} (TCP)")

        while True:
            conn, addr = s.accept()
            conn.settimeout(BUFFER_TIMEOUT)
            with conn:
                logger.info(f"Connection established with {addr}")
                data_buffer = b''

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            break
                        data_buffer += chunk
                        if b'\n' in data_buffer:
                            break

                    if not data_buffer:
                        continue

                    request_json = json.loads(data_buffer.strip().decode('utf-8'))
                    
                    print(request_json)

                    # Lấy path tối ưu
                    path_list, status = get_optimal_path(model, state_builder, request_json, max_hops=20)

                    # nextHopNodeId = node thứ 2 nếu có, ngược lại chính node hiện tại
                    next_hop_id = path_list[1] if len(path_list) > 1 else path_list[0]

                    response_obj = {
                        "nextHopNodeId": next_hop_id,
                        "path": path_list,
                        "algorithm": "RL-DQN",
                        "status": status
                    }

                    response = (json.dumps(response_obj) + "\n").encode('utf-8')
                    conn.sendall(response)

                except socket.timeout:
                    logger.warning(f"Connection timed out with {addr}. Client did not send full request.")
                except json.JSONDecodeError:
                    error_msg = "Invalid JSON format."
                    conn.sendall((json.dumps({"status": "ERROR", "message": error_msg}) + "\n").encode('utf-8'))
                except Exception as e:
                    logger.error(f"Unexpected error processing request from {addr}: {e}", exc_info=True)

if __name__ == "__main__":
    start_tcp_server()
