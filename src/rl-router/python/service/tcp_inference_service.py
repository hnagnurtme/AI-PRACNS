# python/service/tcp_inference_service.py

import socket
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple

# Imports từ các module RL đã xây dựng
from ..utils.db_connector import MongoConnector
from ..utils.state_builder import StateBuilder, MAX_NEIGHBORS # (NOTE) Import MAX_NEIGHBORS (10)
# (NOTE) Import kiến trúc 94/10
from ..rl_agent.dqn_model import DQN, INPUT_SIZE, OUTPUT_SIZE 
# (NOTE) Xóa import NodeService, không cần thiết

# ===================== CẤU HÌNH DỊCH VỤ =====================
HOST = '0.0.0.0'
PORT = 65000
MODEL_PATH = "models/checkpoints/dqn_checkpoint_fullpath_latest.pth"

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== 1. LOAD COMPONENTS =====================
def load_components() -> Tuple[DQN, StateBuilder]:
    """Tải mô hình DQN, kết nối MongoDB, khởi tạo StateBuilder."""
    logger.info("1. Đang tải mô hình DQN...")

    model = DQN(INPUT_SIZE, OUTPUT_SIZE)

    try:
        # (PYTORCH ≥ 2.6) Thêm weights_only=False nếu có
        try:
            checkpoint = torch.load(MODEL_PATH, weights_only=False)
        except TypeError:
            checkpoint = torch.load(MODEL_PATH)

        # 🧠 Kiểm tra loại checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("✅ Đã load model_state_dict từ checkpoint.")
        else:
            # File chỉ chứa state_dict thuần
            model.load_state_dict(checkpoint)
            logger.info("✅ Đã load state_dict trực tiếp.")

    except FileNotFoundError:
        logger.error(f"❌ Không tìm thấy mô hình tại {MODEL_PATH}.")
        raise
    except RuntimeError as e:
        logger.error(f"❌ Kiến trúc model không khớp: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi khi load checkpoint: {e}")
        raise

    model.eval()

    logger.info("2. Kết nối MongoDB...")
    mongo_conn = MongoConnector()
    state_builder = StateBuilder(mongo_conn)

    logger.info(f"✅ DQN Agent (94-In, 10-Out) đã sẵn sàng.")
    return model, state_builder


# ===================== 2. PATH PREDICTION =====================
def get_optimal_path(model: DQN, state_builder: StateBuilder, packet_data: Dict[str, Any],
                     max_hops: int = 50) -> Tuple[List[str], str]:
    """
    (TỐI ƯU) Dự đoán toàn bộ path dựa trên DQN,
    khớp policy và chống lặp vòng hiệu quả.
    """
    start_node = packet_data.get('currentHoldingNodeId')
    if not start_node:
        return ["ERROR_START_NODE"], "ERROR:missing_start_node"

    path: List[str] = [str(start_node)]
    current_packet = packet_data.copy()
    current_packet['path'] = path
    hops = 0
    visited_nodes = set([start_node]) # (TỐI ƯU) Dùng set để chống lặp toàn path

    while hops < max_hops:
        current_node_id = current_packet.get('currentHoldingNodeId')
        dest_node_id = current_packet.get('stationDest')

        if current_node_id == dest_node_id:
            return path, "SUCCESS"

        if current_packet.get('dropped', False) or current_packet.get('ttl', 0) <= 0:
            return path, "DROP_TTL"

        try:
            # 1. Lấy trạng thái vector S (94 chiều)
            state_vector = state_builder.get_state_vector(current_packet)
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)

            # 2. Lấy neighbor thực tế từ DB
            # (NOTE) Đây là danh sách neighbor mà StateBuilder đã dùng
            current_node = state_builder.db.get_node(str(current_node_id), projection={'neighbors': 1})
            neighbor_ids_full = current_node.get('neighbors', []) if current_node else []
            
            # (NOTE) Chỉ lấy tối đa 10 neighbor đầu tiên để khớp policy
            neighbor_ids = neighbor_ids_full[:MAX_NEIGHBORS]
            neighbor_count = len(neighbor_ids)

            if not neighbor_ids:
                return path, "NO_NEIGHBORS"

            # (TỐI ƯU) Xóa bỏ heuristic top-k
            # Agent phải quyết định dựa trên state vector đầy đủ.

            # 3. DQN inference
            with torch.no_grad():
                q_values_tensor = model(state_tensor) # Trả về 10 Q-values
            q_values = q_values_tensor.cpu().numpy().flatten()

            # (TỐI ƯU) Masking Q-values
            # Chỉ quan tâm đến Q-values của các neighbor thực sự tồn tại
            # Gán Q-value của các hành động không hợp lệ (vd: hành động 8, 9
            # khi chỉ có 7 neighbor) là -vô cùng
            valid_q_values = np.full(OUTPUT_SIZE, -np.inf, dtype=np.float32)
            valid_q_values[:neighbor_count] = q_values[:neighbor_count]

            # 4. (TỐI ƯU) Loop prevention
            last_node_id = path[-2] if len(path) >= 2 else None
            
            for _ in range(neighbor_count):
                # Chọn hành động tốt nhất từ Q-values HỢP LỆ
                action_index = int(np.argmax(valid_q_values))
                next_hop = neighbor_ids[action_index]
                
                # Kiểm tra lặp vòng
                if next_hop != last_node_id and next_hop not in visited_nodes:
                    break # Hành động tốt, không lặp
                
                # Bị lặp: Loại bỏ hành động này và chọn lại
                valid_q_values[action_index] = -np.inf # Phạt
            else:
                # (TỐI ƯU) Nếu tất cả 10 hành động đều bị lặp
                return path, "LOOP_STUCK"

            # 5. Cập nhật path và packet state
            path.append(next_hop)
            visited_nodes.add(next_hop) # Thêm vào danh sách đã thăm
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
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f"🚀 RL Router (94/10) listening on {HOST}:{PORT} (TCP)")

        while True:
            conn, addr = s.accept()
            # (TỐI ƯU) Sử dụng makefile để xử lý stream (nhiều req/kết nối)
            with conn, conn.makefile('rb') as rfile, conn.makefile('wb') as wfile:
                logger.info(f"Connection established with {addr}")
                try:
                    # Vòng lặp xử lý nhiều JSON trên 1 kết nối
                    while True:
                        line = rfile.readline() # Đọc đến khi gặp '\n'
                        if not line:
                            break # Client đóng kết nối

                        request_json = json.loads(line.strip().decode('utf-8'))
                        
                        # Lấy path tối ưu
                        path_list, status = get_optimal_path(model, state_builder, request_json, max_hops=20)
                        
                        # Trả về hop tiếp theo
                        next_hop_id = path_list[1] if len(path_list) > 1 else path_list[0]

                        response_obj = {
                            "nextHopNodeId": next_hop_id,
                            "path": path_list,
                            "algorithm": "RL-DQN-10N", # (NOTE) Tên mới
                            "status": status
                        }

                        response_bytes = (json.dumps(response_obj) + "\n").encode('utf-8')
                        wfile.write(response_bytes)
                        wfile.flush() # Đẩy dữ liệu đi ngay

                except json.JSONDecodeError:
                    error_msg = json.dumps({"status": "ERROR", "message": "Invalid JSON format."}) + "\n"
                    wfile.write(error_msg.encode('utf-8'))
                    wfile.flush()
                except Exception as e:
                    logger.error(f"Error processing request from {addr}: {e}", exc_info=True)
                
                logger.info(f"Connection closed with {addr}")

if __name__ == "__main__":
    start_tcp_server()