# client_app/engine/simulation_runner.py
import concurrent.futures
import time
from typing import Dict, List, Any
import random

# IMPORT CÁC MODULE CẦN THIẾT
# Use absolute imports so modules work when the package root is added to sys.path
from utils.json_serializer import create_data_packet_instance
from .shared_ack_listener import SharedACKDataListener 
# Sử dụng import tuyệt đối cho data (avoids "relative import beyond top-level package")
from data.mock_data import generate_mock_ack_result 

# Hàm xử lý một yêu cầu (Chế độ MOCKING)
def run_single_simulation_job(request_index: int, config: Dict[str, Any], listener: SharedACKDataListener) -> Dict[str, Any]:
    
    packet_id = f"REQ-{request_index}-{int(time.time() * 1000)}"
    
    # 1. Giả lập một chút độ trễ cho mỗi luồng (mô phỏng I/O)
    time.sleep(random.uniform(0.01, 0.1)) 
    
    # 2. Quyết định trạng thái (Mocking)
    rand_val = random.random()
    if rand_val < 0.7:
        status = "SUCCESS"
    elif rand_val < 0.9:
        status = "TIMEOUT"
    else:
        status = "DROPPED"
        
    # 3. Tạo và trả về kết quả giả lập
    mock_result = generate_mock_ack_result(packet_id, config['theory_delay'], status)
    
    # Điều chỉnh RTT trong Mock (được tính toán trong hàm giả lập)
    mock_result['rtt_ms'] = mock_result['rl_delay_ms'] * 2
    
    return mock_result
    
def run_load_test(request_count: int, max_workers: int, sim_config: Dict[str, Any], listener: SharedACKDataListener) -> List[Dict[str, Any]]:
    """Chạy Load Test bằng Thread Pool Executor (chế độ Mocking)."""
    
    results = []
    # Sử dụng Thread Pool Executor để chạy các jobs song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_simulation_job, i, sim_config, listener) for i in range(request_count)]
        
        # Thu thập kết quả
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
                
    return results