# client_app/data/mock_data.py
import random
from typing import Dict, Any, List
import time
def generate_mock_ack_result(packet_id: str, theory_delay: float, status: str) -> Dict[str, Any]:
    """Tạo một gói ACK giả lập hoàn chỉnh (dùng cho Load Test)."""
    
    if status == "SUCCESS":
        # Giả lập độ trễ RL
        rl_delay = theory_delay + random.uniform(-15.0, 50.0) 
        if rl_delay < 50:
             rl_delay = 50.0
             
        # Giả lập đường đi RL khác với lý thuyết
        path = ["G01", "A01", "G02"] if random.random() < 0.5 else ["G01", "S01", "A02", "G02"]

        return {
            "id": packet_id,
            "status": "SUCCESS",
            "rl_delay_ms": rl_delay,
            "theory_delay_ms": theory_delay,
            "path": path,
        }
    else:
        # Gói bị Drop hoặc Timeout
        return {
            "id": packet_id,
            "status": status,
            "rl_delay_ms": 9999.0,
            "theory_delay_ms": theory_delay,
            "path": [],
        }

def generate_mock_data_traffic(count: int) -> List[Dict[str, Any]]:
    """Tạo dữ liệu giả lập cho Incoming Traffic Monitor."""
    mock_traffic = []
    service_types = ["VIDEO_STREAMING", "FILE_TRANSFER", "BASIC_DATA"]
    sources = ["A01", "S01", "NodeX"]
    
    for i in range(count):
        # Đảm bảo sử dụng camelCase keys để khớp với Packet model
        mock_traffic.append({
            'packetId': f'DATA-MOCK-{i}-{int(time.time() * 1000)}', 
            'sourceUserId': random.choice(sources), 
            'destinationUserId': 'G01',
            'payloadSizeByte': random.randint(500, 1500),
            'serviceType': random.choice(service_types)
        })
    return mock_traffic