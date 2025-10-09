# utils/mongo_connector.py
import os
from pymongo import MongoClient
from pymongo.errors import ConfigurationError
from typing import Dict, Any, List, Optional
import time

class MongoConnector:
    """Quản lý kết nối MongoDB và các thao tác cơ sở dữ liệu."""
    
    def __init__(self, host: str = "mongodb", port: int = 27017, db_name: str = "sagin_network"):
        # Lấy host từ biến môi trường (Ưu tiên host bên ngoài/Colab nếu có)
        mongo_host = os.getenv("MONGO_HOST_EXTERNAL", host)
        
        try:
            self.client = MongoClient(mongo_host, port, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping') 
            print(f"✅ Kết nối MongoDB tại {mongo_host} thành công.")
            
            self.db = self.client[db_name]
            self.node_info = self.db['NodeInfo']
            self.link_metrics = self.db['LinkMetrics']
            self.replay_buffer = self.db['ExperienceReplayBuffer']
            
        except ConnectionError as e:
            print(f"❌ LỖI: Không thể kết nối MongoDB tại {mongo_host}:{port}. {e}")
            raise

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.node_info.find_one({"nodeId": node_id})
    
    def get_link_metrics(self, source_id: str) -> List[Dict[str, Any]]:
        # Trả về các link đang hoạt động xuất phát từ node nguồn
        return list(self.link_metrics.find({"sourceNodeId": source_id, "isLinkActive": True}))
    
    def get_all_node_ids(self) -> List[str]:
        # Dùng cho ActionMapper
        return [doc["nodeId"] for doc in self.node_info.find({}, {"nodeId": 1})]
    
    def close(self):
        """Đóng kết nối."""
        self.client.close()

# --- HÀM KHỞI TẠO DỮ LIỆU MẪU (Cho init_data.py) ---
def get_initial_node_info():
    ts = int(time.time() * 1000)
    return [
        {"nodeId": "NodeA", "nodeType": "GROUND_STATION", "position": {"latitude": 10.0, "longitude": 10.0, "altitude": 0.0}, "resourceUtilization": 0.2, "currentPacketCount": 50, "packetBufferCapacity": 1000, "isOperational": True, "lastUpdated": ts},
        {"nodeId": "NodeB", "nodeType": "LEO_SATELLITE", "position": {"latitude": 20.0, "longitude": 20.0, "altitude": 500.0}, "resourceUtilization": 0.1, "currentPacketCount": 30, "packetBufferCapacity": 1000, "isOperational": True, "lastUpdated": ts},
        {"nodeId": "NodeC", "nodeType": "LEO_SATELLITE", "position": {"latitude": 30.0, "longitude": 10.0, "altitude": 500.0}, "resourceUtilization": 0.15, "currentPacketCount": 40, "packetBufferCapacity": 1000, "isOperational": True, "lastUpdated": ts},
        {"nodeId": "NodeD", "nodeType": "GROUND_STATION", "position": {"latitude": 40.0, "longitude": 40.0, "altitude": 0.0}, "resourceUtilization": 0.05, "currentPacketCount": 20, "packetBufferCapacity": 1000, "isOperational": True, "lastUpdated": ts}
    ]

def get_initial_link_metrics():
    ts = int(time.time() * 1000)
    return [
        {"linkKey": "NodeA-NodeB", "sourceNodeId": "NodeA", "destinationNodeId": "NodeB", "latencyMs": 5.0, "currentAvailableBandwidthMbps": 900.0, "packetLossRate": 0.005, "isLinkActive": True, "linkScore": 90.0, "lastUpdated": ts},
        {"linkKey": "NodeB-NodeA", "sourceNodeId": "NodeB", "destinationNodeId": "NodeA", "latencyMs": 5.0, "currentAvailableBandwidthMbps": 900.0, "packetLossRate": 0.005, "isLinkActive": True, "linkScore": 90.0, "lastUpdated": ts},
        
        {"linkKey": "NodeA-NodeC", "sourceNodeId": "NodeA", "destinationNodeId": "NodeC", "latencyMs": 10.0, "currentAvailableBandwidthMbps": 700.0, "packetLossRate": 0.01, "isLinkActive": True, "linkScore": 60.0, "lastUpdated": ts},
        {"linkKey": "NodeC-NodeA", "sourceNodeId": "NodeC", "destinationNodeId": "NodeA", "latencyMs": 10.0, "currentAvailableBandwidthMbps": 700.0, "packetLossRate": 0.01, "isLinkActive": True, "linkScore": 60.0, "lastUpdated": ts},
        
        {"linkKey": "NodeC-NodeD", "sourceNodeId": "NodeC", "destinationNodeId": "NodeD", "latencyMs": 8.0, "currentAvailableBandwidthMbps": 850.0, "packetLossRate": 0.008, "isLinkActive": True, "linkScore": 75.0, "lastUpdated": ts},
        {"linkKey": "NodeD-NodeC", "sourceNodeId": "NodeD", "destinationNodeId": "NodeC", "latencyMs": 8.0, "currentAvailableBandwidthMbps": 850.0, "packetLossRate": 0.008, "isLinkActive": True, "linkScore": 75.0, "lastUpdated": ts},
        
        {"linkKey": "NodeB-NodeD", "sourceNodeId": "NodeB", "destinationNodeId": "NodeD", "latencyMs": 20.0, "currentAvailableBandwidthMbps": 400.0, "packetLossRate": 0.05, "isLinkActive": True, "linkScore": 30.0, "lastUpdated": ts},
        {"linkKey": "NodeD-NodeB", "sourceNodeId": "NodeD", "destinationNodeId": "NodeB", "latencyMs": 20.0, "currentAvailableBandwidthMbps": 400.0, "packetLossRate": 0.05, "isLinkActive": True, "linkScore": 30.0, "lastUpdated": ts},
    ]