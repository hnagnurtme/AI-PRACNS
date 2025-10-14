# env/action_mapper.py
from typing import Dict, List, Optional
import random
from utils.static_data import BASE_NODE_INFO 
from data.mongo_manager import MongoDataManager

class ActionMapper:
    """Ánh xạ giữa action index và node ID"""
    
    def __init__(self, mongo_manager: MongoDataManager):
        self.mongo_manager = mongo_manager
        self.node_to_index: Dict[str, int] = {}
        self.index_to_node: Dict[int, str] = {}
        self._sync_node_list()
    
    def _sync_node_list(self):
        """Đồng bộ danh sách node từ MongoDB"""
        snapshot = self.mongo_manager.get_training_snapshot()
        all_nodes = snapshot.get('nodes', {}).keys()
        
        self.node_to_index.clear()
        self.index_to_node.clear()
        
        sorted_nodes = sorted(all_nodes)
        for index, node_id in enumerate(sorted_nodes):
            self.node_to_index[node_id] = index
            self.index_to_node[index] = node_id
        
        print(f"ActionMapper synced: {len(self.node_to_index)} nodes mapped.")
        
    def get_action_index(self, node_id: str) -> Optional[int]:
        """Chuyển node ID thành action index"""
        self._sync_node_list()
        return self.node_to_index.get(node_id)
    
    def map_index_to_node_id(self, index: int) -> str:
        """Chuyển action index thành node ID"""
        self._sync_node_list()
        
        if 0 <= index < len(self.index_to_node):
            return self.index_to_node[index]
        
        # Fallback random
        available_nodes = list(self.node_to_index.keys())
        return random.choice(available_nodes) if available_nodes else "UNKNOWN_NODE"
    
    def get_action_size(self) -> int:
        """Trả về số lượng action (số lượng node hiện có)"""
        self._sync_node_list()
        return len(self.node_to_index)
    
    def get_available_nodes(self) -> List[str]:
        """Trả về danh sách node ID hiện có"""
        self._sync_node_list()
        return list(self.node_to_index.keys())