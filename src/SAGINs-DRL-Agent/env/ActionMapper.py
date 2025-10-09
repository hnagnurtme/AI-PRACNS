# env/ActionMapper.py
from typing import Dict, List, Optional
import random
from utils.static_data import BASE_NODE_INFO 

class ActionMapper:
    
    def __init__(self):
        self.node_to_index: Dict[str, int] = {} 
        self.index_to_node: Dict[int, str] = {}
        self.sync_node_list()

    def sync_node_list(self):
        all_nodes = BASE_NODE_INFO.keys() 
        self.node_to_index.clear()
        self.index_to_node.clear()
        
        sorted_node_ids = sorted(list(all_nodes))
        
        for index, node_id in enumerate(sorted_node_ids):
            self.node_to_index[node_id] = index
            self.index_to_node[index] = node_id
        
    def get_action_index(self, node_id: str) -> Optional[int]:
        self.sync_node_list()
        return self.node_to_index.get(node_id)

    def map_index_to_node_id(self, index: int) -> str:
        self.sync_node_list()
        
        if 0 <= index < len(self.index_to_node):
            return self.index_to_node[index]
        return random.choice(list(self.node_to_index.keys()))

    def get_action_size(self) -> int:
        self.sync_node_list()
        return len(self.node_to_index)