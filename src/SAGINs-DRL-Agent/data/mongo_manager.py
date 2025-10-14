# data/mongo_manager.py
import time
import copy
import random
from typing import Dict, List, Any
from pymongo import MongoClient

class MongoDataManager:
    def __init__(self, host: str = "localhost", port: int = 27017, db_name: str = "sagins-network", username: str = "user", password: str = "password123", auth_source: str = "admin"):
        connection_string = f"mongodb://{username}:{password}@{host}:{port}/?authSource={auth_source}"
        self.client = MongoClient(connection_string)
        
        # Test connection
        try:
            self.client.admin.command('ping')
            print("✅ Connected with authentication")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise
        
        self.db = self.client['SAGSINS']
        self.node_info = self.db['network_nodes']
        
        self._node_cache = {}
        self.last_update_time = 0
        
    def get_training_snapshot(self, cache_duration: int = 300) -> Dict[str, Any]:
        """Lấy snapshot dữ liệu training với cache strategy"""
        current_time = time.time()

        if (current_time - self.last_update_time > cache_duration or not self._node_cache):
            print("🔄 Loading fresh node data from MongoDB...")
            
            # Chỉ lấy nodes từ MongoDB
            nodes = self._fetch_all_nodes()
            
            # Tạo link variants từ nodes
            link_variants = self._create_link_variants_from_nodes(nodes, num_variants=200)
            
            self._node_cache = nodes
            self.last_update_time = current_time
            
            print(f"✅ Loaded {len(nodes)} nodes, created {len(link_variants)} link variants")
            
        return {
            "nodes": self._node_cache,
            "link_variants": []
        }
    
    def _fetch_all_nodes(self) -> Dict[str, Dict]: 
        """Lấy tất cả các nodes từ MongoDB"""
        nodes = {}
        
        # Add debug info
        count = self.node_info.count_documents({})
        print(f"📊 Found {count} documents in collection")
        
        for node_doc in self.node_info.find({}):
            node_id = node_doc.get('_id')
            print(f"🔍 Processing node: {node_id}")  # Debug log
            
            node_data = {
                'nodeId': node_id,
                'batteryChargePercent': node_doc.get('batteryChargePercent'),
                'currentPacketCount': node_doc.get('currentPacketCount'),
                'healthy': node_doc.get('healthy'),
                'lastUpdated': node_doc.get('lastUpdated'),
                'nodeProcessingDelayMs': node_doc.get('nodeProcessingDelayMs'),
                'nodeType': node_doc.get('nodeType'),
                'isOperational': node_doc.get('operational'),
                'orbit': node_doc.get('orbit', {}),
                'packetBufferCapacity': node_doc.get('packetBufferCapacity'),
                'packetLossRate': node_doc.get('packetLossRate'),
                'port': node_doc.get('port'),
                'position': node_doc.get('position', {}),
                'resourceUtilization': node_doc.get('resourceUtilization'),
                'velocity': node_doc.get('velocity', {}),
                'weather': node_doc.get('weather'),
            }
            nodes[node_id] = node_data
            
        print(f"✅ Successfully loaded {len(nodes)} nodes")
        return nodes

    def _create_link_variants_from_nodes(self, nodes: Dict[str, Dict], num_variants: int = 200) -> List[Dict]:
        """Tạo link variants từ nodes"""
        return []
    
    def get_all_node_ids(self) -> List[str]:
        snapshot = self.get_training_snapshot()
        return list(snapshot['nodes'].keys())