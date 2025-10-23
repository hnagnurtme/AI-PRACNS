from pymongo import MongoClient
from typing import List, Dict, Optional
import math
import logging

logger = logging.getLogger(__name__)

class MongoManager:
    def __init__(self, uri: str = "mongodb://user:password123@localhost:27017/?authSource=admin"):
        self.client = MongoClient(uri)
        self.db = self.client['sagsin_network']
        self.nodes = self.db['network_nodes']
        logger.info("Connected to MongoDB at %s", uri)
        
    def get_all_nodes(self) -> List[Dict]:
        return list(self.nodes.find({}))
    
    def update_node(self, node_id: str, updates: Dict):
        self.nodes.update_one({"nodeId": node_id}, {"$set": updates })
        
    def get_closest_gs(self, client_position: Dict) -> str:
        gs_nodes = list(self.nodes.find({"nodeType": "GROUND_STATION"}))
        if not gs_nodes:
            logger.error("No ground stations found in DB")
            raise ValueError("No ground stations available")
        min_dist = float('inf')
        closest = None
        for gs in gs_nodes:
            dist = self.calculate_distance(client_position, gs['position'])
            logger.debug(f"GS {gs['nodeId']} distance: {dist}km")
            if dist < min_dist:
                min_dist = dist
                closest = gs['nodeId']
        if closest is None:
            logger.error("Could not determine the closest ground station")
            raise ValueError("Could not determine the closest ground station")
        return closest
    
    @staticmethod
    def calculate_distance(pos1: Dict, pos2: Dict) -> float:
        try:
            lat1, lon1 = math.radians(pos1['latitude']), math.radians(pos1['longitude'])
            lat2, lon2 = math.radians(pos2['latitude']), math.radians(pos2['longitude'])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            dist = 6371 * c
            alt_diff = abs(pos1.get('altitude', 0) - pos2.get('altitude', 0))
            return math.sqrt(dist**2 + alt_diff**2)
        except KeyError as e:
            logger.error(f"Missing position key: {e}")
            return float('inf')
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a single node by nodeId"""
        node = self.nodes.find_one({"nodeId": node_id})
        if not node:
            logger.warning(f"Node {node_id} not found in database")
        return node
