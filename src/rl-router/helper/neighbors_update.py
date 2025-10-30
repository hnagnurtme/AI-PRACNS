from pymongo import MongoClient
from typing import List, Dict, Any
import math
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NodeType config ---
NODE_TYPE_MAX_RANGE = {
    "GROUND_STATION": 2000.0,   # km
    "LEO_SATELLITE": 3000.0,
    "MEO_SATELLITE": 10000.0,
    "GEO_SATELLITE": 35000.0
}

# --- MongoDB setup ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://user:password123@localhost:27017/?authSource=admin")
DB_NAME = "sagsin_network"
COLLECTION_NAME = "network_nodes"

class NodeService:
    def __init__(self, uri: str = MONGO_URI):
        self.client = MongoClient(uri)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        logger.info("Connected to MongoDB: %s, DB: %s, Collection: %s", uri, DB_NAME, COLLECTION_NAME)

    @staticmethod
    def distance_3d(node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
        """Tính khoảng cách 3D giữa hai node (km)."""
        x1, y1, z1 = NodeService.geo_to_xyz(node1)
        x2, y2, z2 = NodeService.geo_to_xyz(node2)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    @staticmethod
    def geo_to_xyz(node: Dict[str, Any]) -> tuple:
        """Chuyển kinh độ, vĩ độ, altitude sang hệ tọa độ 3D (km)."""
        lat = math.radians(node["position"]["latitude"])
        lon = math.radians(node["position"]["longitude"])
        alt = node["position"].get("altitude", 0.0)

        R = 6371.0 + alt  # bán kính Trái Đất + altitude (km)
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        return x, y, z

    def compute_neighbors(self, nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Tính neighbors dựa trên NodeType và khoảng cách."""
        neighbors_map = {}
        for node in nodes:
            node_id = node["nodeId"]
            node_type = node.get("nodeType", "GROUND_STATION")
            max_range = NODE_TYPE_MAX_RANGE.get(node_type, 2000.0)

            neighbors = [
                n["nodeId"]
                for n in nodes
                if n["nodeId"] != node_id
                and self.distance_3d(node, n) <= max_range
            ]
            neighbors_map[node_id] = neighbors
        return neighbors_map

    def update_neighbors_in_db(self):
        """Load tất cả node, tính neighbors và update vào MongoDB."""
        nodes = list(self.collection.find({}))
        neighbors_map = self.compute_neighbors(nodes)

        for node_id, neighbors in neighbors_map.items():
            self.collection.update_one(
                {"nodeId": node_id},
                {"$set": {"neighbors": neighbors}}
            )
            logger.info("Node %s -> neighbors updated (%d nodes)", node_id, len(neighbors))
        logger.info("✅ All neighbors updated successfully.")

    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    service = NodeService()
    try:
        service.update_neighbors_in_db()
    finally:
        service.close()
