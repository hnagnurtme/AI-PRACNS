import pymongo
from typing import List, Dict, Any, Tuple
import math
import logging
import os

# --- Logger setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NodeType config ---
R_EARTH = 6371.0  # km
NODE_TYPE_MAX_RANGE = {
    "GROUND_STATION": 2000.0,   
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
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        logger.info("Connected to MongoDB: %s, DB: %s, Collection: %s", uri, DB_NAME, COLLECTION_NAME)

    @staticmethod
    def geo_to_xyz(node: Dict[str, Any]) -> Tuple[float, float, float]:
        """Chuyển Lat/Lon/Alt sang tọa độ 3D ECEF (km)."""
        lat = math.radians(node["position"]["latitude"])
        lon = math.radians(node["position"]["longitude"])
        alt = node["position"].get("altitude", 0.0)
        R = R_EARTH + alt
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        return x, y, z

    @staticmethod
    def distance_3d(node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
        x1, y1, z1 = NodeService.geo_to_xyz(node1)
        x2, y2, z2 = NodeService.geo_to_xyz(node2)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    @staticmethod
    def los_max_distance(node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
        """
        Tính khoảng cách tối đa theo LOS (Ground ↔ Satellite / Satellite ↔ Satellite khác loại).
        Sử dụng mô hình hình học đơn giản: GS có thể "nhìn thấy" vệ tinh nếu
        khoảng cách < sqrt((R+h1)^2 + (R+h2)^2 - 2*R^2)  (approx)
        """
        alt1 = node1["position"].get("altitude", 0.0)
        alt2 = node2["position"].get("altitude", 0.0)
        return math.sqrt((R_EARTH + alt1)**2 + (R_EARTH + alt2)**2) - R_EARTH

    def compute_neighbors(self, nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Xây dựng neighbors dựa trên:
        - Node cùng loại: khoảng cách <= NODE_TYPE_MAX_RANGE
        - Node khác loại: khoảng cách <= LOS
        """
        neighbors_map = {}
        for node_a in nodes:
            node_a_id = node_a["nodeId"]
            type_a = node_a.get("nodeType", "GROUND_STATION")
            neighbors = []

            for node_b in nodes:
                if node_b["nodeId"] == node_a_id:
                    continue
                type_b = node_b.get("nodeType", "GROUND_STATION")
                distance = self.distance_3d(node_a, node_b)

                if type_a == type_b:
                    max_dist = NODE_TYPE_MAX_RANGE.get(type_a, 2000.0)
                else:
                    max_dist = self.los_max_distance(node_a, node_b)

                if distance <= max_dist:
                    neighbors.append(node_b["nodeId"])

            neighbors_map[node_a_id] = neighbors
        return neighbors_map

    def update_neighbors_in_db(self):
        """Tải tất cả node, tính neighbors, cập nhật MongoDB."""
        nodes = list(self.collection.find({}))
        logger.info("Calculating neighbors for %d nodes...", len(nodes))
        
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
    except Exception as e:
        logger.error(f"FATAL ERROR during neighbor update: {e}", exc_info=True)
    finally:
        service.close()
