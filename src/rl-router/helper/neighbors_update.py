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

    def compute_neighbors(self, nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Cập nhật neighbors thực tế giữa các loại node:
        - GS ↔ tất cả satellites trong tầm nhìn.
        - Satellites ↔ satellites theo khoảng cách max thực tế.
        """
        neighbors_map = {}

        def elevation_ok(gs: Dict[str, Any], sat: Dict[str, Any], min_elev_deg=5.0) -> bool:
            """Kiểm tra satellite có nằm trong góc nhìn từ GS."""
            alt_gs = gs["position"].get("altitude", 0.0)
            alt_sat = sat["position"].get("altitude", 0.0)
            dist = self.distance_3d(gs, sat)
            max_dist = math.sqrt((R_EARTH + alt_sat)**2 - R_EARTH**2)
            return dist <= max_dist

        for node_a in nodes:
            node_a_id = node_a["nodeId"]
            type_a = node_a.get("nodeType", "GROUND_STATION")
            neighbors = []

            for node_b in nodes:
                if node_b["nodeId"] == node_a_id:
                    continue
                type_b = node_b.get("nodeType", "GROUND_STATION")
                distance = self.distance_3d(node_a, node_b)

                # Same type
                if type_a == type_b:
                    max_dist = NODE_TYPE_MAX_RANGE.get(type_a, 2000.0)
                    if distance <= max_dist:
                        neighbors.append(node_b["nodeId"])
                else:
                    # Cross-type
                    if type_a == "GROUND_STATION" or type_b == "GROUND_STATION":
                        gs = node_a if type_a == "GROUND_STATION" else node_b
                        sat = node_b if gs is node_a else node_a
                        if elevation_ok(gs, sat):
                            neighbors.append(node_b["nodeId"])
                    else:
                        # Satellite ↔ Satellite
                        max_range_map = {
                            ("LEO_SATELLITE","MEO_SATELLITE"): 12000,
                            ("LEO_SATELLITE","GEO_SATELLITE"): 40000,
                            ("MEO_SATELLITE","GEO_SATELLITE"): 30000,
                            ("LEO_SATELLITE","LEO_SATELLITE"): NODE_TYPE_MAX_RANGE["LEO_SATELLITE"],
                            ("MEO_SATELLITE","MEO_SATELLITE"): NODE_TYPE_MAX_RANGE["MEO_SATELLITE"],
                            ("GEO_SATELLITE","GEO_SATELLITE"): NODE_TYPE_MAX_RANGE["GEO_SATELLITE"],
                        }
                        key = (type_a,type_b) if (type_a,type_b) in max_range_map else (type_b,type_a)
                        max_dist = max_range_map.get(key, 15000)
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
            logger.info(f"  {node_id}: {len(neighbors)} neighbors updated")
        logger.info("✅ All neighbors updated successfully.")

    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    service = NodeService()
    try:
        service.update_neighbors_in_db()
    except Exception as e:
        logger.error(f"❌ FATAL ERROR during neighbor update: {e}", exc_info=True)
    finally:
        service.close()
