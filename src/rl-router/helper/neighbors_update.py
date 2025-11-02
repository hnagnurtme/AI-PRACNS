import pymongo
import math
import logging
from typing import List, Dict, Any, Tuple
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

MAX_RANGE_MAP = {
    ("LEO_SATELLITE","MEO_SATELLITE"): 12000,
    ("LEO_SATELLITE","GEO_SATELLITE"): 40000,
    ("MEO_SATELLITE","GEO_SATELLITE"): 30000,
    ("LEO_SATELLITE","LEO_SATELLITE"): NODE_TYPE_MAX_RANGE["LEO_SATELLITE"],
    ("MEO_SATELLITE","MEO_SATELLITE"): NODE_TYPE_MAX_RANGE["MEO_SATELLITE"],
    ("GEO_SATELLITE","GEO_SATELLITE"): NODE_TYPE_MAX_RANGE["GEO_SATELLITE"],
}

# --- MongoDB setup ---
MONGO_URI = "mongodb+srv://admin:SMILEisme0106@mongo1.ragz4ka.mongodb.net/?appName=MONGO1"
DB_NAME = "network"
COLLECTION_NAME = "network_nodes"

class NodeService:
    def __init__(self, uri: str = MONGO_URI):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        logger.info("Connected to MongoDB: %s, DB: %s, Collection: %s", uri, DB_NAME, COLLECTION_NAME)

    @staticmethod
    def geo_to_xyz(node: Dict[str, Any]) -> Tuple[float, float, float]:
        lat = math.radians(node["position"]["latitude"])
        lon = math.radians(node["position"]["longitude"])
        alt = node["position"].get("altitude", 0.0)
        R = R_EARTH + alt
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        return x, y, z

    @staticmethod
    def distance_3d(pos1: Tuple[float,float,float], pos2: Tuple[float,float,float]) -> float:
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    @staticmethod
    def elevation_ok(gs_pos: Tuple[float,float,float], sat_pos: Tuple[float,float,float], min_elev_deg=5.0) -> bool:
        dx = sat_pos[0] - gs_pos[0]
        dy = sat_pos[1] - gs_pos[1]
        dz = sat_pos[2] - gs_pos[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        elev_rad = math.asin(dz/distance)
        return math.degrees(elev_rad) >= min_elev_deg

    def compute_neighbors(self, nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        neighbors_map = {}

        # Cache 3D positions
        pos_cache = {node["nodeId"]: self.geo_to_xyz(node) for node in nodes}

        for node_a in nodes:
            node_a_id = node_a["nodeId"]
            type_a = node_a.get("nodeType", "GROUND_STATION")
            pos_a = pos_cache[node_a_id]
            neighbors = []

            for node_b in nodes:
                node_b_id = node_b["nodeId"]
                if node_b_id == node_a_id:
                    continue
                type_b = node_b.get("nodeType", "GROUND_STATION")
                pos_b = pos_cache[node_b_id]

                # --- GS ↔ GS ---
                if type_a == "GROUND_STATION" and type_b == "GROUND_STATION":
                    max_range = NODE_TYPE_MAX_RANGE["GROUND_STATION"]
                    if self.distance_3d(pos_a, pos_b) <= max_range:
                        neighbors.append(node_b_id)
                    continue

                # --- GS ↔ Satellite ---
                if type_a == "GROUND_STATION" or type_b == "GROUND_STATION":
                    gs_pos = pos_a if type_a == "GROUND_STATION" else pos_b
                    sat_pos = pos_b if gs_pos is pos_a else pos_a
                    if self.elevation_ok(gs_pos, sat_pos):
                        neighbors.append(node_b_id)
                    continue

                # --- Satellite ↔ Satellite same type ---
                if type_a == type_b:
                    max_range = NODE_TYPE_MAX_RANGE.get(type_a, 2000.0)
                    if self.distance_3d(pos_a, pos_b) <= max_range:
                        neighbors.append(node_b_id)
                    continue

                # --- Satellite ↔ Satellite cross-type ---
                key = (type_a,type_b) if (type_a,type_b) in MAX_RANGE_MAP else (type_b,type_a)
                max_range = MAX_RANGE_MAP.get(key, 15000)
                if self.distance_3d(pos_a, pos_b) <= max_range:
                    neighbors.append(node_b_id)

            neighbors_map[node_a_id] = neighbors

        return neighbors_map

    def update_neighbors_in_db(self):
        nodes = list(self.collection.find({}))
        logger.info("Calculating neighbors for %d nodes...", len(nodes))
        neighbors_map = self.compute_neighbors(nodes)

        for node_id, neighbors in neighbors_map.items():
            self.collection.update_one(
                {"nodeId": node_id},
                {"$set": {
                    "neighbors": neighbors,
                    "host": "127.0.0.1"
                }}
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
    finally:
        service.close()
