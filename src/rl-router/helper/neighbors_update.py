import pymongo
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
import os
from datetime import datetime

# Support loading .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Logger setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MongoDB setup ---
from python.utils.db_connector import LOCAL_MONGO_URI, CLOUD_MONGO_URI
DB_NAME = "sagsin_network" # Use the consistent DB name

# Constants for ECEF conversion
R_EARTH = 6371.0  # Earth radius in kilometers

class NodeService:
    def __init__(self, uri: Optional[str] = None, use_cloud_db: bool = False):
        if uri:
            resolved_uri = uri
        elif use_cloud_db:
            resolved_uri = CLOUD_MONGO_URI
        else:
            resolved_uri = LOCAL_MONGO_URI
            
        self.client = pymongo.MongoClient(resolved_uri)
        self.db = self.client[DB_NAME]
        self.collection = self.db["network_nodes"] # Use the consistent collection name
        logger.info("Connected to MongoDB, DB: %s, Collection: %s", DB_NAME, "network_nodes")

    @staticmethod
    def geo_to_ecef(pos: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Converts geographic coordinates (latitude, longitude, altitude) to ECEF (x, y, z).
        Latitude and longitude are in degrees, altitude in km.
        """
        lat = math.radians(pos['latitude'])
        lon = math.radians(pos['longitude'])
        alt = pos['altitude']

        N = R_EARTH # Simplified, ignoring Earth's oblateness for N
        
        x = (N + alt) * math.cos(lat) * math.cos(lon)
        y = (N + alt) * math.cos(lat) * math.sin(lon)
        z = (N + alt) * math.sin(lat) # Simplified, ignoring Earth's oblateness for Z

        return x, y, z

    @staticmethod
    def distance_3d(pos1_ecef: Tuple[float,float,float], pos2_ecef: Tuple[float,float,float]) -> float:
        """
        Calculates the 3D Euclidean distance between two points in ECEF coordinates.
        """
        x1, y1, z1 = pos1_ecef
        x2, y2, z2 = pos2_ecef
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    def compute_neighbors(self, nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        neighbors_map = {node["nodeId"]: [] for node in nodes}

        # Cache ECEF positions and communication ranges
        node_info_cache = {}
        for node in nodes:
            node_info_cache[node["nodeId"]] = {
                "ecef_pos": self.geo_to_ecef(node["position"]),
                "rangeKm": node["communication"].get("rangeKm", 0),
                "maxConnections": node["communication"].get("maxConnections", 0),
                "operational": node.get("operational", False)
            }

        for node1_id, info1 in node_info_cache.items():
            if not info1["operational"]:
                continue

            for node2_id, info2 in node_info_cache.items():
                if node1_id == node2_id or not info2["operational"]:
                    continue

                # Calculate 3D distance
                dist = self.distance_3d(info1["ecef_pos"], info2["ecef_pos"])
                
                # Check if within communication range of node1
                if dist <= info1["rangeKm"] and len(neighbors_map[node1_id]) < info1["maxConnections"]:
                    neighbors_map[node1_id].append(node2_id)
                
                # Check if within communication range of node2 (bidirectional)
                if dist <= info2["rangeKm"] and len(neighbors_map[node2_id]) < info2["maxConnections"]:
                    neighbors_map[node2_id].append(node1_id)

        # Ensure unique neighbors and limit to maxConnections
        for node_id in neighbors_map:
            node_data = next((n for n in nodes if n["nodeId"] == node_id), None)
            if node_data:
                neighbors_map[node_id] = list(set(neighbors_map[node_id]))
                neighbors_map[node_id] = neighbors_map[node_id][:node_data["communication"]["maxConnections"]]

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
                    "lastUpdated": datetime.now() # Update timestamp
                }}
            )
            logger.info(f"  {node_id}: {len(neighbors)} neighbors updated")
        logger.info("✅ All neighbors updated successfully.")

    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Update node neighbors in MongoDB.")
    parser.add_argument("--cloud", action="store_true", help="Update cloud database instead of local.")
    args = parser.parse_args()

    service = NodeService(use_cloud_db=args.cloud)
    try:
        service.update_neighbors_in_db()
    finally:
        service.close()
