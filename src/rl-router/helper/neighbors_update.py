import pymongo
import random
import math
import logging

# ----------------- Logger -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------- Config -----------------
R_EARTH = 6371.0  # km
NODE_TYPE_MAX_RANGE = {
    "GROUND_STATION": 2000.0,
    "LEO_SATELLITE": 3000.0,
    "MEO_SATELLITE": 10000.0,
    "GEO_SATELLITE": 35000.0
}

DB_URI = "mongodb://user:password123@localhost:27017/?authSource=admin"
DB_NAME = "sagsin_network"
COLLECTION_NAME = "network_nodes"

# ----------------- Helper Functions -----------------
def geo_to_xyz(lat: float, lon: float, alt: float) -> tuple:
    """Chuyển lat/lon/alt (km) sang hệ tọa độ ECEF (x, y, z) km"""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    R = R_EARTH + alt
    x = R * math.cos(lat_rad) * math.cos(lon_rad)
    y = R * math.cos(lat_rad) * math.sin(lon_rad)
    z = R * math.sin(lat_rad)
    return x, y, z

def distance_3d(node1: dict, node2: dict) -> float:
    x1, y1, z1 = geo_to_xyz(node1["position"]["latitude"], node1["position"]["longitude"], node1["position"]["altitude"])
    x2, y2, z2 = geo_to_xyz(node2["position"]["latitude"], node2["position"]["longitude"], node2["position"]["altitude"])
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def los_max_distance(node1: dict, node2: dict) -> float:
    """Khoảng cách tối đa theo LOS giữa node khác loại"""
    alt1 = node1["position"].get("altitude", 0.0)
    alt2 = node2["position"].get("altitude", 0.0)
    return math.sqrt((R_EARTH + alt1)**2 + (R_EARTH + alt2)**2) - R_EARTH

# ----------------- Generate Nodes -----------------
gs_nodes = [
    {"nodeId": "N-HANOI", "nodeType": "GROUND_STATION", "position": {"latitude": 21.0285, "longitude": 105.8542, "altitude": 0.02}},
    {"nodeId": "N-BEIJING", "nodeType": "GROUND_STATION", "position": {"latitude": 39.9042, "longitude": 116.4074, "altitude": 0.05}},
    {"nodeId": "N-TOKYO", "nodeType": "GROUND_STATION", "position": {"latitude": 35.6895, "longitude": 139.6917, "altitude": 0.04}},
    {"nodeId": "N-SINGAPORE", "nodeType": "GROUND_STATION", "position": {"latitude": 1.3521, "longitude": 103.8198, "altitude": 0.03}},
]

leo_nodes = []
for i in range(1, 11):  # 10 LEO
    lat = random.uniform(-60, 60)
    lon = random.uniform(60, 150)
    leo_nodes.append({"nodeId": f"SAT-LEO-{i}", "nodeType": "LEO_SATELLITE", "position": {"latitude": lat, "longitude": lon, "altitude": 550}})

meo_nodes = []
for i in range(1, 6):  # 5 MEO
    lat = random.uniform(-40, 40)
    lon = random.uniform(70, 150)
    meo_nodes.append({"nodeId": f"SAT-MEO-{i}", "nodeType": "MEO_SATELLITE", "position": {"latitude": lat, "longitude": lon, "altitude": 20000}})

geo_nodes = []
geo_positions = [100, 110, 120]  # GEO longitudes Asia
for i, lon in enumerate(geo_positions, 1):
    geo_nodes.append({"nodeId": f"SAT-GEO-{i}", "nodeType": "GEO_SATELLITE", "position": {"latitude": 0.0, "longitude": lon, "altitude": 35786}})

all_nodes = gs_nodes + leo_nodes + meo_nodes + geo_nodes

# ----------------- Compute Neighbors -----------------
def compute_neighbors(nodes: list) -> dict:
    neighbors_map = {}
    for node_a in nodes:
        node_a_id = node_a["nodeId"]
        type_a = node_a["nodeType"]
        neighbors = []
        for node_b in nodes:
            if node_b["nodeId"] == node_a_id:
                continue
            type_b = node_b["nodeType"]
            dist = distance_3d(node_a, node_b)
            if type_a == type_b:
                max_dist = NODE_TYPE_MAX_RANGE.get(type_a, 2000.0)
            else:
                max_dist = los_max_distance(node_a, node_b)
            if dist <= max_dist:
                neighbors.append(node_b["nodeId"])
        neighbors_map[node_a_id] = neighbors
    return neighbors_map

neighbors_map = compute_neighbors(all_nodes)
for node in all_nodes:
    node["neighbors"] = neighbors_map[node["nodeId"]]

# ----------------- Insert to MongoDB -----------------
mongo_client = pymongo.MongoClient(DB_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
collection.delete_many({})
collection.insert_many(all_nodes)
logger.info(f"Inserted {len(all_nodes)} nodes with neighbors into MongoDB.")
