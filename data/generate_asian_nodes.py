import random
import math
from datetime import datetime

# Constants for distance calculation
EARTH_RADIUS_KM = 6371.0 # Earth radius in kilometers

def geo_to_ecef(pos: dict) -> tuple:
    """
    Converts geographic coordinates (latitude, longitude, altitude) to ECEF (x, y, z).
    Latitude and longitude are in degrees, altitude in km.
    """
    lat = math.radians(pos['latitude'])
    lon = math.radians(pos['longitude'])
    alt = pos['altitude']

    N = EARTH_RADIUS_KM # Simplified, ignoring Earth's oblateness for N
    
    x = (N + alt) * math.cos(lat) * math.cos(lon)
    y = (N + alt) * math.cos(lat) * math.sin(lon)
    z = (N + alt) * math.sin(lat) # Simplified, ignoring Earth's oblateness for Z

    return x, y, z

def distance_3d(pos1_ecef: tuple, pos2_ecef: tuple) -> float:
    """
    Calculates the 3D Euclidean distance between two points in ECEF coordinates.
    """
    x1, y1, z1 = pos1_ecef
    x2, y2, z2 = pos2_ecef
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def generate_asian_nodes_data():
    """
    Generates a dataset of 30 nodes with a focus on the Asian region.
    Includes a mix of Ground Stations, LEO, MEO, and GEO satellites.
    """

    nodes = []

    # --- Ground Stations (10) ---
    ground_stations_data = {
        "GS_TOKYO": {"latitude": 35.6895, "longitude": 139.6917, "altitude": 0},
        "GS_SEOUL": {"latitude": 37.5665, "longitude": 126.9780, "altitude": 0},
        "GS_BEIJING": {"latitude": 39.9042, "longitude": 116.4074, "altitude": 0},
        "GS_SHANGHAI": {"latitude": 31.2304, "longitude": 121.4737, "altitude": 0},
        "GS_HONGKONG": {"latitude": 22.3193, "longitude": 114.1694, "altitude": 0},
        "GS_SINGAPORE": {"latitude": 1.3521, "longitude": 103.8198, "altitude": 0},
        "GS_KUALALUMPUR": {"latitude": 3.1390, "longitude": 101.6869, "altitude": 0},
        "GS_BANGKOK": {"latitude": 13.7563, "longitude": 100.5018, "altitude": 0},
        "GS_JAKARTA": {"latitude": -6.2088, "longitude": 106.8456, "altitude": 0},
        "GS_DELHI": {"latitude": 28.7041, "longitude": 77.1025, "altitude": 0},
        "GS_HANOI" : {"latitude": 21.0285, "longitude": 105.8542, "altitude": 0},
        "GS_DANANG" : {"latitude": 16.0545, "longitude": 108.2022, "altitude": 0},
        "GS_HOCHIMINH" : {"latitude": 10.7769, "longitude": 106.7009, "altitude": 0},
    }

    for name, pos in ground_stations_data.items():
        nodes.append({
            "nodeId": name,
            "nodeType": "GROUND_STATION",
            "position": pos,
            "communication": {
                "frequencyGHz": 2.4,
                "bandwidthMHz": 100,
                "maxConnections": 50,
                "rangeKm": 5000 # Ground stations can connect to satellites within this range (increased from 1000)
            },
            "operational": True,
            "neighbors": [], # Will be populated later
            "resourceUtilization": random.uniform(0.1, 0.3),
            "currentPacketCount": random.randint(0, 10),
            "packetBufferCapacity": 100,
            "nodeProcessingDelayMs": random.uniform(1.0, 5.0),
            "packetLossRate": random.uniform(0.001, 0.01),
            "lastUpdated": datetime.now()
        })

    # --- LEO Satellites (15) ---
    for i in range(1, 16):
        nodes.append({
            "nodeId": f"LEO_{i:02d}",
            "nodeType": "LEO_SATELLITE",
            "position": {
                "latitude": random.uniform(-10, 50),
                "longitude": random.uniform(70, 140),
                "altitude": random.uniform(500, 2000),
            },
            "communication": {
                "frequencyGHz": 12.0,
                "bandwidthMHz": 500,
                "maxConnections": 20,
                "rangeKm": 10000 # LEOs have a limited range (increased from 3000)
            },
            "operational": True,
            "neighbors": [],
            "resourceUtilization": random.uniform(0.3, 0.7),
            "currentPacketCount": random.randint(10, 50),
            "packetBufferCapacity": 200,
            "nodeProcessingDelayMs": random.uniform(5.0, 15.0),
            "packetLossRate": random.uniform(0.01, 0.05),
            "lastUpdated": datetime.now()
        })

    # --- MEO Satellites (5) ---
    for i in range(1, 6):
        nodes.append({
            "nodeId": f"MEO_{i:02d}",
            "nodeType": "MEO_SATELLITE",
            "position": {
                "latitude": random.uniform(-20, 60),
                "longitude": random.uniform(60, 150),
                "altitude": random.uniform(2000, 35000),
            },
            "communication": {
                "frequencyGHz": 18.0,
                "bandwidthMHz": 1000,
                "maxConnections": 50,
                "rangeKm": 30000 # MEOs have a larger range (increased from 10000)
            },
            "operational": True,
            "neighbors": [],
            "resourceUtilization": random.uniform(0.2, 0.6),
            "currentPacketCount": random.randint(5, 30),
            "packetBufferCapacity": 150,
            "nodeProcessingDelayMs": random.uniform(3.0, 10.0),
            "packetLossRate": random.uniform(0.005, 0.03),
            "lastUpdated": datetime.now()
        })

    # --- GEO Satellites (2) ---
    for i in range(1, 3):
        nodes.append({
            "nodeId": f"GEO_{i:02d}",
            "nodeType": "GEO_SATELLITE",
            "position": {
                "latitude": 0, # GEO satellites are on the equatorial plane
                "longitude": random.uniform(80, 130),
                "altitude": 35786,
            },
            "communication": {
                "frequencyGHz": 20.0,
                "bandwidthMHz": 2000,
                "maxConnections": 100,
                "rangeKm": 80000 # GEOs have a very large range (increased from 40000)
            },
            "operational": True,
            "neighbors": [],
            "resourceUtilization": random.uniform(0.05, 0.2),
            "currentPacketCount": random.randint(0, 5),
            "packetBufferCapacity": 50,
            "nodeProcessingDelayMs": random.uniform(0.5, 2.0),
            "packetLossRate": random.uniform(0.0001, 0.005),
            "lastUpdated": datetime.now()
        })
    
    # --- Generate Neighbors ---
    # Cache ECEF positions
    ecef_positions = {node["nodeId"]: geo_to_ecef(node["position"]) for node in nodes}

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i == j:
                continue

            # Check if nodes are operational
            if not node1.get("operational", False) or not node2.get("operational", False):
                continue

            # Get communication range for each node
            range1 = node1["communication"].get("rangeKm", 0)
            range2 = node2["communication"].get("rangeKm", 0)

            # Calculate 3D distance
            dist = distance_3d(ecef_positions[node1["nodeId"]], ecef_positions[node2["nodeId"]])
            
            # Check if within communication range
            if dist <= range1 and len(node1["neighbors"]) < node1["communication"]["maxConnections"]:
                node1["neighbors"].append(node2["nodeId"])
            
            if dist <= range2 and len(node2["neighbors"]) < node2["communication"]["maxConnections"]:
                node2["neighbors"].append(node1["nodeId"])

    # Ensure unique neighbors and limit to maxConnections
    for node in nodes:
        node["neighbors"] = list(set(node["neighbors"]))
        node["neighbors"] = node["neighbors"][:node["communication"]["maxConnections"]]

    print(len(nodes))
    return nodes
if __name__ == '__main__':
    data = generate_asian_nodes_data()
    import json
    with open('asian_nodes.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("Generated asian_nodes.json")
