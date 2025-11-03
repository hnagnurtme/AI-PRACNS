import json
import math
import random
from datetime import datetime, timezone

# ---------------- Config ----------------
R_EARTH = 6371.0  # km

NUM_GS = 20
NUM_LEO = 50
NUM_MEO = 20
NUM_GEO = 10

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
}

# ---------------- Utils ----------------
def geo_to_xyz(lat_deg, lon_deg, alt_km):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    R = R_EARTH + alt_km
    x = R * math.cos(lat) * math.cos(lon)
    y = R * math.cos(lat) * math.sin(lon)
    z = R * math.sin(lat)
    return x, y, z

def distance_3d(pos1, pos2):
    x1,y1,z1 = pos1
    x2,y2,z2 = pos2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def elevation_ok(gs_pos, sat_pos, min_elev_deg=5.0):
    dx = sat_pos[0] - gs_pos[0]
    dy = sat_pos[1] - gs_pos[1]
    dz = sat_pos[2] - gs_pos[2]
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    elev_rad = math.asin(dz/distance)
    return math.degrees(elev_rad) >= min_elev_deg

# ---------------- Generate Nodes ----------------
nodes = []

# 1️⃣ Ground Stations (thực tế 20 địa điểm)
gs_locations = [
    ("GS_HANOI", 21.0285, 105.8542),
    ("GS_HOCHIMINH", 10.7769, 106.7009),
    ("GS_DANANG", 16.0544, 108.2022),
    ("GS_JAKARTA", -6.2088, 106.8456),
    ("GS_SINGAPORE", 1.3521, 103.8198),
    ("GS_TOKYO", 35.6895, 139.6917),
    ("GS_SEOUL", 37.5665, 126.9780),
    ("GS_BANGKOK", 13.7563, 100.5018),
    ("GS_KUALALUMPUR", 3.1390, 101.6869),
    ("GS_DELHI", 28.6139, 77.2090),
    ("GS_SYDNEY", -33.8688, 151.2093),
    ("GS_LONDON", 51.5074, -0.1278),
    ("GS_PARIS", 48.8566, 2.3522),
    ("GS_BERLIN", 52.5200, 13.4050),
    ("GS_NEWYORK", 40.7128, -74.0060),
    ("GS_SANFRAN", 37.7749, -122.4194),
    ("GS_DUBAI", 25.276987, 55.296249),
    ("GS_MOSCOW", 55.7558, 37.6173),
    ("GS_CAIRE", 30.0444, 31.2357),
    ("GS_RIO", -22.9068, -43.1729)
]

for node_id, lat, lon in gs_locations:
    node = {
        "nodeId": node_id,
        "nodeName": node_id,
        "nodeType": "GROUND_STATION",
        "position": {"latitude": lat, "longitude": lon, "altitude": 0.0},
        "orbit": {"semiMajorAxisKm":0,"eccentricity":0,"inclinationDeg":0,"raanDeg":0,"argumentOfPerigeeDeg":0,"trueAnomalyDeg":0},
        "velocity": {"velocityX":0,"velocityY":0,"velocityZ":0},
        "communication": {
            "frequencyGHz": random.uniform(2,3),
            "bandwidthMHz": random.randint(100,300),
            "transmitPowerDbW": random.randint(15,30),
            "antennaGainDb": random.randint(20,35),
            "beamWidthDeg": random.uniform(10,30),
            "maxRangeKm": NODE_TYPE_MAX_RANGE["GROUND_STATION"],
            "minElevationDeg":5,
            "ipAddress": f"10.0.0.{random.randint(1,254)}",
            "port": 7700 + random.randint(0,99),
            "protocol":"TCP"
        },
        "isOperational": True,
        "batteryChargePercent": random.randint(70,100),
        "nodeProcessingDelayMs": round(random.uniform(1,10),2),
        "packetLossRate": round(random.uniform(0,0.02),4),
        "resourceUtilization": round(random.uniform(0.1,0.7),2),
        "packetBufferCapacity": random.randint(500,5000),
        "currentPacketCount": random.randint(0,500),
        "weather": random.choice(["CLEAR","LIGHT_RAIN","STORM"]),
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "healthy": True,
        "neighbors": []  # sẽ tính sau
    }
    nodes.append(node)

# 2️⃣ LEO Satellites
for i in range(1, NUM_LEO+1):
    node = {
        "nodeId": f"LEO-{i:02d}",
        "nodeName": f"LEO-{i:02d}",
        "nodeType": "LEO_SATELLITE",
        "position": {"latitude": random.uniform(-90,90), "longitude": random.uniform(-180,180), "altitude": 500+random.uniform(0,100)}, # km
        "orbit": {"semiMajorAxisKm":0,"eccentricity":0,"inclinationDeg":random.uniform(0,98),"raanDeg":0,"argumentOfPerigeeDeg":0,"trueAnomalyDeg":0},
        "velocity": {"velocityX":0,"velocityY":0,"velocityZ":0},
        "communication": {"frequencyGHz": random.uniform(10,15),"bandwidthMHz":random.randint(100,500),
                          "transmitPowerDbW":random.randint(20,30),"antennaGainDb":random.randint(20,35),
                          "beamWidthDeg": random.uniform(5,20),"maxRangeKm":NODE_TYPE_MAX_RANGE["LEO_SATELLITE"],
                          "minElevationDeg":5,"ipAddress":f"10.1.0.{i}","port":7800+i,"protocol":"TCP"},
        "isOperational": True,
        "batteryChargePercent": random.randint(50,100),
        "nodeProcessingDelayMs": round(random.uniform(1,10),2),
        "packetLossRate": round(random.uniform(0,0.05),4),
        "resourceUtilization": round(random.uniform(0.1,0.8),2),
        "packetBufferCapacity": random.randint(500,5000),
        "currentPacketCount": random.randint(0,500),
        "weather": random.choice(["CLEAR","LIGHT_RAIN","STORM"]),
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "healthy": True,
        "neighbors": []
    }
    nodes.append(node)

# 3️⃣ MEO Satellites
for i in range(1, NUM_MEO+1):
    node = {
        "nodeId": f"MEO-{i:02d}",
        "nodeName": f"MEO-{i:02d}",
        "nodeType": "MEO_SATELLITE",
        "position": {"latitude": random.uniform(-90,90), "longitude": random.uniform(-180,180), "altitude": 10000+random.uniform(0,500)}, # km
        "orbit": {"semiMajorAxisKm":0,"eccentricity":0,"inclinationDeg":random.uniform(0,98),"raanDeg":0,"argumentOfPerigeeDeg":0,"trueAnomalyDeg":0},
        "velocity": {"velocityX":0,"velocityY":0,"velocityZ":0},
        "communication": {"frequencyGHz": random.uniform(10,15),"bandwidthMHz":random.randint(100,500),
                          "transmitPowerDbW":random.randint(20,30),"antennaGainDb":random.randint(20,35),
                          "beamWidthDeg": random.uniform(5,20),"maxRangeKm":NODE_TYPE_MAX_RANGE["MEO_SATELLITE"],
                          "minElevationDeg":5,"ipAddress":f"10.2.0.{i}","port":7900+i,"protocol":"TCP"},
        "isOperational": True,
        "batteryChargePercent": random.randint(50,100),
        "nodeProcessingDelayMs": round(random.uniform(1,10),2),
        "packetLossRate": round(random.uniform(0,0.05),4),
        "resourceUtilization": round(random.uniform(0.1,0.8),2),
        "packetBufferCapacity": random.randint(500,5000),
        "currentPacketCount": random.randint(0,500),
        "weather": random.choice(["CLEAR","LIGHT_RAIN","STORM"]),
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "healthy": True,
        "neighbors": []
    }
    nodes.append(node)

# 4️⃣ GEO Satellites
for i in range(1, NUM_GEO+1):
    node = {
        "nodeId": f"GEO-{i:02d}",
        "nodeName": f"GEO-{i:02d}",
        "nodeType": "GEO_SATELLITE",
        "position": {"latitude": random.uniform(-90,90), "longitude": random.uniform(-180,180), "altitude": 35786}, # km
        "orbit": {"semiMajorAxisKm":0,"eccentricity":0,"inclinationDeg":0,"raanDeg":0,"argumentOfPerigeeDeg":0,"trueAnomalyDeg":0},
        "velocity": {"velocityX":0,"velocityY":0,"velocityZ":0},
        "communication": {"frequencyGHz": random.uniform(10,15),"bandwidthMHz":random.randint(100,500),
                          "transmitPowerDbW":random.randint(20,30),"antennaGainDb":random.randint(20,35),
                          "beamWidthDeg": random.uniform(5,20),"maxRangeKm":NODE_TYPE_MAX_RANGE["GEO_SATELLITE"],
                          "minElevationDeg":5,"ipAddress":f"10.3.0.{i}","port":8000+i,"protocol":"TCP"},
        "isOperational": True,
        "batteryChargePercent": random.randint(50,100),
        "nodeProcessingDelayMs": round(random.uniform(1,10),2),
        "packetLossRate": round(random.uniform(0,0.05),4),
        "resourceUtilization": round(random.uniform(0.1,0.8),2),
        "packetBufferCapacity": random.randint(500,5000),
        "currentPacketCount": random.randint(0,500),
        "weather": random.choice(["CLEAR","LIGHT_RAIN","STORM"]),
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "healthy": True,
        "neighbors": []
    }
    nodes.append(node)

# ---------------- Save to JSON ----------------
with open("network_nodes.json", "w") as f:
    json.dump(nodes, f, indent=2)
print(f"✅ Generated {len(nodes)} nodes in 'network_nodes.json'")
