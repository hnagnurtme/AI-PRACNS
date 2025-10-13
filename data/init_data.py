from db_config import get_collection
from datetime import datetime

def generate_Node():
    return [
        {
            "nodeId": "GS-01",
            "nodeName": "GroundStation-Hanoi",
            "type": "GROUND_STATION",
            "position": { "latitude": 21.0285, "longitude": 105.8542, "altitude": 0.03 },
            "velocity": { "velocityX": 0.0, "velocityY": 0.0, "velocityZ": 0.0 },
            "orbit": None,
            "communication": {
            "frequencyGHz": 14.0,
            "bandwidthMHz": 300.0,
            "transmitPowerDbW": 25.0,
            "antennaGainDb": 30.0,
            "beamWidthDeg": 10.0,
            "maxRangeKm": 2500.0,
            "minElevationDeg": 10.0,
            "ipAddress": "10.0.0.1",
            "port": 8080,
            "protocol": "TCP"
            },
            "status": { "active": True, "batteryChargePercent": 100, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2022-01-01T00:00:00Z", "notes": "Primary ground uplink station" }
        },
        {
            "nodeId": "GS-02",
            "nodeName": "GroundStation-HoChiMinh",
            "type": "GROUND_STATION",
            "position": { "latitude": 10.7626, "longitude": 106.6602, "altitude": 0.05 },
            "velocity": { "velocityX": 0.0, "velocityY": 0.0, "velocityZ": 0.0 },
            "orbit": None,
            "communication": {
            "frequencyGHz": 14.25,
            "bandwidthMHz": 400.0,
            "transmitPowerDbW": 25.0,
            "antennaGainDb": 28.0,
            "beamWidthDeg": 12.0,
            "maxRangeKm": 2500.0,
            "minElevationDeg": 10.0,
            "ipAddress": "10.0.0.2",
            "port": 8081,
            "protocol": "TCP"
            },
            "status": { "active": True, "batteryChargePercent": 100, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2022-03-01T00:00:00Z", "notes": "Southern region GS" }
        },
        {
            "nodeId": "LEO-001",
            "nodeName": "Sat-LEO-1",
            "type": "LEO_SATELLITE",
            "position": { "latitude": 12.0, "longitude": 110.0, "altitude": 550.0 },
            "velocity": { "velocityX": 0.1, "velocityY": 7.55, "velocityZ": 0.05 },
            "orbit": { "semiMajorAxisKm": 6871.0, "eccentricity": 0.001, "inclinationDeg": 53.0, "raanDeg": 40.0, "argumentOfPerigeeDeg": 0.0, "trueAnomalyDeg": 20.0 },
            "communication": { "frequencyGHz": 14.25, "bandwidthMHz": 500.0, "transmitPowerDbW": 20.0, "antennaGainDb": 15.0, "beamWidthDeg": 3.0, "maxRangeKm": 2000.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.12", "port": 8082, "protocol": "UDP" },
            "status": { "active": True, "batteryChargePercent": 90.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2024-03-01T00:00:00Z", "notes": "Low-latency orbit" }
        },
        {
            "nodeId": "LEO-002",
            "nodeName": "Sat-LEO-2",
            "type": "LEO_SATELLITE",
            "position": { "latitude": -5.0, "longitude": 100.0, "altitude": 560.0 },
            "velocity": { "velocityX": -0.2, "velocityY": 7.56, "velocityZ": 0.0 },
            "orbit": { "semiMajorAxisKm": 6872.0, "eccentricity": 0.002, "inclinationDeg": 55.0, "raanDeg": 60.0, "argumentOfPerigeeDeg": 10.0, "trueAnomalyDeg": 160.0 },
            "communication": { "frequencyGHz": 14.25, "bandwidthMHz": 450.0, "transmitPowerDbW": 21.0, "antennaGainDb": 14.0, "beamWidthDeg": 4.0, "maxRangeKm": 1900.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.13", "port": 8083, "protocol": "UDP" },
            "status": { "active": True, "batteryChargePercent": 87.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2024-03-01T00:00:00Z", "notes": "Crosslink with LEO-001" }
        },
        {
            "nodeId": "LEO-003",
            "nodeName": "Sat-LEO-3",
            "type": "LEO_SATELLITE",
            "position": { "latitude": 15.0, "longitude": 90.0, "altitude": 550.0 },
            "velocity": { "velocityX": 0.15, "velocityY": 7.53, "velocityZ": 0.02 },
            "orbit": { "semiMajorAxisKm": 6870.0, "eccentricity": 0.0005, "inclinationDeg": 51.6, "raanDeg": 100.0, "argumentOfPerigeeDeg": 0.0, "trueAnomalyDeg": 300.0 },
            "communication": { "frequencyGHz": 14.1, "bandwidthMHz": 480.0, "transmitPowerDbW": 22.0, "antennaGainDb": 16.0, "beamWidthDeg": 3.5, "maxRangeKm": 2100.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.14", "port": 8084, "protocol": "UDP" },
            "status": { "active": True, "batteryChargePercent": 92.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2024-03-01T00:00:00Z", "notes": "Equatorial coverage" }
        },
        {
            "nodeId": "LEO-004",
            "nodeName": "Sat-LEO-4",
            "type": "LEO_SATELLITE",
            "position": { "latitude": 30.0, "longitude": 70.0, "altitude": 550.0 },
            "velocity": { "velocityX": 0.0, "velocityY": 7.54, "velocityZ": -0.1 },
            "orbit": { "semiMajorAxisKm": 6871.0, "eccentricity": 0.001, "inclinationDeg": 60.0, "raanDeg": 140.0, "argumentOfPerigeeDeg": 5.0, "trueAnomalyDeg": 240.0 },
            "communication": { "frequencyGHz": 14.5, "bandwidthMHz": 450.0, "transmitPowerDbW": 21.0, "antennaGainDb": 15.0, "beamWidthDeg": 3.5, "maxRangeKm": 2000.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.15", "port": 8085, "protocol": "UDP" },
            "status": { "active": True, "batteryChargePercent": 88.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2024-03-01T00:00:00Z", "notes": "LEO polar path" }
        },
        {
            "nodeId": "MEO-001",
            "nodeName": "Sat-MEO-1",
            "type": "MEO_SATELLITE",
            "position": { "latitude": 5.0, "longitude": 60.0, "altitude": 20200.0 },
            "velocity": { "velocityX": 0.0, "velocityY": 3.9, "velocityZ": 0.0 },
            "orbit": { "semiMajorAxisKm": 26571.0, "eccentricity": 0.01, "inclinationDeg": 56.0, "raanDeg": 250.0, "argumentOfPerigeeDeg": 0.0, "trueAnomalyDeg": 45.0 },
            "communication": { "frequencyGHz": 12.0, "bandwidthMHz": 600.0, "transmitPowerDbW": 23.0, "antennaGainDb": 18.0, "beamWidthDeg": 4.0, "maxRangeKm": 10000.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.16", "port": 8086, "protocol": "UDP" },
            "status": { "active": True, "batteryChargePercent": 95.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2023-01-01T00:00:00Z", "notes": "Mid-orbit relay" }
        },
        {
            "nodeId": "MEO-002",
            "nodeName": "Sat-MEO-2",
            "type": "MEO_SATELLITE",
            "position": { "latitude": -10.0, "longitude": 150.0, "altitude": 20200.0 },
            "velocity": { "velocityX": -0.1, "velocityY": 3.8, "velocityZ": 0.0 },
            "orbit": { "semiMajorAxisKm": 26570.0, "eccentricity": 0.01, "inclinationDeg": 55.0, "raanDeg": 270.0, "argumentOfPerigeeDeg": 0.0, "trueAnomalyDeg": 120.0 },
            "communication": { "frequencyGHz": 12.0, "bandwidthMHz": 600.0, "transmitPowerDbW": 23.0, "antennaGainDb": 18.0, "beamWidthDeg": 4.0, "maxRangeKm": 10000.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.17", "port": 8087, "protocol": "UDP" },
            "status": { "active": True, "batteryChargePercent": 94.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2023-01-01T00:00:00Z", "notes": "MEO east node" }
        },
        {
            "nodeId": "GEO-001",
            "nodeName": "Sat-GEO-Asia",
            "type": "GEO_SATELLITE",
            "position": { "latitude": 0.0, "longitude": 105.0, "altitude": 35786.0 },
            "velocity": { "velocityX": 0.0, "velocityY": 3.07, "velocityZ": 0.0 },
            "orbit": { "semiMajorAxisKm": 42164.0, "eccentricity": 0.0, "inclinationDeg": 0.0, "raanDeg": 0.0, "argumentOfPerigeeDeg": 0.0, "trueAnomalyDeg": 0.0 },
            "communication": { "frequencyGHz": 11.5, "bandwidthMHz": 1000.0, "transmitPowerDbW": 28.0, "antennaGainDb": 25.0, "beamWidthDeg": 0.5, "maxRangeKm": 50000.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.18", "port": 8088, "protocol": "TCP" },
            "status": { "active": True, "batteryChargePercent": 97.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2023-06-01T00:00:00Z", "notes": "Stable backbone relay" }
        },
        {
            "nodeId": "GEO-002",
            "nodeName": "Sat-GEO-Pacific",
            "type": "GEO_SATELLITE",
            "position": { "latitude": 0.0, "longitude": 150.0, "altitude": 35786.0 },
            "velocity": { "velocityX": 0.0, "velocityY": 3.07, "velocityZ": 0.0 },
            "orbit": { "semiMajorAxisKm": 42164.0, "eccentricity": 0.0, "inclinationDeg": 0.0, "raanDeg": 0.0, "argumentOfPerigeeDeg": 0.0, "trueAnomalyDeg": 0.0 },
            "communication": { "frequencyGHz": 11.5, "bandwidthMHz": 900.0, "transmitPowerDbW": 28.0, "antennaGainDb": 25.0, "beamWidthDeg": 0.5, "maxRangeKm": 50000.0, "minElevationDeg": 10.0, "ipAddress": "10.0.0.19", "port": 8089, "protocol": "TCP" },
            "status": { "active": True, "batteryChargePercent": 97.0, "lastUpdated": "2025-10-13T16:00:00Z" },
            "metadata": { "operator": "VNPT Space", "launchDate": "2023-06-01T00:00:00Z", "notes": "Pacific relay node" }
        }
        ]

def init_Node():
    list_node = generate_Node()
    nodes = get_collection("nodes")
    for node in list_node:
        node["status"]["lastUpdated"] = datetime.utcnow().isoformat() + "Z"
        nodes.update_one({"nodeId": node["nodeId"]}, {"$set": node}, upsert=True)
    print(f"Initialized {len(list_node)} nodes.")

if __name__ == "__main__":
    init_Node()