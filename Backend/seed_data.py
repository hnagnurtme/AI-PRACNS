#!/usr/bin/env python3
"""
Deterministic Seed Data Generator for SAGIN RL Training
Creates structured, reproducible data optimized for RL training and evaluation
"""
from pymongo import MongoClient
from datetime import datetime
import os
import math
from typing import Dict, List, Tuple

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://admin:password@localhost:27017/aiprancs?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'aiprancs')

# Physical constants (from constants.py)
EARTH_RADIUS_KM = 6371.0
EARTH_MU = 398600.4418
SPEED_OF_LIGHT_KM_S = 299792.458

# Training-optimized configuration
NUM_LEO_SATELLITES = 24  # 4 planes x 6 satellites for good coverage
NUM_MEO_SATELLITES = 6   # Even distribution
NUM_GEO_SATELLITES = 3   # Equatorial coverage
NUM_GROUND_STATIONS = 20 # Strategic locations
NUM_TERMINALS = 30       # Diverse scenarios

# Deterministic seed for reproducibility
DETERMINISTIC_SEED = 42


def calculate_orbital_velocity(altitude_km: float) -> float:
    """Calculate orbital velocity using vis-viva equation"""
    r = EARTH_RADIUS_KM + altitude_km
    return math.sqrt(EARTH_MU / r)


def calculate_orbital_period(altitude_km: float) -> float:
    """Calculate orbital period in minutes"""
    r = EARTH_RADIUS_KM + altitude_km
    period_seconds = 2 * math.pi * math.sqrt(r**3 / EARTH_MU)
    return period_seconds / 60


def generate_node_id(index: int, node_type: str) -> str:
    """Generate deterministic node ID"""
    prefix = {
        'LEO_SATELLITE': 'LEO',
        'MEO_SATELLITE': 'MEO',
        'GEO_SATELLITE': 'GEO',
        'GROUND_STATION': 'GS'
    }.get(node_type, 'NODE')
    return f"{prefix}-{index:03d}"


def generate_terminal_id(index: int) -> str:
    """Generate deterministic terminal ID"""
    return f"TERM-{index:04d}"


def generate_leo_constellation() -> List[Dict]:
    """
    Generate LEO satellites in Walker-Delta constellation pattern
    Deterministic positions for consistent training
    """
    satellites = []
    num_planes = 4
    sats_per_plane = NUM_LEO_SATELLITES // num_planes
    altitude_km = 550
    inclination = 53.0
    
    for plane in range(num_planes):
        for sat_in_plane in range(sats_per_plane):
            index = plane * sats_per_plane + sat_in_plane
            
            # Walker-Delta pattern
            raan = plane * (360.0 / num_planes)
            true_anomaly = sat_in_plane * (360.0 / sats_per_plane)
            
            # Convert to lat/lon (simplified for deterministic positions)
            lon = (raan + true_anomaly) % 360 - 180
            lat = math.sin(math.radians(true_anomaly)) * inclination
            
            velocity_orbital = calculate_orbital_velocity(altitude_km)
            velocity_ms = velocity_orbital * 1000
            
            angle_rad = math.radians(true_anomaly)
            raan_rad = math.radians(raan)
            
            satellites.append({
                'index': index,
                'plane': plane,
                'sat_in_plane': sat_in_plane,
                'position': {
                    'latitude': round(lat, 4),
                    'longitude': round(lon, 4),
                    'altitude': altitude_km * 1000
                },
                'velocity': {
                    'velocityX': round(velocity_ms * math.cos(angle_rad) * math.cos(raan_rad), 3),
                    'velocityY': round(velocity_ms * math.cos(angle_rad) * math.sin(raan_rad), 3),
                    'velocityZ': round(velocity_ms * math.sin(angle_rad) * math.sin(math.radians(inclination)), 3)
                },
                'altitude_km': altitude_km,
                'orbital_period': calculate_orbital_period(altitude_km)
            })
    
    return satellites


def generate_meo_constellation() -> List[Dict]:
    """Generate MEO satellites in deterministic pattern"""
    satellites = []
    altitude_km = 8000
    inclination = 70.0
    
    for i in range(NUM_MEO_SATELLITES):
        longitude = -180 + (i * 360.0 / NUM_MEO_SATELLITES)
        true_anomaly = i * (360.0 / NUM_MEO_SATELLITES)
        latitude = math.sin(math.radians(true_anomaly)) * inclination
        
        velocity_orbital = calculate_orbital_velocity(altitude_km)
        velocity_ms = velocity_orbital * 1000
        angle_rad = math.radians(true_anomaly)
        
        satellites.append({
            'index': i,
            'position': {
                'latitude': round(latitude, 4),
                'longitude': round(longitude, 4),
                'altitude': altitude_km * 1000
            },
            'velocity': {
                'velocityX': round(velocity_ms * math.cos(angle_rad), 3),
                'velocityY': round(velocity_ms * math.sin(angle_rad), 3),
                'velocityZ': round(velocity_ms * 0.3, 3)
            },
            'altitude_km': altitude_km,
            'orbital_period': calculate_orbital_period(altitude_km)
        })
    
    return satellites


def generate_geo_constellation() -> List[Dict]:
    """Generate GEO satellites at fixed equatorial positions"""
    satellites = []
    altitude_km = 35786
    
    longitudes = [-120.0, 0.0, 120.0]  # Americas, Europe/Africa, Asia/Pacific
    
    for i, lon in enumerate(longitudes):
        satellites.append({
            'index': i,
            'position': {
                'latitude': 0.0,
                'longitude': lon,
                'altitude': altitude_km * 1000
            },
            'velocity': {
                'velocityX': 0.0,
                'velocityY': 0.0,
                'velocityZ': 0.0
            },
            'altitude_km': altitude_km,
            'orbital_period': calculate_orbital_period(altitude_km)
        })
    
    return satellites


def generate_ground_stations() -> List[Dict]:
    """
    Generate ground stations at strategic locations
    Distributed globally for good coverage
    """
    # Strategic locations for training scenarios
    locations = [
        # North America - Easy scenarios
        {'lat': 40.7128, 'lon': -74.0060, 'name': 'New York', 'alt': 10, 'region': 'NA'},
        {'lat': 34.0522, 'lon': -118.2437, 'name': 'Los Angeles', 'alt': 100, 'region': 'NA'},
        {'lat': 41.8781, 'lon': -87.6298, 'name': 'Chicago', 'alt': 180, 'region': 'NA'},
        {'lat': 29.7604, 'lon': -95.3698, 'name': 'Houston', 'alt': 15, 'region': 'NA'},
        {'lat': 45.5017, 'lon': -73.5673, 'name': 'Montreal', 'alt': 36, 'region': 'NA'},
        
        # Europe - Medium scenarios
        {'lat': 51.5074, 'lon': -0.1278, 'name': 'London', 'alt': 15, 'region': 'EU'},
        {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris', 'alt': 35, 'region': 'EU'},
        {'lat': 52.5200, 'lon': 13.4050, 'name': 'Berlin', 'alt': 34, 'region': 'EU'},
        {'lat': 55.7558, 'lon': 37.6173, 'name': 'Moscow', 'alt': 156, 'region': 'EU'},
        {'lat': 41.9028, 'lon': 12.4964, 'name': 'Rome', 'alt': 57, 'region': 'EU'},
        
        # Asia - Hard scenarios (long distances)
        {'lat': 35.6762, 'lon': 139.6503, 'name': 'Tokyo', 'alt': 40, 'region': 'ASIA'},
        {'lat': 22.3193, 'lon': 114.1694, 'name': 'Hong Kong', 'alt': 5, 'region': 'ASIA'},
        {'lat': 1.3521, 'lon': 103.8198, 'name': 'Singapore', 'alt': 15, 'region': 'ASIA'},
        {'lat': 28.6139, 'lon': 77.2090, 'name': 'New Delhi', 'alt': 216, 'region': 'ASIA'},
        {'lat': 31.2304, 'lon': 121.4737, 'name': 'Shanghai', 'alt': 4, 'region': 'ASIA'},
        
        # Oceania & Others - Cross-continental scenarios
        {'lat': -33.8688, 'lon': 151.2093, 'name': 'Sydney', 'alt': 25, 'region': 'OCEANIA'},
        {'lat': -37.8136, 'lon': 144.9631, 'name': 'Melbourne', 'alt': 31, 'region': 'OCEANIA'},
        {'lat': -23.5505, 'lon': -46.6333, 'name': 'Sao Paulo', 'alt': 760, 'region': 'SA'},
        {'lat': -34.6037, 'lon': -58.3816, 'name': 'Buenos Aires', 'alt': 25, 'region': 'SA'},
        {'lat': -1.2921, 'lon': 36.8219, 'name': 'Nairobi', 'alt': 1795, 'region': 'AFRICA'},
    ]
    
    return locations[:NUM_GROUND_STATIONS]


def get_deterministic_qos(service_type: str, index: int) -> Dict:
    """
    Generate deterministic QoS based on service type and index
    Creates consistent requirements for training
    """
    base_profiles = {
        'VIDEO_STREAM': {
            'maxLatencyMs': 200.0,
            'minBandwidthMbps': 15.0,
            'maxLossRate': 0.005,
            'priority': 7
        },
        'AUDIO_CALL': {
            'maxLatencyMs': 100.0,
            'minBandwidthMbps': 0.5,
            'maxLossRate': 0.01,
            'priority': 9
        },
        'IMAGE_TRANSFER': {
            'maxLatencyMs': 500.0,
            'minBandwidthMbps': 5.0,
            'maxLossRate': 0.01,
            'priority': 5
        },
        'TEXT_MESSAGE': {
            'maxLatencyMs': 1000.0,
            'minBandwidthMbps': 0.05,
            'maxLossRate': 0.05,
            'priority': 3
        },
        'FILE_TRANSFER': {
            'maxLatencyMs': 3000.0,
            'minBandwidthMbps': 50.0,
            'maxLossRate': 0.02,
            'priority': 4
        }
    }
    
    qos = base_profiles.get(service_type, base_profiles['FILE_TRANSFER']).copy()
    qos['serviceType'] = service_type
    
    # Add small deterministic variations based on index
    variation = (index % 10) / 100.0
    qos['maxLatencyMs'] = round(qos['maxLatencyMs'] * (1 + variation * 0.1), 2)
    qos['minBandwidthMbps'] = round(qos['minBandwidthMbps'] * (1 + variation * 0.1), 2)
    
    return qos


def get_node_attributes(node_type: str, index: int, total: int) -> Dict:
    """
    Generate deterministic node attributes for training
    Creates patterns: some nodes are better, some have constraints
    """
    # Base attributes by node type
    if node_type == 'LEO_SATELLITE':
        base_utilization = 30.0 + (index % 5) * 10.0  # 30-70%
        base_battery = 85.0 + (index % 3) * 5.0  # 85-95%
        base_loss = 0.001 + (index % 3) * 0.002  # 0.001-0.005
        processing_delay = 3.0 + (index % 4) * 1.5  # 3-9ms
        buffer_capacity = 5000
        current_packets = int(buffer_capacity * (0.2 + (index % 5) * 0.15))
        
    elif node_type == 'MEO_SATELLITE':
        base_utilization = 40.0 + (index % 4) * 10.0  # 40-70%
        base_battery = 90.0 + (index % 2) * 5.0  # 90-95%
        base_loss = 0.0005 + (index % 2) * 0.001  # 0.0005-0.0015
        processing_delay = 8.0 + (index % 3) * 3.0  # 8-14ms
        buffer_capacity = 10000
        current_packets = int(buffer_capacity * (0.15 + (index % 4) * 0.1))
        
    elif node_type == 'GEO_SATELLITE':
        base_utilization = 50.0 + (index % 3) * 15.0  # 50-80%
        base_battery = 95.0 + (index % 2) * 2.5  # 95-97.5%
        base_loss = 0.0001 + (index % 2) * 0.0002  # 0.0001-0.0003
        processing_delay = 12.0 + (index % 2) * 4.0  # 12-16ms
        buffer_capacity = 20000
        current_packets = int(buffer_capacity * (0.1 + (index % 3) * 0.05))
        
    else:  # GROUND_STATION
        base_utilization = 20.0 + (index % 6) * 8.0  # 20-60%
        base_battery = 100.0  # Always full
        base_loss = 0.0001 + (index % 3) * 0.0002  # 0.0001-0.0005
        processing_delay = 1.0 + (index % 3) * 0.5  # 1-2.5ms
        buffer_capacity = 10000
        current_packets = int(buffer_capacity * (0.1 + (index % 5) * 0.08))
    
    # Resource breakdown (for Dijkstra features)
    cpu_util = base_utilization + (index % 3) * 5.0
    mem_util = base_utilization - (index % 3) * 3.0
    bw_util = base_utilization + (index % 2) * 4.0
    
    return {
        'resourceUtilization': round(base_utilization, 1),
        'cpu': {'utilization': round(min(100.0, cpu_util), 1)},
        'memory': {'utilization': round(min(100.0, mem_util), 1)},
        'bandwidth': {'utilization': round(min(100.0, bw_util), 1)},
        'batteryChargePercent': round(base_battery, 1),
        'packetLossRate': round(base_loss, 5),
        'nodeProcessingDelayMs': round(processing_delay, 2),
        'packetBufferCapacity': buffer_capacity,
        'currentPacketCount': current_packets,
        'isOperational': True  # All operational for training
    }


def create_nodes(db) -> List[str]:
    """
    Create deterministic nodes optimized for training
    Note: Assumes existing data has already been cleared
    """
    nodes_collection = db['nodes']
    nodes = []
    node_index = 0
    
    # LEO Satellites
    leo_constellation = generate_leo_constellation()
    for leo_data in leo_constellation:
        attrs = get_node_attributes('LEO_SATELLITE', leo_data['index'], NUM_LEO_SATELLITES)
        
        node = {
            'id': generate_node_id(node_index, 'LEO_SATELLITE'),
            'nodeId': generate_node_id(node_index, 'LEO_SATELLITE'),
            'nodeName': f'LEO Satellite {leo_data["index"]+1} (Plane {leo_data["plane"]+1})',
            'nodeType': 'LEO_SATELLITE',
            'position': leo_data['position'],
            'velocity': leo_data['velocity'],
            'orbit': {
                'semiMajorAxisKm': EARTH_RADIUS_KM + leo_data['altitude_km'],
                'eccentricity': 0.0001,
                'inclinationDeg': 53.0,
                'raanDeg': leo_data['plane'] * (360.0 / 4),
                'argumentOfPerigeeDeg': 0.0,
                'trueAnomalyDeg': leo_data['sat_in_plane'] * (360.0 / 6),
                'orbitalPeriodMin': round(leo_data['orbital_period'], 2)
            },
            'communication': {
                'frequencyGHz': 11.7 + (leo_data['index'] % 3) * 0.5,
                'bandwidthMHz': 300 + (leo_data['index'] % 4) * 50,
                'transmitPowerDbW': 40.0 + (leo_data['index'] % 3) * 1.0,
                'antennaGainDb': 30.0 + (leo_data['index'] % 2) * 1.0,
                'beamWidthDeg': 10.0 + (leo_data['index'] % 3) * 2.0,
                'maxRangeKm': 2500.0,
                'minElevationDeg': 25.0,
                'protocol': 'DVB-S2X',
                'ipAddress': f'10.1.{leo_data["index"]//256}.{leo_data["index"]%256}',
                'port': 8080 + leo_data['index']
            },
            **attrs,
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True
        }
        nodes.append(node)
        node_index += 1
    
    # MEO Satellites
    meo_constellation = generate_meo_constellation()
    for meo_data in meo_constellation:
        attrs = get_node_attributes('MEO_SATELLITE', meo_data['index'], NUM_MEO_SATELLITES)
        
        node = {
            'id': generate_node_id(node_index, 'MEO_SATELLITE'),
            'nodeId': generate_node_id(node_index, 'MEO_SATELLITE'),
            'nodeName': f'MEO Satellite {meo_data["index"]+1}',
            'nodeType': 'MEO_SATELLITE',
            'position': meo_data['position'],
            'velocity': meo_data['velocity'],
            'orbit': {
                'semiMajorAxisKm': EARTH_RADIUS_KM + meo_data['altitude_km'],
                'eccentricity': 0.0001,
                'inclinationDeg': 70.0,
                'raanDeg': meo_data['index'] * (360.0 / NUM_MEO_SATELLITES),
                'argumentOfPerigeeDeg': 0.0,
                'trueAnomalyDeg': meo_data['index'] * (360.0 / NUM_MEO_SATELLITES),
                'orbitalPeriodMin': round(meo_data['orbital_period'], 2)
            },
            'communication': {
                'frequencyGHz': 18.0 + (meo_data['index'] % 2) * 1.0,
                'bandwidthMHz': 500 + (meo_data['index'] % 3) * 100,
                'transmitPowerDbW': 45.0 + (meo_data['index'] % 2) * 1.0,
                'antennaGainDb': 35.0 + (meo_data['index'] % 2) * 1.5,
                'beamWidthDeg': 6.0 + (meo_data['index'] % 2) * 1.0,
                'maxRangeKm': 10000.0,
                'minElevationDeg': 20.0,
                'protocol': 'DVB-S2X',
                'ipAddress': f'10.2.{meo_data["index"]//256}.{meo_data["index"]%256}',
                'port': 8080 + meo_data['index']
            },
            **attrs,
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True
        }
        nodes.append(node)
        node_index += 1
    
    # GEO Satellites
    geo_constellation = generate_geo_constellation()
    for geo_data in geo_constellation:
        attrs = get_node_attributes('GEO_SATELLITE', geo_data['index'], NUM_GEO_SATELLITES)
        
        node = {
            'id': generate_node_id(node_index, 'GEO_SATELLITE'),
            'nodeId': generate_node_id(node_index, 'GEO_SATELLITE'),
            'nodeName': f'GEO Satellite {geo_data["index"]+1}',
            'nodeType': 'GEO_SATELLITE',
            'position': geo_data['position'],
            'velocity': geo_data['velocity'],
            'orbit': {
                'semiMajorAxisKm': EARTH_RADIUS_KM + geo_data['altitude_km'],
                'eccentricity': 0.0001,
                'inclinationDeg': 0.0,
                'raanDeg': 0.0,
                'argumentOfPerigeeDeg': 0.0,
                'trueAnomalyDeg': geo_data['index'] * (360.0 / NUM_GEO_SATELLITES),
                'orbitalPeriodMin': round(geo_data['orbital_period'], 2)
            },
            'communication': {
                'frequencyGHz': 20.0,
                'bandwidthMHz': 750 + geo_data['index'] * 100,
                'transmitPowerDbW': 50.0 + geo_data['index'] * 1.0,
                'antennaGainDb': 40.0 + geo_data['index'] * 2.0,
                'beamWidthDeg': 1.0 + geo_data['index'] * 0.5,
                'maxRangeKm': 40000.0,
                'minElevationDeg': 10.0,
                'protocol': 'DVB-S2X',
                'ipAddress': f'10.3.{geo_data["index"]//256}.{geo_data["index"]%256}',
                'port': 8080 + geo_data['index']
            },
            **attrs,
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True,
            'propagationDelayMs': round(geo_data['altitude_km'] / SPEED_OF_LIGHT_KM_S, 2)
        }
        nodes.append(node)
        node_index += 1
    
    # Ground Stations
    gs_locations = generate_ground_stations()
    for i, location in enumerate(gs_locations):
        attrs = get_node_attributes('GROUND_STATION', i, NUM_GROUND_STATIONS)
        
        node = {
            'id': generate_node_id(node_index, 'GROUND_STATION'),
            'nodeId': generate_node_id(node_index, 'GROUND_STATION'),
            'nodeName': f'Ground Station {location["name"]}',
            'nodeType': 'GROUND_STATION',
            'position': {
                'latitude': location['lat'],
                'longitude': location['lon'],
                'altitude': location['alt']
            },
            'velocity': {'velocityX': 0.0, 'velocityY': 0.0, 'velocityZ': 0.0},
            'orbit': None,
            'communication': {
                'frequencyGHz': 12.5 + (i % 3) * 0.5,
                'bandwidthMHz': 200 + (i % 5) * 50,
                'transmitPowerDbW': 52.0 + (i % 2) * 1.0,
                'antennaGainDb': 50.0 + (i % 3) * 3.0,
                'beamWidthDeg': 1.0 + (i % 3) * 0.5,
                'maxRangeKm': 40000.0,
                'minElevationDeg': 10.0,
                'protocol': 'TCP/IP',
                'ipAddress': f'10.10.{i//256}.{i%256}',
                'port': 8080 + i
            },
            **attrs,
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True,
            'location': location['name'],
            'region': location['region'],
            'host': f'gs-{location["name"].lower().replace(" ", "-")}.aiprancs.local'
        }
        nodes.append(node)
        node_index += 1
    
    # Insert all nodes
    if nodes:
        result = nodes_collection.insert_many(nodes)
        print(f"✅ Created {len(result.inserted_ids)} nodes:")
        print(f"   - LEO satellites: {NUM_LEO_SATELLITES}")
        print(f"   - MEO satellites: {NUM_MEO_SATELLITES}")
        print(f"   - GEO satellites: {NUM_GEO_SATELLITES}")
        print(f"   - Ground stations: {NUM_GROUND_STATIONS}")
        return [str(id) for id in result.inserted_ids]
    return []


def calculate_distance_km(pos1: Dict, pos2: Dict) -> float:
    """Calculate distance in km using Haversine formula"""
    lat1 = math.radians(pos1['latitude'])
    lon1 = math.radians(pos1['longitude'])
    lat2 = math.radians(pos2['latitude'])
    lon2 = math.radians(pos2['longitude'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def create_terminals(db) -> List[str]:
    """
    Create terminals with training-optimized scenarios:
    - Easy: Short distances, good coverage
    - Medium: Medium distances, some constraints
    - Hard: Long distances, challenging paths
    
    Note: This function assumes data has already been cleared
    """
    terminals_collection = db['terminals']
    
    nodes_collection = db['nodes']
    ground_stations = list(nodes_collection.find(
        {'nodeType': 'GROUND_STATION'},
        {'nodeId': 1, 'position': 1, 'region': 1}
    ))
    satellites = list(nodes_collection.find(
        {'nodeType': {'$in': ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']}},
        {'nodeId': 1, 'nodeType': 1, 'position': 1}
    ))
    
    terminals = []
    
    # Training scenarios: Easy (0-9), Medium (10-19), Hard (20-29)
    service_types = ['VIDEO_STREAM', 'AUDIO_CALL', 'IMAGE_TRANSFER', 'TEXT_MESSAGE', 'FILE_TRANSFER']
    terminal_types = ['MOBILE', 'FIXED', 'VEHICLE', 'AIRCRAFT', 'MARITIME']
    
    if not ground_stations:
        print("⚠️  Warning: No ground stations found. Cannot create terminals.")
        return []
    
    if not satellites:
        print("⚠️  Warning: No satellites found. Terminals will be created without connections.")
    
    print(f"Creating {NUM_TERMINALS} terminals...")
    
    for i in range(NUM_TERMINALS):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{NUM_TERMINALS} terminals created...")
        scenario_type = 'EASY' if i < 10 else 'MEDIUM' if i < 20 else 'HARD'
        service_type = service_types[i % len(service_types)]
        terminal_type = terminal_types[i % len(terminal_types)]
        
        # Select source and destination based on scenario (deterministic)
        if scenario_type == 'EASY':
            # Same region, close ground stations
            source_idx = i % len(ground_stations)
            source_gs = ground_stations[source_idx]
            
            # Find closest ground station (deterministic search with limit)
            best_dest_idx = (source_idx + 1) % len(ground_stations)
            best_distance = calculate_distance_km(source_gs['position'], ground_stations[best_dest_idx]['position'])
            
            # Check next few stations to find closest (max 5 attempts)
            for offset in range(2, min(7, len(ground_stations))):
                candidate_idx = (source_idx + offset) % len(ground_stations)
                candidate_distance = calculate_distance_km(source_gs['position'], ground_stations[candidate_idx]['position'])
                if candidate_distance < best_distance and candidate_distance < 2000:
                    best_dest_idx = candidate_idx
                    best_distance = candidate_distance
            
            dest_gs = ground_stations[best_dest_idx]
            
        elif scenario_type == 'MEDIUM':
            # Different regions, medium distance
            source_idx = i % len(ground_stations)
            source_gs = ground_stations[source_idx]
            dest_gs = ground_stations[(source_idx + 10) % len(ground_stations)]
            
        else:  # HARD
            # Cross-continental, long distance
            source_idx = i % len(ground_stations)
            source_gs = ground_stations[source_idx]
            
            # Find farthest ground station (deterministic search with limit)
            best_dest_idx = (source_idx + 15) % len(ground_stations)
            best_distance = calculate_distance_km(source_gs['position'], ground_stations[best_dest_idx]['position'])
            
            # Check next few stations to find farthest (max 5 attempts)
            for offset in range(12, min(20, len(ground_stations))):
                candidate_idx = (source_idx + offset) % len(ground_stations)
                candidate_distance = calculate_distance_km(source_gs['position'], ground_stations[candidate_idx]['position'])
                if candidate_distance > best_distance and candidate_distance > 5000:
                    best_dest_idx = candidate_idx
                    best_distance = candidate_distance
            
            dest_gs = ground_stations[best_dest_idx]
        
        # Position terminal near source ground station
        source_pos = source_gs['position']
        offset_lat = (i % 5) * 0.1 - 0.2  # -0.2 to 0.2 degrees
        offset_lon = ((i // 5) % 5) * 0.1 - 0.2
        
        terminal_lat = source_pos['latitude'] + offset_lat
        terminal_lon = source_pos['longitude'] + offset_lon
        
        # Altitude based on type
        if terminal_type == 'AIRCRAFT':
            altitude = 10000 + (i % 3) * 1000
        elif terminal_type == 'MARITIME':
            altitude = (i % 3) * 10
        elif terminal_type == 'VEHICLE':
            altitude = (i % 5) * 200
        else:
            altitude = (i % 3) * 50
        
        # Connection status: 60% connected for training diversity
        is_connected = (i % 10) < 6
        connected_node = None
        connection_metrics = None
        
        if is_connected and satellites:
            # Select appropriate satellite based on terminal type
            preferred_types = {
                'AIRCRAFT': ['LEO_SATELLITE', 'MEO_SATELLITE'],
                'MOBILE': ['LEO_SATELLITE'],
                'FIXED': ['GEO_SATELLITE', 'MEO_SATELLITE'],
                'VEHICLE': ['LEO_SATELLITE'],
                'MARITIME': ['GEO_SATELLITE', 'LEO_SATELLITE']
            }
            
            preferred = preferred_types.get(terminal_type, ['LEO_SATELLITE'])
            suitable = [s for s in satellites if s['nodeType'] in preferred]
            if not suitable:
                suitable = satellites
            
            connected_node = suitable[i % len(suitable)]['nodeId']
            node_type = suitable[i % len(suitable)]['nodeType']
            
            # Deterministic connection metrics
            if node_type == 'LEO_SATELLITE':
                connection_metrics = {
                    'latencyMs': 30.0 + (i % 5) * 2.0,
                    'bandwidthMbps': 100.0 + (i % 3) * 20.0,
                    'packetLossRate': 0.003 + (i % 3) * 0.002,
                    'signalStrength': -70.0 + (i % 5) * 2.0,
                    'snrDb': 12.0 + (i % 4) * 2.0,
                    'jitterMs': 5.0 + (i % 3) * 2.0
                }
            elif node_type == 'MEO_SATELLITE':
                connection_metrics = {
                    'latencyMs': 90.0 + (i % 4) * 5.0,
                    'bandwidthMbps': 60.0 + (i % 3) * 15.0,
                    'packetLossRate': 0.002 + (i % 2) * 0.003,
                    'signalStrength': -75.0 + (i % 4) * 2.0,
                    'snrDb': 10.0 + (i % 3) * 2.0,
                    'jitterMs': 8.0 + (i % 2) * 2.0
                }
            else:  # GEO
                connection_metrics = {
                    'latencyMs': 260.0 + (i % 3) * 5.0,
                    'bandwidthMbps': 40.0 + (i % 3) * 10.0,
                    'packetLossRate': 0.0005 + (i % 2) * 0.0005,
                    'signalStrength': -80.0 + (i % 3) * 2.0,
                    'snrDb': 8.0 + (i % 2) * 2.0,
                    'jitterMs': 12.0 + (i % 2) * 1.0
                }
        
        qos = get_deterministic_qos(service_type, i)
        
        terminal = {
            'id': generate_terminal_id(i),
            'terminalId': generate_terminal_id(i),
            'terminalName': f'{terminal_type} Terminal {i+1} ({scenario_type})',
            'terminalType': terminal_type,
            'position': {
                'latitude': round(terminal_lat, 4),
                'longitude': round(terminal_lon, 4),
                'altitude': round(altitude, 2)
            },
            'status': 'connected' if is_connected else 'idle',
            'connectedNodeId': connected_node,
            'qosRequirements': qos,
            'metadata': {
                'description': f'{terminal_type} terminal - {scenario_type} scenario',
                'scenarioType': scenario_type,
                'serviceType': service_type,
                'mobility': terminal_type in ['MOBILE', 'VEHICLE', 'AIRCRAFT', 'MARITIME']
            },
            'lastUpdated': datetime.now().isoformat()
        }
        
        if connection_metrics:
            terminal['connectionMetrics'] = connection_metrics
        
        terminals.append(terminal)
    
    if terminals:
        result = terminals_collection.insert_many(terminals)
        print(f"✅ Created {len(result.inserted_ids)} terminals:")
        print(f"   - Easy scenarios: 10")
        print(f"   - Medium scenarios: 10")
        print(f"   - Hard scenarios: 10")
        print(f"   - Connected: {sum(1 for t in terminals if t['status'] == 'connected')}")
        print(f"   - Idle: {sum(1 for t in terminals if t['status'] == 'idle')}")
        return [str(id) for id in result.inserted_ids]
    return []


def create_indexes(db):
    """Create indexes for better query performance"""
    try:
        nodes_collection = db['nodes']
        nodes_collection.create_index('nodeId', unique=True)
        nodes_collection.create_index('nodeType')
        nodes_collection.create_index('isOperational')
        nodes_collection.create_index('lastUpdated')
        nodes_collection.create_index([('position.latitude', 1), ('position.longitude', 1)])
        nodes_collection.create_index('healthy')
        
        terminals_collection = db['terminals']
        terminals_collection.create_index('terminalId', unique=True)
        terminals_collection.create_index('terminalType')
        terminals_collection.create_index('status')
        terminals_collection.create_index('connectedNodeId')
        terminals_collection.create_index('lastUpdated')
        terminals_collection.create_index([('position.latitude', 1), ('position.longitude', 1)])
        terminals_collection.create_index('qosRequirements.serviceType')
        terminals_collection.create_index('metadata.scenarioType')
        
        print("✅ Created database indexes successfully")
    except Exception as e:
        print(f"⚠️  Index creation warning: {e}")


def print_summary_statistics(db):
    """Print summary statistics of the seeded data"""
    nodes_collection = db['nodes']
    terminals_collection = db['terminals']
    
    print("\n" + "="*60)
    print("DATABASE SUMMARY STATISTICS")
    print("="*60)
    
    total_nodes = nodes_collection.count_documents({})
    operational_nodes = nodes_collection.count_documents({'isOperational': True})
    
    print(f"\n📡 NODES: {total_nodes} total ({operational_nodes} operational)")
    for node_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE', 'GROUND_STATION']:
        count = nodes_collection.count_documents({'nodeType': node_type})
        if count > 0:
            print(f"   - {node_type}: {count}")
    
    total_terminals = terminals_collection.count_documents({})
    connected_terminals = terminals_collection.count_documents({'status': 'connected'})
    
    print(f"\n📱 TERMINALS: {total_terminals} total ({connected_terminals} connected)")
    
    for term_type in ['MOBILE', 'FIXED', 'VEHICLE', 'AIRCRAFT', 'MARITIME']:
        count = terminals_collection.count_documents({'terminalType': term_type})
        if count > 0:
            print(f"   - {term_type}: {count}")
    
    print(f"\n📊 SCENARIOS:")
    for scenario in ['EASY', 'MEDIUM', 'HARD']:
        count = terminals_collection.count_documents({'metadata.scenarioType': scenario})
        if count > 0:
            print(f"   - {scenario}: {count}")
    
    print(f"\n🔧 SERVICE TYPES:")
    for service in ['VIDEO_STREAM', 'AUDIO_CALL', 'FILE_TRANSFER', 'IMAGE_TRANSFER', 'TEXT_MESSAGE']:
        count = terminals_collection.count_documents({'qosRequirements.serviceType': service})
        if count > 0:
            print(f"   - {service}: {count}")
    
    print("\n" + "="*60)


def clear_existing_data(db, force: bool = False):
    """
    Clear all existing nodes and terminals from database
    Returns True if data was cleared, False if no data existed
    """
    nodes_collection = db['nodes']
    terminals_collection = db['terminals']
    
    node_count = nodes_collection.count_documents({})
    terminal_count = terminals_collection.count_documents({})
    
    if node_count == 0 and terminal_count == 0:
        print("ℹ️  No existing data found. Starting fresh seed.")
        return False
    
    print(f"\n⚠️  Existing data found:")
    print(f"   - Nodes: {node_count}")
    print(f"   - Terminals: {terminal_count}")
    
    if force:
        print("\n🗑️  Clearing existing data...")
        nodes_collection.delete_many({})
        terminals_collection.delete_many({})
        print("✅ All existing data cleared successfully")
        return True
    else:
        print("\n🔄 Auto-clearing existing data for fresh seed...")
        nodes_collection.delete_many({})
        terminals_collection.delete_many({})
        print("✅ All existing data cleared successfully")
        return True


def main(force_clear: bool = True):
    """
    Main function to seed database with deterministic training data
    Automatically clears existing data before seeding
    
    Args:
        force_clear: If True, automatically clear existing data without prompt
    """
    print("="*60)
    print("🌱 SEEDING SAGIN DATABASE WITH DETERMINISTIC TRAINING DATA")
    print("="*60)
    print(f"Configuration:")
    print(f"   - LEO Satellites: {NUM_LEO_SATELLITES}")
    print(f"   - MEO Satellites: {NUM_MEO_SATELLITES}")
    print(f"   - GEO Satellites: {NUM_GEO_SATELLITES}")
    print(f"   - Ground Stations: {NUM_GROUND_STATIONS}")
    print(f"   - Terminals: {NUM_TERMINALS}")
    print(f"   - Deterministic Seed: {DETERMINISTIC_SEED}")
    print("="*60)
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        # Test connection
        client.server_info()
        print("✅ Connected to MongoDB successfully")
        
        # Clear existing data if any
        data_existed = clear_existing_data(db, force=force_clear)
        
        if data_existed:
            print("\n" + "-" * 60)
        
        # Create indexes first (will recreate if needed)
        print("\n📑 Creating/updating database indexes...")
        create_indexes(db)
        
        # Create sample nodes
        print("\n📡 Creating deterministic nodes...")
        print("-" * 60)
        node_ids = create_nodes(db)
        
        # Create sample terminals
        print("\n📱 Creating training-optimized terminals...")
        print("-" * 60)
        terminal_ids = create_terminals(db)
        
        # Print summary
        print_summary_statistics(db)
        
        print("\n✅ Database seeding completed successfully!")
        print(f"   MongoDB URI: {MONGODB_URI}")
        print(f"   Database: {DB_NAME}")
        print(f"\n💡 This data is optimized for RL training:")
        print(f"   - All values are deterministic and reproducible")
        print(f"   - Scenarios range from Easy to Hard")
        print(f"   - Nodes have varied but predictable attributes")
        print(f"   - Terminals cover diverse service types and distances")
        print(f"   - Model can be trained and evaluated on this same data")
        print(f"\n🔄 To reseed: Run this script again (will auto-clear existing data)")
        
        client.close()
        
    except Exception as e:
        print(f"\n❌ Error seeding database: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    main()