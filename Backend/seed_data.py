#!/usr/bin/env python3
"""
Script to seed database with realistic sample data for SAGIN network simulation
Adds sample nodes (satellites and ground stations) and terminals with accurate physics
"""
from pymongo import MongoClient
from datetime import datetime
import os
import random
import math

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://admin:password@localhost:27017/aiprancs?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'aiprancs')

# Physical constants
EARTH_RADIUS_KM = 6371.0  # km
EARTH_MU = 398600.4418  # Earth's gravitational parameter (km³/s²)
SPEED_OF_LIGHT = 299792.458  # km/s

def calculate_orbital_velocity(altitude_km):
    """
    Calculate orbital velocity using vis-viva equation
    v = sqrt(μ/r) where r = Earth radius + altitude
    Returns velocity in km/s
    """
    r = EARTH_RADIUS_KM + altitude_km
    return math.sqrt(EARTH_MU / r)

def calculate_orbital_period(altitude_km):
    """
    Calculate orbital period in minutes
    T = 2π * sqrt(r³/μ)
    """
    r = EARTH_RADIUS_KM + altitude_km
    period_seconds = 2 * math.pi * math.sqrt(r**3 / EARTH_MU)
    return period_seconds / 60  # Convert to minutes

def generate_terminal_id(index: int) -> str:
    """Generate a unique terminal ID"""
    timestamp = int(datetime.now().timestamp() * 1000)
    return f"TERM-{timestamp}-{index:04d}"

def generate_node_id(index: int, node_type: str) -> str:
    """Generate a unique node ID"""
    prefix = {
        'LEO_SATELLITE': 'LEO',
        'MEO_SATELLITE': 'MEO', 
        'GEO_SATELLITE': 'GEO',
        'GROUND_STATION': 'GS'
    }.get(node_type, 'NODE')
    return f"{prefix}-{index:03d}"

def generate_realistic_qos(service_type):
    """
    Generate realistic QoS requirements based on service type
    Based on ITU-T Y.1541 and 3GPP QoS standards
    """
    qos_profiles = {
        'VIDEO_STREAM': {
            'maxLatencyMs': round(random.uniform(100, 300), 2),
            'minBandwidthMbps': round(random.uniform(5, 25), 2),
            'maxLossRate': round(random.uniform(0.001, 0.01), 4),  # 0.1-1%
            'priority': random.randint(6, 8)
        },
        'AUDIO_CALL': {
            'maxLatencyMs': round(random.uniform(50, 150), 2),
            'minBandwidthMbps': round(random.uniform(0.064, 1), 3),  # 64 kbps - 1 Mbps
            'maxLossRate': round(random.uniform(0.001, 0.01), 4),
            'priority': random.randint(8, 10)
        },
        'IMAGE_TRANSFER': {
            'maxLatencyMs': round(random.uniform(200, 1000), 2),
            'minBandwidthMbps': round(random.uniform(1, 10), 2),
            'maxLossRate': round(random.uniform(0.001, 0.02), 4),
            'priority': random.randint(4, 6)
        },
        'TEXT_MESSAGE': {
            'maxLatencyMs': round(random.uniform(500, 2000), 2),
            'minBandwidthMbps': round(random.uniform(0.01, 0.1), 3),
            'maxLossRate': round(random.uniform(0.01, 0.05), 4),
            'priority': random.randint(2, 4)
        },
        'FILE_TRANSFER': {
            'maxLatencyMs': round(random.uniform(1000, 5000), 2),
            'minBandwidthMbps': round(random.uniform(10, 100), 2),
            'maxLossRate': round(random.uniform(0.001, 0.03), 4),
            'priority': random.randint(3, 5)
        }
    }
    return qos_profiles.get(service_type, qos_profiles['FILE_TRANSFER'])

def generate_random_qos():
    """Generate random QoS with realistic service type"""
    service_types = ['VIDEO_STREAM', 'AUDIO_CALL', 'IMAGE_TRANSFER', 'TEXT_MESSAGE', 'FILE_TRANSFER']
    service_type = random.choice(service_types)
    qos = generate_realistic_qos(service_type)
    qos['serviceType'] = service_type
    return qos

def generate_leo_position(index: int, total: int, plane: int = 0):
    """
    Generate LEO satellite position with inclined orbit
    Using walker-delta constellation pattern (similar to Starlink)
    """
    # Distribute satellites in multiple orbital planes
    num_planes = min(6, total)  # 6 orbital planes (tăng từ 4) để coverage tốt hơn
    sats_per_plane = total // num_planes
    
    current_plane = index // sats_per_plane
    sat_in_plane = index % sats_per_plane
    
    # RAAN spacing between planes
    raan = current_plane * (360 / num_planes)
    
    # True anomaly spacing within plane
    true_anomaly = sat_in_plane * (360 / sats_per_plane)
    
    # Inclined orbit (like Starlink: 53°)
    inclination = 53.0
    altitude = 550  # km
    
    # Calculate position from orbital elements (simplified)
    # For demonstration, we'll use a simplified circular orbit
    lon = raan + true_anomaly
    lat = math.sin(math.radians(true_anomaly)) * inclination
    
    return {
        'latitude': round(lat, 4),
        'longitude': round(lon % 360 - 180, 4),  # -180 to 180
        'altitude': altitude * 1000  # Convert to meters
    }

def generate_leo_velocity(index: int, total: int, altitude_km: float):
    """
    Generate realistic LEO satellite velocity vector
    LEO at 550km has velocity ~7.6 km/s
    """
    v_orbital = calculate_orbital_velocity(altitude_km)  # km/s
    v_ms = v_orbital * 1000  # Convert to m/s
    
    # Distribute satellites in orbital planes
    num_planes = min(6, total)  # 6 planes (tăng từ 4)
    sats_per_plane = total // num_planes
    current_plane = index // sats_per_plane
    sat_in_plane = index % sats_per_plane
    
    # Velocity direction based on position in orbit
    angle = (sat_in_plane * 360 / sats_per_plane) * math.pi / 180
    raan_angle = (current_plane * 360 / num_planes) * math.pi / 180
    
    # Velocity components (simplified for circular orbit)
    vx = v_ms * math.cos(angle) * math.cos(raan_angle)
    vy = v_ms * math.cos(angle) * math.sin(raan_angle)
    vz = v_ms * math.sin(angle) * math.sin(math.radians(53.0))  # Inclination component
    
    return {
        'velocityX': round(vx, 3),
        'velocityY': round(vy, 3),
        'velocityZ': round(vz, 3)
    }

def generate_meo_position(index: int, total: int):
    """
    Generate MEO satellite position (like O3b constellation)
    Typically at ~8,000 km or ~20,000 km altitude
    """
    longitude = ((index * 360 / total) % 360) - 180
    # MEO often uses inclined orbits for better coverage
    inclination = 70.0  # degrees
    true_anomaly = (index * 360 / total) % 360
    latitude = math.sin(math.radians(true_anomaly)) * inclination
    altitude = 8000  # km (O3b uses ~8,000 km, GPS uses ~20,000 km)
    
    return {
        'latitude': round(latitude, 4),
        'longitude': round(longitude, 4),
        'altitude': altitude * 1000
    }

def generate_meo_velocity(index: int, total: int, altitude_km: float):
    """Generate realistic MEO satellite velocity"""
    v_orbital = calculate_orbital_velocity(altitude_km)  # km/s
    v_ms = v_orbital * 1000  # Convert to m/s (~3.9 km/s for 8000km altitude)
    
    angle = (index * 360 / total) * math.pi / 180
    
    # Velocity perpendicular to radius vector
    vx = v_ms * math.cos(angle)
    vy = v_ms * math.sin(angle)
    vz = v_ms * 0.3  # Small z-component due to inclination
    
    return {
        'velocityX': round(vx, 3),
        'velocityY': round(vy, 3),
        'velocityZ': round(vz, 3)
    }

def generate_geo_position(index: int, total: int):
    """
    Generate GEO satellite position
    Geostationary orbit at 35,786 km altitude, 0° inclination
    """
    # Distribute GEO satellites along equator
    longitude = -180 + (index * 360 / total)
    latitude = 0  # Geostationary at equator
    altitude = 35786  # km
    
    return {
        'latitude': latitude,
        'longitude': round(longitude, 4),
        'altitude': altitude * 1000
    }

def generate_geo_velocity():
    """
    GEO satellite velocity relative to Earth (appears stationary)
    Actual velocity is ~3.07 km/s but matches Earth rotation
    """
    # In Earth-centered, Earth-fixed (ECEF) frame, GEO appears stationary
    return {
        'velocityX': 0.0,
        'velocityY': 0.0,
        'velocityZ': 0.0
    }

def generate_ground_station_position(index: int):
    """Generate ground station position at major locations - tăng coverage"""
    # Real ground station locations - thêm nhiều stations hơn để có coverage tốt
    major_stations = [
        # North America
        {'lat': 40.7128, 'lon': -74.0060, 'name': 'New York', 'alt': 10},
        {'lat': 34.0522, 'lon': -118.2437, 'name': 'Los Angeles', 'alt': 100},
        {'lat': 41.8781, 'lon': -87.6298, 'name': 'Chicago', 'alt': 180},
        {'lat': 29.7604, 'lon': -95.3698, 'name': 'Houston', 'alt': 15},
        {'lat': 45.5017, 'lon': -73.5673, 'name': 'Montreal', 'alt': 36},
        # Europe
        {'lat': 51.5074, 'lon': -0.1278, 'name': 'London', 'alt': 15},
        {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris', 'alt': 35},
        {'lat': 52.5200, 'lon': 13.4050, 'name': 'Berlin', 'alt': 34},
        {'lat': 55.7558, 'lon': 37.6173, 'name': 'Moscow', 'alt': 156},
        {'lat': 41.9028, 'lon': 12.4964, 'name': 'Rome', 'alt': 57},
        # Asia
        {'lat': 35.6762, 'lon': 139.6503, 'name': 'Tokyo', 'alt': 40},
        {'lat': 22.3193, 'lon': 114.1694, 'name': 'Hong Kong', 'alt': 5},
        {'lat': 1.3521, 'lon': 103.8198, 'name': 'Singapore', 'alt': 15},
        {'lat': 28.6139, 'lon': 77.2090, 'name': 'New Delhi', 'alt': 216},
        {'lat': 31.2304, 'lon': 121.4737, 'name': 'Shanghai', 'alt': 4},
        # Oceania & Others
        {'lat': -33.8688, 'lon': 151.2093, 'name': 'Sydney', 'alt': 25},
        {'lat': -37.8136, 'lon': 144.9631, 'name': 'Melbourne', 'alt': 31},
        # South America
        {'lat': -23.5505, 'lon': -46.6333, 'name': 'Sao Paulo', 'alt': 760},
        {'lat': -34.6037, 'lon': -58.3816, 'name': 'Buenos Aires', 'alt': 25},
        # Africa
        {'lat': -1.2921, 'lon': 36.8219, 'name': 'Nairobi', 'alt': 1795},
        {'lat': -26.2041, 'lon': 28.0473, 'name': 'Johannesburg', 'alt': 1753},
    ]
    
    station = major_stations[index % len(major_stations)]
    
    # Add small random offset for variation (giảm offset để stations gần nhau hơn)
    lat_offset = random.uniform(-0.3, 0.3)  # Giảm từ 0.5 xuống 0.3
    lon_offset = random.uniform(-0.3, 0.3)
    
    return {
        'latitude': round(station['lat'] + lat_offset, 4),
        'longitude': round(station['lon'] + lon_offset, 4),
        'altitude': station['alt'] + random.uniform(-5, 5),  # meters
        'name': station['name']
    }

def calculate_propagation_delay(distance_km):
    """Calculate signal propagation delay in ms"""
    delay_s = distance_km / SPEED_OF_LIGHT
    return round(delay_s * 1000, 2)  # Convert to ms

def create_sample_nodes(db):
    """Create sample nodes (satellites and ground stations) with realistic parameters"""
    nodes_collection = db['nodes']
    
    # Clear existing nodes
    nodes_collection.delete_many({})
    
    nodes = []
    
    # ===== LEO Satellites (30 satellites in 6 orbital planes for 80% coverage) =====
    num_leo = 30  # Tăng từ 12 lên 30 để đạt 80% success rate
    leo_altitude = 550  # km
    leo_velocity_orbital = calculate_orbital_velocity(leo_altitude)
    leo_period = calculate_orbital_period(leo_altitude)
    
    print(f"LEO orbital velocity: {leo_velocity_orbital:.2f} km/s ({leo_velocity_orbital*3600:.0f} km/h)")
    print(f"LEO orbital period: {leo_period:.1f} minutes")
    
    for i in range(num_leo):
        position = generate_leo_position(i, num_leo)
        velocity = generate_leo_velocity(i, num_leo, leo_altitude)
        
        # Orbital elements - 6 planes cho 30 satellites
        num_planes = 6  # Tăng từ 4 lên 6
        sats_per_plane = num_leo // num_planes  # 5 sats per plane
        current_plane = i // sats_per_plane
        sat_in_plane = i % sats_per_plane
        
        node = {
            'id': generate_node_id(i, 'LEO_SATELLITE'),
            'nodeId': generate_node_id(i, 'LEO_SATELLITE'),
            'nodeName': f'LEO Satellite {i+1} (Plane {current_plane+1})',
            'nodeType': 'LEO_SATELLITE',
            'position': position,
            'velocity': velocity,
            'orbit': {
                'semiMajorAxisKm': EARTH_RADIUS_KM + leo_altitude,
                'eccentricity': 0.0001,  # Nearly circular
                'inclinationDeg': 53.0,  # Starlink-like inclination
                'raanDeg': current_plane * (360 / num_planes),
                'argumentOfPerigeeDeg': 0.0,
                'trueAnomalyDeg': sat_in_plane * (360 / sats_per_plane),
                'orbitalPeriodMin': round(leo_period, 2)
            },
            'communication': {
                'frequencyGHz': round(random.uniform(10.7, 12.75), 2),  # Ku-band downlink
                'bandwidthMHz': round(random.uniform(250, 500), 1),  # Modern LEO bandwidth
                'transmitPowerDbW': round(random.uniform(37, 43), 1),  # 5-20 kW
                'antennaGainDb': round(random.uniform(28, 32), 1),
                'beamWidthDeg': round(random.uniform(8, 15), 1),
                'maxRangeKm': 2500.0,
                'minElevationDeg': 25.0,  # Typical minimum for LEO
                'protocol': 'DVB-S2X',
                'ipAddress': f'10.1.{i//256}.{i%256}',
                'port': 8080 + i
            },
            'isOperational': random.choice([True, True, True, True, False]),  # 80% operational
            'batteryChargePercent': round(random.uniform(75, 100), 1),
            'nodeProcessingDelayMs': round(random.uniform(2, 8), 2),
            'packetLossRate': round(random.uniform(0.0001, 0.005), 4),
            'resourceUtilization': round(random.uniform(20, 75), 1),
            'packetBufferCapacity': 5000,
            'currentPacketCount': random.randint(0, 3000),
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True,
            'maxDopplerShiftKhz': round(leo_velocity_orbital * (12.5 / SPEED_OF_LIGHT) * 1e6 / 1000, 2)  # Doppler at 12.5 GHz
        }
        nodes.append(node)
    
    # ===== MEO Satellites (8 satellites for better coverage) =====
    num_meo = 8  # Tăng từ 6 lên 8
    meo_altitude = 8000  # km (O3b altitude)
    meo_velocity_orbital = calculate_orbital_velocity(meo_altitude)
    meo_period = calculate_orbital_period(meo_altitude)
    
    print(f"\nMEO orbital velocity: {meo_velocity_orbital:.2f} km/s")
    print(f"MEO orbital period: {meo_period:.1f} minutes")
    
    for i in range(num_meo):
        position = generate_meo_position(i, num_meo)
        velocity = generate_meo_velocity(i, num_meo, meo_altitude)
        
        node = {
            'id': generate_node_id(i, 'MEO_SATELLITE'),
            'nodeId': generate_node_id(i, 'MEO_SATELLITE'),
            'nodeName': f'MEO Satellite {i+1}',
            'nodeType': 'MEO_SATELLITE',
            'position': position,
            'velocity': velocity,
            'orbit': {
                'semiMajorAxisKm': EARTH_RADIUS_KM + meo_altitude,
                'eccentricity': 0.0001,
                'inclinationDeg': 70.0,  # O3b-like inclination
                'raanDeg': i * (360 / num_meo),
                'argumentOfPerigeeDeg': 0.0,
                'trueAnomalyDeg': i * (360 / num_meo),
                'orbitalPeriodMin': round(meo_period, 2)
            },
            'communication': {
                'frequencyGHz': round(random.uniform(17.7, 20.2), 2),  # Ka-band
                'bandwidthMHz': round(random.uniform(400, 800), 1),
                'transmitPowerDbW': round(random.uniform(43, 47), 1),  # 20-50 kW
                'antennaGainDb': round(random.uniform(33, 38), 1),
                'beamWidthDeg': round(random.uniform(4, 8), 1),
                'maxRangeKm': 10000.0,
                'minElevationDeg': 20.0,
                'protocol': 'DVB-S2X',
                'ipAddress': f'10.2.{i//256}.{i%256}',
                'port': 8080 + i
            },
            'isOperational': True,
            'batteryChargePercent': round(random.uniform(80, 98), 1),
            'nodeProcessingDelayMs': round(random.uniform(5, 15), 2),
            'packetLossRate': round(random.uniform(0.0001, 0.002), 4),
            'resourceUtilization': round(random.uniform(30, 65), 1),
            'packetBufferCapacity': 10000,
            'currentPacketCount': random.randint(0, 6000),
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True,
            'maxDopplerShiftKhz': round(meo_velocity_orbital * (18.0 / SPEED_OF_LIGHT) * 1e6 / 1000, 2)
        }
        nodes.append(node)
    
    # ===== GEO Satellites (3 satellites) =====
    num_geo = 3
    geo_altitude = 35786  # km
    geo_period = calculate_orbital_period(geo_altitude)
    
    print(f"\nGEO orbital period: {geo_period:.1f} minutes (~24 hours)")
    
    for i in range(num_geo):
        position = generate_geo_position(i, num_geo)
        velocity = generate_geo_velocity()
        
        # One-way propagation delay from GEO to ground
        one_way_delay = calculate_propagation_delay(geo_altitude)
        
        node = {
            'id': generate_node_id(i, 'GEO_SATELLITE'),
            'nodeId': generate_node_id(i, 'GEO_SATELLITE'),
            'nodeName': f'GEO Satellite {i+1}',
            'nodeType': 'GEO_SATELLITE',
            'position': position,
            'velocity': velocity,  # Stationary relative to Earth
            'orbit': {
                'semiMajorAxisKm': EARTH_RADIUS_KM + geo_altitude,
                'eccentricity': 0.0001,
                'inclinationDeg': 0.0,  # Equatorial
                'raanDeg': 0.0,
                'argumentOfPerigeeDeg': 0.0,
                'trueAnomalyDeg': i * (360 / num_geo),
                'orbitalPeriodMin': round(geo_period, 2)
            },
            'communication': {
                'frequencyGHz': round(random.uniform(19.7, 20.2), 2),  # Ka-band downlink
                'bandwidthMHz': round(random.uniform(500, 1000), 1),  # GEO typically has more bandwidth
                'transmitPowerDbW': round(random.uniform(47, 52), 1),  # 50-150 kW
                'antennaGainDb': round(random.uniform(38, 45), 1),
                'beamWidthDeg': round(random.uniform(0.5, 3), 1),  # Narrow beams
                'maxRangeKm': 40000.0,
                'minElevationDeg': 10.0,  # Lower elevation OK for GEO
                'protocol': 'DVB-S2X',
                'ipAddress': f'10.3.{i//256}.{i%256}',
                'port': 8080 + i
            },
            'isOperational': True,
            'batteryChargePercent': round(random.uniform(85, 100), 1),
            'nodeProcessingDelayMs': round(random.uniform(8, 20), 2),
            'packetLossRate': round(random.uniform(0.00001, 0.0005), 5),  # Very low loss
            'resourceUtilization': round(random.uniform(40, 80), 1),
            'packetBufferCapacity': 20000,
            'currentPacketCount': random.randint(0, 12000),
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True,
            'propagationDelayMs': round(one_way_delay, 2),  # One-way delay
            'maxDopplerShiftKhz': 0.0  # Negligible for GEO
        }
        nodes.append(node)
    
    # ===== Ground Stations (40 stations for 80% success rate) =====
    num_ground = 40  # Tăng từ 20 lên 40 để đạt 80% success rate
    weather_conditions = ['CLEAR', 'CLEAR', 'CLEAR', 'CLEAR', 'CLEAR', 
                          'LIGHT_RAIN', 'RAIN', 'CLOUDY', 'FOG']
    
    for i in range(num_ground):
        position_data = generate_ground_station_position(i)
        
        node = {
            'id': generate_node_id(i, 'GROUND_STATION'),
            'nodeId': generate_node_id(i, 'GROUND_STATION'),
            'nodeName': f'Ground Station {position_data["name"]}',
            'nodeType': 'GROUND_STATION',
            'position': {
                'latitude': position_data['latitude'],
                'longitude': position_data['longitude'],
                'altitude': position_data['altitude']
            },
            'velocity': {
                'velocityX': 0.0,
                'velocityY': 0.0,
                'velocityZ': 0.0
            },
            'orbit': None,
            'communication': {
                'frequencyGHz': round(random.uniform(10.7, 14.5), 2),  # Ku/Ka-band uplink
                'bandwidthMHz': round(random.uniform(100, 500), 1),
                'transmitPowerDbW': round(random.uniform(50, 55), 1),  # 100-300 kW
                'antennaGainDb': round(random.uniform(45, 60), 1),  # Large parabolic antennas
                'beamWidthDeg': round(random.uniform(0.2, 2.0), 2),
                'maxRangeKm': 40000.0,  # Can reach GEO
                'minElevationDeg': 10.0,
                'protocol': 'TCP/IP',
                'ipAddress': f'10.10.{i//256}.{i%256}',
                'port': 8080 + i
            },
            'isOperational': True,
            'batteryChargePercent': 100.0,  # Mains powered, backup battery
            'nodeProcessingDelayMs': round(random.uniform(0.5, 3), 2),
            'packetLossRate': round(random.uniform(0.00001, 0.0005), 5),
            'resourceUtilization': round(random.uniform(15, 55), 1),
            'packetBufferCapacity': 10000,
            'currentPacketCount': random.randint(0, 4000),
            'weather': random.choice(weather_conditions),  # Only ground stations have weather
            'lastUpdated': datetime.now().isoformat(),
            'healthy': True,
            'location': position_data['name'],
            'host': f'gs-{position_data["name"].lower().replace(" ", "-")}.aiprancs.local',
            'port': 8080 + i
        }
        nodes.append(node)
    
    # Insert all nodes
    if nodes:
        result = nodes_collection.insert_many(nodes)
        print(f"\n✅ Created {len(result.inserted_ids)} nodes:")
        print(f"   - LEO satellites: {num_leo}")
        print(f"   - MEO satellites: {num_meo}")
        print(f"   - GEO satellites: {num_geo}")
        print(f"   - Ground stations: {num_ground}")
        return [str(id) for id in result.inserted_ids]
    return []

def create_sample_terminals(db, count=30):
    """Create sample terminals with realistic positions and QoS requirements"""
    terminals_collection = db['terminals']
    
    # Clear existing terminals
    terminals_collection.delete_many({})
    
    # Get available nodes for connection
    nodes_collection = db['nodes']
    available_nodes = list(nodes_collection.find({
        'isOperational': True,
        'nodeType': {'$in': ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE']}
    }, {'nodeId': 1, 'nodeType': 1, 'position': 1}))
    
    terminals = []
    terminal_types = ['MOBILE', 'FIXED', 'VEHICLE', 'AIRCRAFT', 'MARITIME']
    
    # Service type distribution (realistic usage)
    service_distributions = {
        'MOBILE': ['VIDEO_STREAM', 'AUDIO_CALL', 'TEXT_MESSAGE', 'IMAGE_TRANSFER'],
        'FIXED': ['VIDEO_STREAM', 'FILE_TRANSFER', 'AUDIO_CALL'],
        'VEHICLE': ['TEXT_MESSAGE', 'AUDIO_CALL', 'IMAGE_TRANSFER'],
        'AIRCRAFT': ['AUDIO_CALL', 'TEXT_MESSAGE', 'VIDEO_STREAM'],
        'MARITIME': ['TEXT_MESSAGE', 'AUDIO_CALL', 'IMAGE_TRANSFER']
    }
    
    # Geographic regions with realistic distribution
    regions = [
        {'minLat': 30, 'maxLat': 50, 'minLon': -125, 'maxLon': -70, 'name': 'North America', 'terminals': 8},
        {'minLat': 40, 'maxLat': 60, 'minLon': -10, 'maxLon': 30, 'name': 'Europe', 'terminals': 7},
        {'minLat': 20, 'maxLat': 45, 'minLon': 100, 'maxLon': 145, 'name': 'East Asia', 'terminals': 6},
        {'minLat': -40, 'maxLat': -10, 'minLon': 110, 'maxLon': 155, 'name': 'Australia', 'terminals': 4},
        {'minLat': -35, 'maxLat': 35, 'minLon': -20, 'maxLon': 50, 'name': 'Africa', 'terminals': 3},
        {'minLat': -55, 'maxLat': 10, 'minLon': -80, 'maxLon': -35, 'name': 'South America', 'terminals': 2},
    ]
    
    # Calculate how many terminals should be connected
    connected_count = 0
    max_connected = min(int(count * 0.6), len(available_nodes) * 3)  # 60% connected, max 3 per node
    
    terminal_index = 0
    
    # Get ground stations để đảm bảo terminals gần ground stations
    ground_stations = list(nodes_collection.find(
        {'nodeType': 'GROUND_STATION', 'isOperational': True},
        {'nodeId': 1, 'position': 1}
    ))
    
    # Helper function để tính khoảng cách (Haversine)
    def calculate_distance_km(pos1, pos2):
        """Calculate distance in km between two positions"""
        import math
        lat1 = math.radians(pos1['latitude'])
        lon1 = math.radians(pos1['longitude'])
        lat2 = math.radians(pos2['latitude'])
        lon2 = math.radians(pos2['longitude'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        R = 6371  # Earth radius in km
        return R * c
    
    # Helper function để tìm ground stations trong phạm vi
    def find_nearby_ground_stations(lat, lon, max_range_km=100):
        """Tìm ground stations trong phạm vi max_range_km"""
        nearby = []
        terminal_pos = {'latitude': lat, 'longitude': lon}
        for gs in ground_stations:
            if gs.get('position'):
                distance = calculate_distance_km(terminal_pos, gs['position'])
                if distance <= max_range_km:
                    nearby.append((gs, distance))
        return nearby
    
    # Phân bố thực tế hơn: 20% có 1 GS, 60% có 2 GS, 20% có 3 GS trong phạm vi
    # Giảm target 3 GS để tránh warnings
    terminal_gs_distribution = []
    for i in range(count):
        rand = random.random()
        if rand < 0.2:
            terminal_gs_distribution.append(1)  # 1 GS - 20%
        elif rand < 0.8:
            terminal_gs_distribution.append(2)  # 2 GS - 60%
        else:
            terminal_gs_distribution.append(3)  # 3 GS - 20%
    
    for region in regions:
        num_terminals = region['terminals']
        
        for i in range(num_terminals):
            if terminal_index >= count:
                break
                
            terminal_type = random.choice(terminal_types)
            
            # Tìm ground stations trong region
            suitable_stations = [
                gs for gs in ground_stations
                if gs.get('position') and
                   region['minLat'] - 10 <= gs['position'].get('latitude', 0) <= region['maxLat'] + 10 and
                   region['minLon'] - 10 <= gs['position'].get('longitude', 0) <= region['maxLon'] + 10
            ]
            if not suitable_stations:
                suitable_stations = [gs for gs in ground_stations if gs.get('position')]
            
            if not suitable_stations:
                # Nếu không có GS trong region, tìm GS gần nhất
                all_gs = [gs for gs in ground_stations if gs.get('position')]
                if all_gs:
                    # Tìm GS gần nhất với center của region
                    region_center_lat = (region['minLat'] + region['maxLat']) / 2
                    region_center_lon = (region['minLon'] + region['maxLon']) / 2
                    region_center = {'latitude': region_center_lat, 'longitude': region_center_lon}
                    
                    closest_gs = min(all_gs, key=lambda gs: calculate_distance_km(region_center, gs['position']))
                    base_lat = closest_gs['position']['latitude']
                    base_lon = closest_gs['position']['longitude']
                    
                    # Tạo terminal gần GS này (trong 50-80km)
                    lat_offset = random.uniform(0.45, 0.72)  # ~50-80km
                    lon_offset = random.uniform(0.45, 0.72) / math.cos(math.radians(base_lat))
                    
                    lat = base_lat + random.choice([-1, 1]) * lat_offset
                    lon = base_lon + random.choice([-1, 1]) * lon_offset
                    
                    # Đảm bảo trong region bounds
                    lat = max(region['minLat'], min(region['maxLat'], lat))
                    lon = max(region['minLon'], min(region['maxLon'], lon))
                else:
                    # Fallback cuối cùng: random trong region
                    lat = random.uniform(region['minLat'], region['maxLat'])
                    lon = random.uniform(region['minLon'], region['maxLon'])
            else:
                # Chọn số lượng GS mong muốn cho terminal này
                target_gs_count = terminal_gs_distribution[terminal_index] if terminal_index < len(terminal_gs_distribution) else 2
                
                # Tìm vị trí sao cho có đủ số lượng GS trong phạm vi 50-100km
                max_attempts = 150  # Tăng số lần thử
                lat, lon = None, None
                nearby_count = 0
                best_position = None
                best_count = 0
                
                # Thử nhiều chiến lược khác nhau
                strategies = [
                    # Strategy 1: Gần một GS (dễ có 1-2 GS)
                    lambda: (random.choice(suitable_stations), 0.45, 0.72),  # 50-80km
                    # Strategy 2: Trung bình (có thể có 2-3 GS)
                    lambda: (random.choice(suitable_stations), 0.6, 0.9),   # 65-100km
                    # Strategy 3: Giữa 2 GS (dễ có 2 GS)
                    lambda: (random.choice(suitable_stations[:min(2, len(suitable_stations))]), 0.5, 0.8),
                ]
                
                for attempt in range(max_attempts):
                    # Chọn strategy dựa trên target
                    if target_gs_count == 1:
                        strategy_idx = 0  # Gần một GS
                    elif target_gs_count == 2:
                        strategy_idx = random.choice([0, 2])  # Gần một GS hoặc giữa 2 GS
                    else:
                        strategy_idx = random.choice([1, 2])  # Trung bình hoặc giữa 2 GS
                    
                    base_station, min_offset, max_offset = strategies[strategy_idx]()
                    base_lat = base_station['position']['latitude']
                    base_lon = base_station['position']['longitude']
                    
                    # Tạo terminal trong vòng offset của ground station
                    lat_offset = random.uniform(min_offset, max_offset)
                    lon_offset = random.uniform(min_offset, max_offset) / max(math.cos(math.radians(base_lat)), 0.1)
                    
                    # Random direction (có thể thử nhiều hướng)
                    angle = random.uniform(0, 2 * math.pi)
                    lat = base_lat + lat_offset * math.cos(angle)
                    lon = base_lon + lon_offset * math.sin(angle)
                    
                    # Đảm bảo trong region bounds
                    lat = max(region['minLat'], min(region['maxLat'], lat))
                    lon = max(region['minLon'], min(region['maxLon'], lon))
                    
                    # Kiểm tra số lượng GS trong phạm vi
                    nearby_gs = find_nearby_ground_stations(lat, lon, max_range_km=100)
                    nearby_count = len(nearby_gs)
                    
                    # Lưu vị trí tốt nhất (ưu tiên đủ target, sau đó là nhiều nhất)
                    if nearby_count >= target_gs_count:
                        # Tìm được đủ target, chấp nhận ngay
                        break
                    elif nearby_count >= 1:
                        # Lưu vị trí tốt nhất (có ít nhất 1 GS)
                        if best_count == 0 or (nearby_count >= best_count and nearby_count < target_gs_count):
                            best_position = (lat, lon)
                            best_count = nearby_count
                
                # Nếu không tìm được vị trí với đủ số lượng GS, dùng vị trí tốt nhất
                if nearby_count < target_gs_count:
                    if best_position and best_count >= 1:
                        lat, lon = best_position
                        nearby_count = best_count
                        # Chỉ warning nếu target > 1 và không đạt được
                        if target_gs_count > 1 and nearby_count < target_gs_count:
                            pass  # Không log warning nữa để giảm noise
                    elif nearby_count < 1:
                        # Nếu vẫn không có GS, đặt terminal ngay tại một GS
                        base_station = random.choice(suitable_stations)
                        lat = base_station['position']['latitude'] + random.uniform(-0.1, 0.1)
                        lon = base_station['position']['longitude'] + random.uniform(-0.1, 0.1)
                        nearby_gs = find_nearby_ground_stations(lat, lon, max_range_km=100)
                        nearby_count = len(nearby_gs)
                
                # Log kết quả (chỉ log success, không log warnings nữa)
                if nearby_count >= 1:
                    status = "✅" if nearby_count >= target_gs_count else "✓"
                    print(f"{status} Terminal {terminal_index} at ({lat:.2f}, {lon:.2f}) has {nearby_count} GS in range")
                else:
                    print(f"❌ ERROR: Terminal {terminal_index} at ({lat:.2f}, {lon:.2f}) has {nearby_count} GS in range - THIS SHOULD NOT HAPPEN!")
            
            # Altitude based on terminal type
            if terminal_type == 'AIRCRAFT':
                altitude = random.uniform(8000, 13000)  # Commercial flight altitude
            elif terminal_type == 'MARITIME':
                altitude = random.uniform(0, 50)  # Sea level
            elif terminal_type == 'VEHICLE':
                altitude = random.uniform(0, 2000)  # Ground to mountains
            else:
                altitude = random.uniform(0, 500)  # Fixed/Mobile
            
            # Decide if this terminal should be connected
            should_connect = connected_count < max_connected and available_nodes and random.random() < 0.6
            status = 'connected' if should_connect else 'idle'
            connected_node = None
            connection_metrics = None
            
            if should_connect and available_nodes:
                # Select node based on terminal type preference
                # LEO for latency-sensitive, GEO for high bandwidth, MEO for balance
                node_preferences = {
                    'AIRCRAFT': ['LEO_SATELLITE', 'MEO_SATELLITE'],
                    'MOBILE': ['LEO_SATELLITE', 'MEO_SATELLITE'],
                    'FIXED': ['GEO_SATELLITE', 'MEO_SATELLITE', 'LEO_SATELLITE'],
                    'VEHICLE': ['LEO_SATELLITE', 'MEO_SATELLITE'],
                    'MARITIME': ['GEO_SATELLITE', 'LEO_SATELLITE']
                }
                
                preferred_types = node_preferences.get(terminal_type, ['LEO_SATELLITE'])
                
                # Try to find a preferred node type
                suitable_nodes = [n for n in available_nodes if n['nodeType'] in preferred_types]
                if not suitable_nodes:
                    suitable_nodes = available_nodes
                
                node = random.choice(suitable_nodes)
                connected_node = node['nodeId']
                
                # Calculate realistic connection metrics based on node type
                node_type = node['nodeType']
                
                if node_type == 'LEO_SATELLITE':
                    base_latency = random.uniform(20, 40)
                    bandwidth = random.uniform(50, 150)
                    loss_rate = random.uniform(0.001, 0.01)
                    signal = random.uniform(-75, -55)
                elif node_type == 'MEO_SATELLITE':
                    base_latency = random.uniform(70, 120)
                    bandwidth = random.uniform(30, 100)
                    loss_rate = random.uniform(0.001, 0.008)
                    signal = random.uniform(-80, -60)
                else:  # GEO
                    base_latency = random.uniform(240, 280)  # ~250ms one-way
                    bandwidth = random.uniform(20, 80)
                    loss_rate = random.uniform(0.0001, 0.005)
                    signal = random.uniform(-85, -70)
                
                connection_metrics = {
                    'latencyMs': round(base_latency, 2),
                    'bandwidthMbps': round(bandwidth, 2),
                    'packetLossRate': round(loss_rate, 4),
                    'signalStrength': round(signal, 1),
                    'snrDb': round(random.uniform(5, 20), 1),
                    'jitterMs': round(random.uniform(2, 15), 2)
                }
                connected_count += 1
            
            # Select service type based on terminal type
            possible_services = service_distributions.get(terminal_type, ['TEXT_MESSAGE'])
            service_type = random.choice(possible_services)
            
            terminal = {
                'id': generate_terminal_id(terminal_index),
                'terminalId': generate_terminal_id(terminal_index),
                'terminalName': f'{terminal_type} Terminal {terminal_index + 1}',
                'terminalType': terminal_type,
                'position': {
                    'latitude': round(lat, 4),
                    'longitude': round(lon, 4),
                    'altitude': round(altitude, 2)
                },
                'status': status,
                'connectedNodeId': connected_node,
                'qosRequirements': generate_realistic_qos(service_type),
                'metadata': {
                    'description': f'{terminal_type} terminal in {region["name"]}',
                    'region': region['name'],
                    'serviceType': service_type,
                    'mobility': terminal_type in ['MOBILE', 'VEHICLE', 'AIRCRAFT', 'MARITIME']
                },
                'lastUpdated': datetime.now().isoformat()
            }
            
            # Add connection metrics if connected
            if connection_metrics:
                terminal['connectionMetrics'] = connection_metrics
            
            terminals.append(terminal)
            terminal_index += 1
    
    # Insert all terminals
    if terminals:
        result = terminals_collection.insert_many(terminals)
        print(f"\n✅ Created {len(result.inserted_ids)} terminals:")
        print(f"   - Connected: {connected_count}")
        print(f"   - Idle: {len(result.inserted_ids) - connected_count}")
        
        # Print distribution by type
        type_counts = {}
        for t in terminals:
            type_counts[t['terminalType']] = type_counts.get(t['terminalType'], 0) + 1
        print(f"   - Distribution: {type_counts}")
        
        return [str(id) for id in result.inserted_ids]
    return []

def create_indexes(db):
    """Create indexes for better query performance"""
    try:
        # Nodes indexes
        nodes_collection = db['nodes']
        nodes_collection.create_index('nodeId', unique=True)
        nodes_collection.create_index('nodeType')
        nodes_collection.create_index('isOperational')
        nodes_collection.create_index('lastUpdated')
        nodes_collection.create_index([('position.latitude', 1), ('position.longitude', 1)])
        nodes_collection.create_index('healthy')
        
        # Terminals indexes
        terminals_collection = db['terminals']
        terminals_collection.create_index('terminalId', unique=True)
        terminals_collection.create_index('terminalType')
        terminals_collection.create_index('status')
        terminals_collection.create_index('connectedNodeId')
        terminals_collection.create_index('lastUpdated')
        terminals_collection.create_index([('position.latitude', 1), ('position.longitude', 1)])
        terminals_collection.create_index('qosRequirements.serviceType')
        
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
    
    # Node statistics
    total_nodes = nodes_collection.count_documents({})
    operational_nodes = nodes_collection.count_documents({'isOperational': True})
    
    print(f"\n📡 NODES: {total_nodes} total ({operational_nodes} operational)")
    for node_type in ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE', 'GROUND_STATION']:
        count = nodes_collection.count_documents({'nodeType': node_type})
        if count > 0:
            print(f"   - {node_type}: {count}")
    
    # Terminal statistics
    total_terminals = terminals_collection.count_documents({})
    connected_terminals = terminals_collection.count_documents({'status': 'connected'})
    
    print(f"\n📱 TERMINALS: {total_terminals} total ({connected_terminals} connected)")
    for term_type in ['MOBILE', 'FIXED', 'VEHICLE', 'AIRCRAFT', 'MARITIME']:
        count = terminals_collection.count_documents({'terminalType': term_type})
        if count > 0:
            print(f"   - {term_type}: {count}")
    
    # Service type distribution
    print(f"\n🔧 SERVICE TYPES:")
    for service in ['VIDEO_STREAM', 'AUDIO_CALL', 'FILE_TRANSFER', 'IMAGE_TRANSFER', 'TEXT_MESSAGE']:
        count = terminals_collection.count_documents({'qosRequirements.serviceType': service})
        if count > 0:
            print(f"   - {service}: {count}")
    
    print("\n" + "="*60)

def main():
    """Main function to seed database"""
    print("="*60)
    print("🌱 SEEDING SAGIN DATABASE WITH REALISTIC DATA")
    print("="*60)
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        # Test connection
        client.server_info()
        print("✅ Connected to MongoDB successfully\n")
        
        # Create indexes first
        create_indexes(db)
        
        # Create sample nodes
        print("\n📡 Creating sample nodes (satellites & ground stations)...")
        print("-" * 60)
        node_ids = create_sample_nodes(db)
        
        # Create sample terminals
        print("\n📱 Creating sample terminals...")
        print("-" * 60)
        terminal_ids = create_sample_terminals(db, count=30)
        
        # Print summary
        print_summary_statistics(db)
        
        print("\n✅ Database seeding completed successfully!")
        print(f"   MongoDB URI: {MONGODB_URI}")
        print(f"   Database: {DB_NAME}")
        
        client.close()
        
    except Exception as e:
        print(f"\n❌ Error seeding database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()