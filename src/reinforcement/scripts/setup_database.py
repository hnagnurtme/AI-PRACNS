#!/usr/bin/env python
"""
Database setup script - Initializes MongoDB with sample SAGIN network data.

Usage:
    python scripts/setup_database.py --config configs/network_config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
from datetime import datetime, timezone
from data.mongodb.connection import MongoDBManager


def create_sample_network_data():
    """
    Create sample SAGIN network with satellites, UAVs, and ground stations.

    Returns:
        Dictionary with sample network snapshot
    """
    nodes = []

    # Ground Stations
    ground_stations = [
        {'id': 'gs1', 'lat': 37.7749, 'lon': -122.4194, 'alt': 0.050},  # San Francisco
        {'id': 'gs2', 'lat': 40.7128, 'lon': -74.0060, 'alt': 0.010},   # New York
        {'id': 'gs3', 'lat': 51.5074, 'lon': -0.1278, 'alt': 0.015},    # London
        {'id': 'gs4', 'lat': 35.6762, 'lon': 139.6503, 'alt': 0.020},   # Tokyo
        {'id': 'gs5', 'lat': -33.8688, 'lon': 151.2093, 'alt': 0.030},  # Sydney
    ]

    for gs in ground_stations:
        nodes.append({
            'node_id': gs['id'],
            'nodeType': 'GROUND_STATION',
            'position': {
                'lat': gs['lat'],
                'lon': gs['lon'],
                'alt': gs['alt']
            },
            'operational': True,
            'battery': 100.0,
            'congestion': 0.2,
            'weather': 'CLEAR',
            'packet_count': 0,
            'delay': 1.0,
            'link_quality': 1.0,
            'batteryChargePercent': 100.0,
            'currentPacketCount': 0,
            'resourceUtilization': 0.2,
            'packetLossRate': 0.0,
            'nodeProcessingDelayMs': 1.0,
            'isOperational': True,
            'healthy': True,
            'communication': {
                'bandwidthMHz': 1000.0
            },
            'packetBufferCapacity': 1000,
            'neighbors': []
        })

    # LEO Satellites (Low Earth Orbit)
    leo_satellites = [
        {'id': 'leo1', 'lat': 40.0, 'lon': -100.0, 'alt': 550.0},
        {'id': 'leo2', 'lat': 45.0, 'lon': -90.0, 'alt': 550.0},
        {'id': 'leo3', 'lat': 35.0, 'lon': -110.0, 'alt': 550.0},
        {'id': 'leo4', 'lat': 50.0, 'lon': -80.0, 'alt': 550.0},
        {'id': 'leo5', 'lat': 30.0, 'lon': -120.0, 'alt': 550.0},
        {'id': 'leo6', 'lat': 42.0, 'lon': -95.0, 'alt': 550.0},
    ]

    for leo in leo_satellites:
        nodes.append({
            'node_id': leo['id'],
            'nodeType': 'LEO_SATELLITE',
            'position': {
                'lat': leo['lat'],
                'lon': leo['lon'],
                'alt': leo['alt']
            },
            'operational': True,
            'battery': 90.0,
            'congestion': 0.3,
            'weather': 'CLEAR',  # Satellites above weather
            'packet_count': 0,
            'delay': 5.0,
            'link_quality': 0.9,
            'batteryChargePercent': 90.0,
            'currentPacketCount': 0,
            'resourceUtilization': 0.3,
            'packetLossRate': 0.01,
            'nodeProcessingDelayMs': 5.0,
            'isOperational': True,
            'healthy': True,
            'communication': {
                'bandwidthMHz': 500.0
            },
            'packetBufferCapacity': 800,
            'neighbors': []
        })

    # MEO Satellites (Medium Earth Orbit)
    meo_satellites = [
        {'id': 'meo1', 'lat': 40.0, 'lon': -100.0, 'alt': 10000.0},
        {'id': 'meo2', 'lat': 45.0, 'lon': -90.0, 'alt': 10000.0},
        {'id': 'meo3', 'lat': 35.0, 'lon': -110.0, 'alt': 10000.0},
    ]

    for meo in meo_satellites:
        nodes.append({
            'node_id': meo['id'],
            'nodeType': 'MEO_SATELLITE',
            'position': {
                'lat': meo['lat'],
                'lon': meo['lon'],
                'alt': meo['alt']
            },
            'operational': True,
            'battery': 95.0,
            'congestion': 0.25,
            'weather': 'CLEAR',
            'packet_count': 0,
            'delay': 8.0,
            'link_quality': 0.95,
            'batteryChargePercent': 95.0,
            'currentPacketCount': 0,
            'resourceUtilization': 0.25,
            'packetLossRate': 0.005,
            'nodeProcessingDelayMs': 8.0,
            'isOperational': True,
            'healthy': True,
            'communication': {
                'bandwidthMHz': 1000.0
            },
            'packetBufferCapacity': 1200,
            'neighbors': []
        })

    # GEO Satellite (Geostationary)
    nodes.append({
        'node_id': 'geo1',
        'nodeType': 'GEO_SATELLITE',
        'position': {
            'lat': 0.0,
            'lon': -100.0,
            'alt': 35786.0
        },
        'operational': True,
        'battery': 98.0,
        'congestion': 0.2,
        'weather': 'CLEAR',
        'packet_count': 0,
        'delay': 10.0,
        'link_quality': 0.98,
        'batteryChargePercent': 98.0,
        'currentPacketCount': 0,
        'resourceUtilization': 0.2,
        'packetLossRate': 0.002,
        'nodeProcessingDelayMs': 10.0,
        'isOperational': True,
        'healthy': True,
        'communication': {
            'bandwidthMHz': 2000.0
        },
        'packetBufferCapacity': 1500,
        'neighbors': []
    })

    # UAVs (Unmanned Aerial Vehicles)
    uavs = [
        {'id': 'uav1', 'lat': 37.5, 'lon': -122.0, 'alt': 10.0},
        {'id': 'uav2', 'lat': 40.5, 'lon': -74.5, 'alt': 10.0},
        {'id': 'uav3', 'lat': 51.3, 'lon': -0.5, 'alt': 10.0},
        {'id': 'uav4', 'lat': 35.5, 'lon': 139.5, 'alt': 10.0},
    ]

    for uav in uavs:
        nodes.append({
            'node_id': uav['id'],
            'nodeType': 'UAV',
            'position': {
                'lat': uav['lat'],
                'lon': uav['lon'],
                'alt': uav['alt']
            },
            'operational': True,
            'battery': 70.0,
            'congestion': 0.4,
            'weather': 'CLEAR',
            'packet_count': 0,
            'delay': 2.0,
            'link_quality': 0.85,
            'batteryChargePercent': 70.0,
            'currentPacketCount': 0,
            'resourceUtilization': 0.4,
            'packetLossRate': 0.02,
            'nodeProcessingDelayMs': 2.0,
            'isOperational': True,
            'healthy': True,
            'communication': {
                'bandwidthMHz': 200.0
            },
            'packetBufferCapacity': 500,
            'neighbors': []
        })

    # Compute neighbors based on communication ranges
    import numpy as np

    def calculate_distance(pos1, pos2):
        """Calculate 3D distance between two positions."""
        lat1, lon1, alt1 = pos1['lat'], pos1['lon'], pos1['alt']
        lat2, lon2, alt2 = pos2['lat'], pos2['lon'], pos2['alt']

        # Haversine for surface distance
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        earth_radius = 6371.0  # km
        surface_distance = earth_radius * c

        altitude_diff = abs(alt2 - alt1)
        return np.sqrt(surface_distance**2 + altitude_diff**2)

    def get_comm_range(node_type):
        """Get communication range by node type (in km)."""
        ranges = {
            'GEO_SATELLITE': 40000.0,
            'MEO_SATELLITE': 10000.0,
            'LEO_SATELLITE': 3000.0,
            'UAV': 100.0,
            'GROUND_STATION': 5000.0,
        }
        return ranges.get(node_type, 1000.0)

    # Compute neighbors for each node
    for node in nodes:
        node_pos = node['position']
        node_type = node['nodeType']
        comm_range = get_comm_range(node_type)

        neighbors = []
        for other_node in nodes:
            if node['node_id'] == other_node['node_id']:
                continue

            other_pos = other_node['position']
            distance = calculate_distance(node_pos, other_pos)

            if distance <= comm_range:
                neighbors.append(other_node['node_id'])

        node['neighbors'] = neighbors

    # Create network snapshot
    snapshot = {
        'time': 0.0,
        'nodes': {node['node_id']: node for node in nodes},
        'weather_intensity': 0.0,
        'traffic_load': 1.0
    }

    return snapshot


def main():
    parser = argparse.ArgumentParser(description='Setup SAGIN Database with Sample Data')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to the configuration file'
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Drop existing database and recreate'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    db_config = config.get('database', {})
    host = db_config.get('host', 'localhost')
    port = db_config.get('port', 27017)
    db_name = db_config.get('db_name', 'sagin_simulation')
    username = db_config.get('username')
    password = db_config.get('password')
    auth_source = db_config.get('auth_source', 'admin')

    connection_string = f"mongodb://{host}:{port}/"

    print("="*70)
    print("SAGIN DATABASE SETUP")
    print("="*70)
    print(f"Connection: {connection_string}")
    print(f"Database: {db_name}")
    print(f"Reset: {args.reset}")
    print("="*70 + "\n")

    # Connect to MongoDB
    try:
        print("Connecting to MongoDB...")
        db_manager = MongoDBManager(connection_string, db_name, username=username, password=password, auth_source=auth_source)
        print("✓ Connected successfully!")

    except Exception as e:
        print(f"✗ ERROR: Could not connect to MongoDB: {e}")
        print("\nMake sure MongoDB is running:")
        print("  mongod --dbpath ~/data/db")
        return 1

    # Reset database if requested
    if args.reset:
        response = input("\nWARNING: This will delete all existing data! Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

        print("\nDropping existing collections...")
        db_manager.client.drop_database(db_name)
        db_manager = MongoDBManager(connection_string, db_name, username=username, password=password, auth_source=auth_source)
        print("✓ Database reset complete!")

    # Create sample network data
    print("\nGenerating sample network data...")
    snapshot = create_sample_network_data()

    print(f"✓ Created network with {len(snapshot['nodes'])} nodes:")
    node_types = {}
    for node in snapshot['nodes'].values():
        node_type = node['nodeType']
        node_types[node_type] = node_types.get(node_type, 0) + 1

    for node_type, count in sorted(node_types.items()):
        print(f"  - {node_type}: {count}")

    # Save to database
    print("\nSaving network snapshot to database...")
    try:
        db_manager.save_network_snapshot(snapshot)
        print("✓ Network snapshot saved!")

    except Exception as e:
        print(f"✗ ERROR saving data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verify
    print("\nVerifying data...")
    nodes = db_manager.get_all_nodes()
    print(f"✓ Verified: {len(nodes)} nodes loaded from database")

    # Summary
    print("\n" + "="*70)
    print("DATABASE SETUP COMPLETE!")
    print("="*70)
    print(f"Database '{db_name}' is ready for training.")
    print(f"\nYou can now run:")
    print(f"  python main.py --mode train")
    print(f"  python scripts/train.py --config configs/dynamic_config.yaml")
    print("="*70 + "\n")

    db_manager.close_connection()
    return 0


if __name__ == "__main__":
    exit(main())
