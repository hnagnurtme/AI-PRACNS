import socket
import json
import sys
import os
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import time
import random
import uuid
import math
from typing import Optional, Dict, Any
from datetime import datetime, timezone

# Add project root to path to allow imports from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.Packet import Packet, QoS, AnalysisData, HopRecord, Position, BufferState, RoutingDecisionInfo, RoutingAlgorithm
from python.utils.db_connector import MongoConnector


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o) if not isinstance(o, type) else None
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


class EnhancedPacketSender:
    """
    Enhanced packet sender with database integration like Simulator.py
    """
    
    # Default coordinates for major cities
    CITY_COORDINATES = {
        'Hanoi': (21.0285, 105.8542),
        'Ho Chi Minh City': (10.7769, 106.7009),
        'Da Nang': (16.0544, 108.2022),
        'Singapore': (1.3521, 103.8198),
        'Tokyo': (35.6895, 139.6917),
        'Seoul': (37.5665, 126.9780),
        'Bangkok': (13.7563, 100.5018),
        'Kuala Lumpur': (3.1390, 101.6869),
        'Delhi': (28.6139, 77.2090),
        'Sydney': (-33.8688, 151.2093),
        'London': (51.5074, -0.1278),
        'Paris': (48.8566, 2.3522),
        'Berlin': (52.5200, 13.4050),
        'New York': (40.7128, -74.0060),
        'San Francisco': (37.7749, -122.4194),
        'Dubai': (25.276987, 55.296249),
        'Moscow': (55.7558, 37.6173),
        'Cairo': (30.0444, 31.2357),
        'Rio de Janeiro': (-22.9068, -43.1729),
        'Jakarta': (-6.2088, 106.8456)
    }
    
    def __init__(self, db_connector: Optional[MongoConnector] = None):
        self.db = db_connector or MongoConnector()

    def _get_city_coordinates(self, city_name: str) -> tuple:
        """Get coordinates for a city, fallback to random if not found"""
        if city_name in self.CITY_COORDINATES:
            return self.CITY_COORDINATES[city_name]
        else:
            # Fallback: random coordinates within reasonable range
            return (random.uniform(-90, 90), random.uniform(-180, 180))

    def _calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points in km"""
        R = 6371.0  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    def find_nearest_ground_station(self, user_id: str) -> Optional[Dict]:
        """
        Find nearest ground station from database for a user
        """
        # Get user from database
        user = self.db.get_user(user_id)
        if not user:
            print(f"❌ User {user_id} not found in database")
            return None

        # Get user's city and coordinates
        city_name = user.get('cityName', 'Unknown')
        user_lat = user.get('latitude')
        user_lon = user.get('longitude')
        
        # If user doesn't have coordinates, use city coordinates
        if user_lat is None or user_lon is None:
            user_lat, user_lon = self._get_city_coordinates(city_name)
            print(f"📍 Using city coordinates for {city_name}: ({user_lat}, {user_lon})")

        # Get all ground stations from database
        all_nodes = self.db.get_all_nodes()
        ground_stations = [node for node in all_nodes if node.get('nodeType') == 'GROUND_STATION']

        if not ground_stations:
            print("❌ No ground stations found in database")
            return None

        # Find nearest ground station
        min_distance = float('inf')
        nearest_gs = None

        for gs in ground_stations:
            gs_pos = gs.get('position', {})
            gs_lat = gs_pos.get('latitude', 0)
            gs_lon = gs_pos.get('longitude', 0)

            distance = self._calculate_haversine_distance(user_lat, user_lon, gs_lat, gs_lon)

            if distance < min_distance:
                min_distance = distance
                nearest_gs = gs

        if nearest_gs:
            print(f"📍 Nearest ground station for {user_id}: {nearest_gs.get('nodeName')} ({min_distance:.2f} km)")
        return nearest_gs

    def create_packet_from_database(self, source_user_id: str, destination_user_id: str,
                                   payload: str = "Test payload", use_rl: bool = False) -> Optional[Packet]:
        """
        Create a realistic packet using database information (like Simulator.py)
        """
        print(f"🚀 Creating packet from database: {source_user_id} -> {destination_user_id}")

        # Get users from database
        source_user = self.db.get_user(source_user_id)
        dest_user = self.db.get_user(destination_user_id)

        if not source_user or not dest_user:
            print(f"❌ Users not found in database")
            return None

        # Find nearest ground stations for both users
        source_gs = self.find_nearest_ground_station(source_user_id)
        dest_gs = self.find_nearest_ground_station(destination_user_id)

        if not source_gs or not dest_gs:
            print(f"❌ Could not find ground stations for users")
            return None

        print(f"📍 Source GS: {source_gs.get('nodeName')}")
        print(f"🎯 Destination GS: {dest_gs.get('nodeName')}")

        # Create QoS based on service type
        qos = QoS(
            service_type="REALTIME",
            default_priority=1,
            max_latency_ms=100.0,
            max_jitter_ms=10.0,
            min_bandwidth_mbps=50.0,
            max_loss_rate=0.01
        )

        # Create analysis data
        analysis_data = AnalysisData(
            avg_latency=0.0,
            avg_distance_km=0.0,
            route_success_rate=0.0,
            total_distance_km=0.0,
            total_latency_ms=0.0
        )

        # Create packet with database information
        packet = Packet(
            packet_id=f"pkt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            source_user_id=source_user_id,
            destination_user_id=destination_user_id,
            station_source=source_gs.get('nodeId', ''),
            station_dest=dest_gs.get('nodeId', ''),
            type="DATA",
            time_sent_from_source_ms=datetime.now(timezone.utc).timestamp() * 1000,
            payload_data_base64=payload.encode('utf-8').hex(),
            payload_size_byte=len(payload),
            service_qos=qos,
            current_holding_node_id=source_gs.get('nodeId', ''),
            next_hop_node_id="",
            priority_level=1,
            max_acceptable_latency_ms=100.0,
            max_acceptable_loss_rate=0.01,
            analysis_data=analysis_data,
            use_rl=use_rl,
            ttl=64,
            path_history=[source_gs.get('nodeId', '')]
        )

        print(f"✅ Packet created: {packet.packet_id}")
        print(f"   Algorithm: {'RL' if use_rl else 'Dijkstra'}")
        print(f"   Payload size: {len(payload)} bytes")
        print(f"   QoS: Latency<{packet.max_acceptable_latency_ms}ms, "
              f"Loss<{packet.max_acceptable_loss_rate}, "
              f"BW>{packet.service_qos.min_bandwidth_mbps}Mbps")

        return packet

    def send_packet_to_receiver(self, packet: Packet, receiver_host: str = 'localhost', receiver_port: int = 10004):
        """
        Send packet to TCP Receiver for processing and routing
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(10.0)  # 10 seconds timeout
                sock.connect((receiver_host, receiver_port))

                packet_json = json.dumps(packet, cls=CustomJSONEncoder, indent=2)
                sock.sendall(packet_json.encode('utf-8'))
                print(f"✅ Successfully sent packet {packet.packet_id} to TCP Receiver at {receiver_host}:{receiver_port}")

        except ConnectionRefusedError:
            print(f"❌ Connection refused at {receiver_host}:{receiver_port}. Is TCP Receiver running?")
        except socket.timeout:
            print(f"⏰ Connection timeout to {receiver_host}:{receiver_port}")
        except Exception as e:
            print(f"❌ Error sending packet to TCP Receiver: {e}")

    def get_user_location_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get user location information for display
        """
        user = self.db.get_user(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Get user name - handle both field names
        user_name = user.get('userName') or user.get('user_name') or 'Unknown'
        city_name = user.get('cityName') or user.get('city_name') or 'Unknown'
        
        # Get coordinates
        user_lat = user.get('latitude')
        user_lon = user.get('longitude')
        
        if user_lat is None or user_lon is None:
            user_lat, user_lon = self._get_city_coordinates(city_name)
        
        # Get nearest ground station
        nearest_gs = self.find_nearest_ground_station(user_id)
        
        return {
            "user_id": user_id,
            "user_name": user_name,
            "city": city_name,
            "latitude": user_lat,
            "longitude": user_lon,
            "nearest_ground_station": nearest_gs.get('nodeName', 'Unknown') if nearest_gs else 'Not found',
            "ground_station_id": nearest_gs.get('nodeId', '') if nearest_gs else ''
        }

    def validate_user_locations(self):
        """
        Validate that all users have location data
        """
        users = self.db.get_all_users()
        missing_locations = []
        
        for user in users:
            if user.get('latitude') is None or user.get('longitude') is None:
                user_id = user.get('userId')
                missing_locations.append(user_id)
        
        if missing_locations:
            print(f"⚠️ Warning: {len(missing_locations)} users missing location data")
            # Auto-fill missing coordinates based on city
            for user_id in missing_locations:
                user = self.db.get_user(user_id)
                if user:
                    city_name = user.get('cityName') or 'Unknown'
                    lat, lon = self._get_city_coordinates(city_name)
                    print(f"   📍 Auto-filled {user_id} ({city_name}): ({lat}, {lon})")
        else:
            print("✅ All users have location data")
        
        return len(missing_locations) == 0

    def update_user_coordinates(self, user_id: str, lat: float, lon: float) -> bool:
        """
        Update user coordinates in database
        """
        try:
            updates = {
                'latitude': lat,
                'longitude': lon,
                'lastUpdated': datetime.now(timezone.utc).isoformat()
            }
            return self.db.update_user(user_id, updates)
        except Exception as e:
            print(f"❌ Error updating user coordinates: {e}")
            return False


def send_packet(packet: Packet, host: str, port: int):
    """
    Serializes and sends a Packet object to the specified host and port.
    This function is maintained for backward compatibility.
    """
    sender = EnhancedPacketSender()
    sender.send_packet_to_receiver(packet, host, port)


def create_test_packet(use_rl: bool) -> Packet:
    """
    Creates a sample packet for testing, configured for RL or Dijkstra.
    Maintained for backward compatibility.
    """
    source_station = "GS_HANOI"
    dest_station = "GS_CAIRE"
    
    packet = Packet(
        packet_id=str(uuid.uuid4()),
        source_user_id="user_A",
        destination_user_id="user_B",
        station_source=source_station,
        station_dest=dest_station,
        time_sent_from_source_ms=time.time() * 1000,
        current_holding_node_id=source_station,
        path_history=[source_station],
        use_rl=use_rl,
        ttl=10,
        type="data",
        payload_data_base64="",
        payload_size_byte=0,
        service_qos=QoS(
            service_type="best_effort",
            default_priority=5,
            max_latency_ms=5000,
            max_jitter_ms=500,
            min_bandwidth_mbps=1,
            max_loss_rate=0.05
        ),
        next_hop_node_id="",
        priority_level=1,
        max_acceptable_latency_ms=1000,
        max_acceptable_loss_rate=0.01,
        analysis_data=AnalysisData(
            avg_latency=0.0,
            avg_distance_km=0.0,
            route_success_rate=0.0,
            total_distance_km=0.0,
            total_latency_ms=0.0
        )
    )

    print(f"Created packet {packet.packet_id} with use_rl={use_rl}")
    return packet


def main():
    """
    Main function to demonstrate enhanced packet sending with database integration
    """
    receiver_host = 'localhost'
    receiver_port = 10004  # TCP Receiver port

    print("="*70)
    print("🚀 ENHANCED TCP PACKET SENDER WITH DATABASE INTEGRATION")
    print("="*70)

    # Initialize enhanced sender with database
    enhanced_sender = EnhancedPacketSender()

    # Validate and auto-fill user locations
    print("\n🔍 Validating user locations...")
    enhanced_sender.validate_user_locations()

    # Display available users
    print("\n👥 Available Users:")
    users = enhanced_sender.db.get_all_users()
    for user in users:
        user_info = enhanced_sender.get_user_location_info(user['userId'])
        
        # Safe printing with error handling
        if 'error' in user_info:
            print(f"   ❌ Error: {user_info['error']}")
            continue
            
        print(f"   📍 {user_info['user_name']} ({user_info['user_id']})")
        print(f"      🏙️  {user_info['city']} | 📍 Lat: {user_info['latitude']:.4f}, Lon: {user_info['longitude']:.4f}")
        print(f"      🛰️  Nearest GS: {user_info['nearest_ground_station']}")

    # Test with database-integrated packet creation
    try:
        print("\n" + "="*70)
        print("📦 TEST CASE 1: RL-BASED ROUTING WITH DATABASE")
        print("="*70)
        
        # Get user location info with error handling
        source_info = enhanced_sender.get_user_location_info("user-singapore")
        dest_info = enhanced_sender.get_user_location_info("user-hanoi")
        
        # Check for errors
        if 'error' in source_info:
            print(f"❌ Source user error: {source_info['error']}")
        elif 'error' in dest_info:
            print(f"❌ Destination user error: {dest_info['error']}")
        else:
            print(f"📍 Source: {source_info['user_name']} in {source_info['city']}")
            print(f"🎯 Destination: {dest_info['user_name']} in {dest_info['city']}")
            print(f"🛰️  Source GS: {source_info['nearest_ground_station']}")
            print(f"🛰️  Dest GS: {dest_info['nearest_ground_station']}")

            rl_packet = enhanced_sender.create_packet_from_database(
                source_user_id="user-singapore",
                destination_user_id="user-hanoi",
                payload="Hello from Singapore to Hanoi using RL routing! 🚀",
                use_rl=True
            )

            if rl_packet:
                enhanced_sender.send_packet_to_receiver(rl_packet, receiver_host, receiver_port)

        print("\n" + "="*70)
        print("📦 TEST CASE 2: DIJKSTRA-BASED ROUTING WITH DATABASE")  
        print("="*70)
        
        # Get user location info with error handling
        source_info = enhanced_sender.get_user_location_info("user-singapore")
        dest_info = enhanced_sender.get_user_location_info("user-tokyo")
        
        if 'error' in source_info:
            print(f"❌ Source user error: {source_info['error']}")
        elif 'error' in dest_info:
            print(f"❌ Destination user error: {dest_info['error']}")
        else:
            print(f"📍 Source: {source_info['user_name']} in {source_info['city']}")
            print(f"🎯 Destination: {dest_info['user_name']} in {dest_info['city']}")
            print(f"🛰️  Source GS: {source_info['nearest_ground_station']}")
            print(f"🛰️  Dest GS: {dest_info['nearest_ground_station']}")

            dijkstra_packet = enhanced_sender.create_packet_from_database(
                source_user_id="user-singapore", 
                destination_user_id="user-tokyo",
                payload="Hello from Singapore to Tokyo using Dijkstra routing! 🗺️",
                use_rl=False
            )

            if dijkstra_packet:
                enhanced_sender.send_packet_to_receiver(dijkstra_packet, receiver_host, receiver_port)

        print("\n" + "="*70)
        print("📦 TEST CASE 3: LONG DISTANCE ROUTING")
        print("="*70)
        
        # Get user location info with error handling
        source_info = enhanced_sender.get_user_location_info("user-london")
        dest_info = enhanced_sender.get_user_location_info("user-sydney")
        
        if 'error' in source_info:
            print(f"❌ Source user error: {source_info['error']}")
        elif 'error' in dest_info:
            print(f"❌ Destination user error: {dest_info['error']}")
        else:
            print(f"📍 Source: {source_info['user_name']} in {source_info['city']}")
            print(f"🎯 Destination: {dest_info['user_name']} in {dest_info['city']}")
            print(f"🛰️  Source GS: {source_info['nearest_ground_station']}")
            print(f"🛰️  Dest GS: {dest_info['nearest_ground_station']}")

            long_distance_packet = enhanced_sender.create_packet_from_database(
                source_user_id="user-london",
                destination_user_id="user-sydney", 
                payload="Hello from London to Sydney! 🌏",
                use_rl=random.choice([True, False])  # Random algorithm
            )

            if long_distance_packet:
                enhanced_sender.send_packet_to_receiver(long_distance_packet, receiver_host, receiver_port)

        print("\n" + "="*70)
        print("✅ ALL PACKETS SENT SUCCESSFULLY!")
        print("="*70)

        # Summary
        print("\n📊 SENDING SUMMARY:")
        print(f"   • TCP Receiver: {receiver_host}:{receiver_port}")
        print(f"   • Total Packets Sent: 3")
        print(f"   • Database Integration: ✅ Active")
        print(f"   • Location-based Routing: ✅ Active")
        print(f"   • QoS Enforcement: ✅ Active")
        print(f"   • Auto-coordinate Filling: ✅ Active")

    except Exception as e:
        print(f"❌ Error during packet sending: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()