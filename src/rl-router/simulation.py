print("Script is running...")
import os
import sys
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from python.utils.db_connector import MongoConnector
from model.Node import Node, Position, Orbit, Velocity, Communication
from model.User import User
from model.Packet import Packet, HopRecord, QoS, AnalysisData
from service.DijkstraService import DijkstraService, geo_to_ecef, distance_3d

class Simulation:
    def __init__(self):
        load_dotenv()
        self.db_connector = MongoConnector()

    def _ensure_users_exist(self):
        """Checks if users exist in the DB and creates them if not."""
        if self.db_connector.get_all_users():
            return

        print("No users found in the database. Creating sample users...")
        sample_users = [
            {"userId": "user-hanoi", "userName": "User Hanoi", "cityName": "Hanoi", "ipAddress": "127.0.0.1", "port": 10001},
            {"userId": "user-tokyo", "userName": "User Tokyo", "cityName": "Tokyo", "ipAddress": "127.0.0.1", "port": 10002},
            {"userId": "user-singapore", "userName": "User Singapore", "cityName": "Singapore", "ipAddress": "127.0.0.1", "port": 10003},
            {"userId": "user-ny", "userName": "User New York", "cityName": "New York", "ipAddress": "127.0.0.1", "port": 10004},
        ]
        self.db_connector.clear_and_insert_users(sample_users)
        print(f"Inserted {len(sample_users)} sample users.")

    def run(self):
        print("Starting simulation...")

        # 0. Ensure users exist in the database
        self._ensure_users_exist()

        # 1. Fetch network nodes from the database
        nodes_data = self.db_connector.get_all_nodes()
        if not nodes_data:
            print("No network nodes found in the database. Aborting.")
            return

        network_nodes = []
        for node_data in nodes_data:
            pos = Position(**node_data['position'])
            orbit = Orbit(**node_data['orbit'])
            vel = Velocity(**node_data['velocity'])
            comm = Communication(**node_data['communication'])
            node = Node(
                nodeId=node_data['nodeId'],
                nodeName=node_data['nodeName'],
                nodeType=node_data['nodeType'],
                position=pos,
                orbit=orbit,
                velocity=vel,
                communication=comm,
                isOperational=node_data.get('isOperational', True),
                batteryChargePercent=node_data.get('batteryChargePercent', 100),
                nodeProcessingDelayMs=node_data.get('nodeProcessingDelayMs', 1.0),
                packetLossRate=node_data.get('packetLossRate', 0.0),
                resourceUtilization=node_data.get('resourceUtilization', 0.1),
                packetBufferCapacity=node_data.get('packetBufferCapacity', 1000),
                currentPacketCount=node_data.get('currentPacketCount', 0),
                weather=node_data.get('weather', "CLEAR"),
                healthy=node_data.get('healthy', True),
                neighbors=node_data.get('neighbors', [])
            )
            network_nodes.append(node)
        print(f"Loaded {len(network_nodes)} network nodes.")

        # 2. Setup source and destination users
        all_users_data = self.db_connector.get_all_users()
        if len(all_users_data) < 2:
            print("Not enough users in the database to run the simulation. Aborting.")
            return
            
        print("Available users:")
        for user_data in all_users_data:
            print(f"  - ID: {user_data['userId']}, Name: {user_data['userName']}, City: {user_data['cityName']}")

        source_user_data = all_users_data[0]
        dest_user_data = all_users_data[1]

        source_user_data_clean = {k: v for k, v in source_user_data.items() if k != '_id'}
        dest_user_data_clean = {k: v for k, v in dest_user_data.items() if k != '_id'}

        source_user = User(**source_user_data_clean)
        dest_user = User(**dest_user_data_clean)

        print(f"Source: {source_user.userName} ({source_user.userId})")
        print(f"Destination: {dest_user.userName} ({dest_user.userId})")

        # 3. Find the closest ground stations
        ground_stations = [node for node in network_nodes if node.nodeType == "GROUND_STATION"]
        if not ground_stations:
            print("No ground stations found in the network. Aborting.")
            return

        closest_source_station = min(
            ground_stations,
            key=lambda station: distance_3d(
                geo_to_ecef({'latitude': source_user.latitude, 'longitude': source_user.longitude, 'altitude': 0}),
                geo_to_ecef({'latitude': station.latitude, 'longitude': station.longitude, 'altitude': station.altitude})
            )
        )
        
        closest_dest_station = min(
            ground_stations,
            key=lambda station: distance_3d(
                geo_to_ecef({'latitude': dest_user.latitude, 'longitude': dest_user.longitude, 'altitude': 0}),
                geo_to_ecef({'latitude': station.latitude, 'longitude': station.longitude, 'altitude': station.altitude})
            )
        )

        print(f"Closest station to source: {closest_source_station.nodeId}")
        print(f"Closest station to destination: {closest_dest_station.nodeId}")

        # 4. Find the path using Dijkstra's algorithm
        dijkstra_service = DijkstraService(self.db_connector)
        full_path = dijkstra_service.find_shortest_path(closest_source_station.nodeId, closest_dest_station.nodeId)
        
        if not full_path:
            print(f"No path found from {closest_source_station.nodeId} to {closest_dest_station.nodeId}. Aborting.")
            return
            
        print(f"Full path: {' -> '.join(full_path)}")

        # 5. Simulate packet traversal and create hop records
        print("Simulating packet traversal...")
        hop_records = []
        total_latency = 0
        nodes_map = {node.nodeId: node for node in network_nodes}

        for i in range(len(full_path) - 1):
            current_node_id = full_path[i]
            next_node_id = full_path[i+1]

            current_node = nodes_map.get(current_node_id)
            next_node = nodes_map.get(next_node_id)

            if not current_node or not next_node:
                print(f"Error: Could not find node data for hop {current_node_id} -> {next_node_id}. Aborting.")
                return

            node1_pos_dict = {'latitude': current_node.position.latitude, 'longitude': current_node.position.longitude, 'altitude': current_node.position.altitude}
            node2_pos_dict = {'latitude': next_node.position.latitude, 'longitude': next_node.position.longitude, 'altitude': next_node.position.altitude}
            
            distance = distance_3d(geo_to_ecef(node1_pos_dict), geo_to_ecef(node2_pos_dict))
            propagation_delay = (distance / 299792) * 1000  # Speed of light in km/s, result in ms
            processing_delay = current_node.nodeProcessingDelayMs
            
            hop_latency = propagation_delay + processing_delay
            total_latency += hop_latency

            hop_record = HopRecord(
                from_node_id=current_node_id,
                to_node_id=next_node_id,
                latency_ms=hop_latency,
                timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
                distance_km=distance
            )
            hop_records.append(hop_record)
            print(f"  Hop: {current_node_id} -> {next_node_id}, Latency: {hop_latency:.4f} ms")

        print(f"Simulation complete. Total latency: {total_latency:.4f} ms")

        # 6. Create the Packet object with the simulation results
        # Mock data for fields not calculated in this simplified simulation
        mock_qos = QoS(
            service_type="standard",
            default_priority=3,
            max_latency_ms=500.0,
            max_jitter_ms=50.0,
            min_bandwidth_mbps=10.0,
            max_loss_rate=0.01
        )
        mock_analysis = AnalysisData(
            avg_latency=total_latency / len(hop_records) if hop_records else 0,
            avg_distance_km=0, # not calculated
            route_success_rate=1.0,
            total_distance_km=0, # not calculated
            total_latency_ms=total_latency
        )

        packet = Packet(
            packet_id="sim_packet_dijkstra_001",
            source_user_id=source_user.userId,
            destination_user_id=dest_user.userId,
            station_source=closest_source_station.nodeId,
            station_dest=closest_dest_station.nodeId,
            type="DATA",
            time_sent_from_source_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            payload_data_base64="U2FtcGxlIFBheWxvYWQ=", # "Sample Payload"
            payload_size_byte=1024,
            service_qos=mock_qos,
            current_holding_node_id=full_path[-1],
            next_hop_node_id="USER_DEST",
            priority_level=3,
            max_acceptable_latency_ms=500.0,
            max_acceptable_loss_rate=0.01,
            analysis_data=mock_analysis,
            use_rl=False,
            ttl=64,
            hop_records=hop_records,
            path_history=full_path
        )

        # 7. "Deliver" the packet
        print("\n--- Final Packet State ---")
        # The dataclass does not have a to_json method, so we'll just print it.
        print(packet)
        print("--------------------------\n")
        
        # Here you could potentially use TCPSender to send the result,
        # but for now, we just print it.
