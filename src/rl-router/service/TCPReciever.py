import socket
import json
import sys
import os
import time
import random
import math
import torch
import numpy as np
from dataclasses import asdict
from typing import Dict, Any, Optional

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.Packet import Packet, QoS, AnalysisData, HopRecord, Position, BufferState, RoutingDecisionInfo, RoutingAlgorithm
from service.TCPSender import send_packet
from service.DijkstraService import DijkstraService
from python.utils.db_connector import MongoConnector
from python.utils.state_builder import StateBuilder

# Try to import RL components (optional)
try:
    from python.rl_agent.dqn_model import DuelingDQN, INPUT_SIZE
    RL_AVAILABLE = True
except ImportError:
    print("Warning: RL model not available. Will use fallback routing.")
    RL_AVAILABLE = False


OUTPUT_SIZE = 10


def deserialize_packet(data: dict) -> Packet:
    """
    Deserializes a dictionary into a Packet object, handling nested dataclasses.
    """
    # Handle nested dataclasses by manually converting dicts to objects
    if 'service_qos' in data and isinstance(data['service_qos'], dict):
        data['service_qos'] = QoS(**data['service_qos'])

    if 'analysis_data' in data and isinstance(data['analysis_data'], dict):
        data['analysis_data'] = AnalysisData(**data['analysis_data'])

    if 'hop_records' in data and data['hop_records'] is not None:
        hop_records = []
        for hr_data in data['hop_records']:
            if 'from_node_position' in hr_data and hr_data['from_node_position'] is not None:
                hr_data['from_node_position'] = Position(**hr_data['from_node_position'])
            if 'to_node_position' in hr_data and hr_data['to_node_position'] is not None:
                hr_data['to_node_position'] = Position(**hr_data['to_node_position'])
            if 'from_node_buffer_state' in hr_data and hr_data['from_node_buffer_state'] is not None:
                hr_data['from_node_buffer_state'] = BufferState(**hr_data['from_node_buffer_state'])
            if 'routing_decision_info' in hr_data and hr_data['routing_decision_info'] is not None:
                if 'algorithm' in hr_data['routing_decision_info']:
                    hr_data['routing_decision_info']['algorithm'] = RoutingAlgorithm(
                        hr_data['routing_decision_info']['algorithm'])
                hr_data['routing_decision_info'] = RoutingDecisionInfo(**hr_data['routing_decision_info'])
            hop_records.append(HopRecord(**hr_data))
        data['hop_records'] = hop_records

    return Packet(**data)


def calculate_distance_km(pos1: Position, pos2: Position) -> float:
    """
    Calculate 3D Euclidean distance between two positions in km.
    """
    R_EARTH = 6371.0  # km

    lat1_rad = math.radians(pos1.latitude)
    lon1_rad = math.radians(pos1.longitude)
    lat2_rad = math.radians(pos2.latitude)
    lon2_rad = math.radians(pos2.longitude)

    R1 = R_EARTH + pos1.altitude
    R2 = R_EARTH + pos2.altitude

    x1 = R1 * math.cos(lat1_rad) * math.cos(lon1_rad)
    y1 = R1 * math.cos(lat1_rad) * math.sin(lon1_rad)
    z1 = R1 * math.sin(lat1_rad)

    x2 = R2 * math.cos(lat2_rad) * math.cos(lon2_rad)
    y2 = R2 * math.cos(lat2_rad) * math.sin(lon2_rad)
    z2 = R2 * math.sin(lat2_rad)

    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


class TCPReceiver:


    def __init__(self, host: str, port: int, model_path: str = "models/checkpoints/dqn_latest.pth"):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

        # Initialize database connector and services
        self.db = MongoConnector()
        self.dijkstra_service = DijkstraService(self.db)
        self.state_builder = StateBuilder(self.db)

        # Load RL model if available
        self.rl_model = None
        if RL_AVAILABLE:
            try:
                self.rl_model = DuelingDQN(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, weights_only=False)
                    except TypeError:
                        checkpoint = torch.load(model_path)

                    if isinstance(checkpoint, dict) and "model" in checkpoint:
                        self.rl_model.load_state_dict(checkpoint["model"])
                    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        self.rl_model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        self.rl_model.load_state_dict(checkpoint)

                    self.rl_model.eval()
                    print(f"✅ RL Model loaded successfully from {model_path}")
                else:
                    print(f"⚠️ Model file not found at {model_path}. Using fallback routing.")
                    self.rl_model = None
            except Exception as e:
                print(f"⚠️ Error loading RL model: {e}. Using fallback routing.")
                self.rl_model = None

    def listen(self):
        self.sock.listen(5)
        print(f"Listening for incoming connections on {self.host}:{self.port}")
        print(f"RL Model: {'Loaded ✅' if self.rl_model else 'Not Available ⚠️'}")
        while True:
            conn, addr = self.sock.accept()
            print(f"Accepted connection from {addr}")
            try:
                self.handle_client(conn)
            except Exception as e:
                print(f"Error handling client {addr}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()

    def handle_client(self, conn):
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk

        if not data:
            print("No data received.")
            return

        try:
            packet_data = json.loads(data.decode('utf-8'))
            packet = deserialize_packet(packet_data)

            print("\n--- Received Packet ---")
            print(f"Packet ID: {packet.packet_id}")
            print(f"From: {packet.source_user_id} (Station: {packet.station_source})")
            print(f"To: {packet.destination_user_id} (Station: {packet.station_dest})")
            print(f"Current Holder: {packet.current_holding_node_id}")
            print(f"Path History: {packet.path_history}")
            print(f"Using RL: {packet.use_rl}")

            if packet.current_holding_node_id == packet.station_dest:
                self.deliver_to_user(packet)
            else:
                self.process_and_forward_packet(packet)

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from received data: {e}")
        except (TypeError, KeyError) as e:
            print(f"Error deserializing packet: {e}")
            import traceback
            traceback.print_exc()

    def process_and_forward_packet(self, packet: Packet):
        """
        Determines the next hop, updates the packet with hop record from database, and forwards it.
        """
        if packet.ttl <= 0:
            print(f"Packet {packet.packet_id} TTL expired. Dropping packet.")
            packet.dropped = True
            return

        # Determine next hop based on use_rl flag
        if packet.use_rl:
            next_hop_id = self.get_rl_next_hop(packet)
            algo = RoutingAlgorithm.REINFORCEMENT_LEARNING
        else:
            next_hop_id = self.get_dijkstra_next_hop(packet)
            algo = RoutingAlgorithm.DIJKSTRA

        if not next_hop_id:
            print(f"Could not determine next hop for packet {packet.packet_id}. Dropping packet.")
            packet.dropped = True
            return

        print(f"Next hop for packet {packet.packet_id} is {next_hop_id} (using {algo.name})")

        # Fetch COMPLETE node data from database to create hop record
        from_node_id = packet.current_holding_node_id
        from_node_data = self.db.get_node(from_node_id)
        to_node_data = self.db.get_node(next_hop_id)

        if not from_node_data or not to_node_data:
            print(f"Could not fetch node data from database. Dropping packet.")
            packet.dropped = True
            return

        # Create Position objects from database node data
        from_pos_dict = from_node_data.get('position', {})
        to_pos_dict = to_node_data.get('position', {})

        from_node_pos = Position(
            latitude=from_pos_dict.get('latitude', 0.0),
            longitude=from_pos_dict.get('longitude', 0.0),
            altitude=from_pos_dict.get('altitude', 0.0)
        )
        to_node_pos = Position(
            latitude=to_pos_dict.get('latitude', 0.0),
            longitude=to_pos_dict.get('longitude', 0.0),
            altitude=to_pos_dict.get('altitude', 0.0)
        )

        # Calculate distance and latency
        distance = calculate_distance_km(from_node_pos, to_node_pos)

        # Propagation delay (speed of light) + processing delay + random jitter
        propagation_delay = (distance / 299792.458) * 1000  # ms
        processing_delay = from_node_data.get('nodeProcessingDelayMs', 1.0)
        jitter = random.uniform(0.5, 2.0)
        latency = propagation_delay + processing_delay + jitter

        # Create buffer state from DATABASE node data
        from_buffer_state = BufferState(
            queue_size=from_node_data.get('currentPacketCount', 0),
            bandwidth_utilization=from_node_data.get('resourceUtilization', 0.0)
        )

        # Get scenario type from database (if available)
        scenario_type = from_node_data.get('scenarioType', 'NORMAL')

        # Create comprehensive hop record with ALL database fields
        new_hop_record = HopRecord(
            from_node_id=from_node_id,
            to_node_id=next_hop_id,
            latency_ms=latency,
            timestamp_ms=time.time() * 1000,
            distance_km=distance,
            from_node_position=from_node_pos,
            to_node_position=to_node_pos,
            from_node_buffer_state=from_buffer_state,
            routing_decision_info=RoutingDecisionInfo(
                algorithm=algo,
                metric="distance",
                reward=None  # Could be calculated if needed
            ),
            scenario_type=scenario_type,
            node_load_percent=from_node_data.get('resourceUtilization', 0.0) * 100,
            drop_reason_details=None
        )

        # Add hop record to packet
        packet.add_hop_record(new_hop_record)
        packet.accumulated_delay_ms += latency

        # Update packet state
        packet.current_holding_node_id = next_hop_id
        packet.next_hop_node_id = ""
        packet.ttl -= 1

        # Get address from node's communication field in DATABASE
        to_comm = to_node_data.get('communication', {})
        next_hop_address = {
            'host': to_comm.get('ipAddress', 'localhost'),
            'port': to_comm.get('port', 65432)
        }

        print(f"Forwarding packet to {next_hop_id} at {next_hop_address['host']}:{next_hop_address['port']}")
        print(f"  Distance: {distance:.2f} km, Latency: {latency:.2f} ms")
        print(f"  From Buffer: Queue={from_buffer_state.queue_size}, Utilization={from_buffer_state.bandwidth_utilization:.2%}")

        send_packet(packet, next_hop_address['host'], next_hop_address['port'])

    def get_rl_next_hop(self, packet: Packet) -> Optional[str]:
        """
        Gets the next hop using integrated RL model (no external service).
        """
        print("--- Using RL for routing (Integrated DQN) ---")

        if not self.rl_model:
            print("⚠️ RL model not available, using fallback (random neighbor)")
            return self.get_fallback_next_hop(packet)

        try:
            # Prepare packet data for state builder
            packet_state = {
                'currentHoldingNodeId': packet.current_holding_node_id,
                'stationDest': packet.station_dest,
                'path': packet.path_history,
                'ttl': packet.ttl,
                'dropped': packet.dropped,
                'accumulatedDelayMs': packet.accumulated_delay_ms,
                'serviceQoS': {
                    'maxLatencyMs': packet.max_acceptable_latency_ms
                }
            }

            # Get state vector from database
            state_vector = self.state_builder.get_state_vector(packet_state)
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)

            # Get neighbors from database
            current_node = self.db.get_node(packet.current_holding_node_id)
            if not current_node:
                return None

            neighbor_ids = current_node.get('neighbors', [])[:10]  # Max 10 neighbors

            if not neighbor_ids:
                print("No neighbors found in database")
                return None

            # Run RL model inference
            with torch.no_grad():
                q_values_tensor = self.rl_model(state_tensor)
            q_values = q_values_tensor.cpu().numpy().flatten()

            # Mask invalid actions
            valid_q_values = np.full(OUTPUT_SIZE, -np.inf, dtype=np.float32)
            valid_q_values[:len(neighbor_ids)] = q_values[:len(neighbor_ids)]

            # Prevent backtracking
            last_node_id = packet.path_history[-2] if len(packet.path_history) >= 2 else None

            for attempt in range(len(neighbor_ids)):
                action_index = int(np.argmax(valid_q_values))
                next_hop = neighbor_ids[action_index]

                # Check if this hop is valid (not backtracking, not in path)
                if next_hop != last_node_id and next_hop not in packet.path_history:
                    print(f"RL selected: {next_hop} (Q-value: {q_values[action_index]:.4f})")
                    return next_hop

                # Mark this action as invalid and try next best
                valid_q_values[action_index] = -np.inf

            # All neighbors are in path, pick best one anyway
            action_index = int(np.argmax(q_values[:len(neighbor_ids)]))
            next_hop = neighbor_ids[action_index]
            print(f"RL selected (forced): {next_hop} (Q-value: {q_values[action_index]:.4f})")
            return next_hop

        except Exception as e:
            print(f"Error in RL inference: {e}")
            import traceback
            traceback.print_exc()
            return self.get_fallback_next_hop(packet)

    def get_fallback_next_hop(self, packet: Packet) -> Optional[str]:
        """
        Fallback routing: random neighbor from database.
        """
        current_node = self.db.get_node(packet.current_holding_node_id)
        if not current_node:
            return None

        neighbors = current_node.get('neighbors', [])
        if not neighbors:
            return None

        # Try to avoid already visited nodes
        possible_next_hops = [n for n in neighbors if n not in packet.path_history]
        if possible_next_hops:
            return random.choice(possible_next_hops)

        # If all visited, pick any neighbor
        return random.choice(neighbors)

    def get_dijkstra_next_hop(self, packet: Packet) -> Optional[str]:
        """
        Gets the next hop using Dijkstra's algorithm from database.
        """
        print("--- Using Dijkstra for routing ---")
        path = self.dijkstra_service.find_shortest_path(
            packet.current_holding_node_id,
            packet.station_dest
        )

        if path and len(path) > 1:
            print(f"Dijkstra path: {' -> '.join(path)}")
            return path[1]
        else:
            print("No path found by Dijkstra")
            return None

    def deliver_to_user(self, packet: Packet):
        """
        Delivers the packet to the final user and calculates final metrics.
        """
        print("\n--- Packet Reached Destination Station ---")

        # Look up destination user from database
        dest_user = self.db.get_user(packet.destination_user_id)

        if not dest_user:
            print(f"Warning: User {packet.destination_user_id} not found in database")
            # Calculate final metrics anyway
            self.calculate_final_metrics(packet)
            return

        # Send packet to user
        user_address = {
            'host': dest_user.get('ipAddress', 'localhost'),
            'port': dest_user.get('port', 10000)
        }

        print(f"Delivering packet to user {dest_user.get('userName')} at {user_address['host']}:{user_address['port']}")

        # Calculate final metrics before delivery
        self.calculate_final_metrics(packet)

        # Send to user
        try:
            send_packet(packet, user_address['host'], user_address['port'])
            print("✅ Packet delivered successfully to user")
        except Exception as e:
            print(f"❌ Error delivering packet to user: {e}")

    def calculate_final_metrics(self, packet: Packet):
        """
        Calculates final metrics when the packet reaches its destination.
        """
        print("\n" + "="*60)
        print("--- FINAL PACKET ANALYSIS ---")
        print("="*60)

        total_latency = sum(hr.latency_ms for hr in packet.hop_records)
        total_distance = sum(hr.distance_km for hr in packet.hop_records)
        num_hops = len(packet.hop_records)

        avg_latency = total_latency / num_hops if num_hops > 0 else 0
        avg_distance = total_distance / num_hops if num_hops > 0 else 0

        packet.analysis_data.total_latency_ms = total_latency
        packet.analysis_data.total_distance_km = total_distance
        packet.analysis_data.avg_latency = avg_latency
        packet.analysis_data.avg_distance_km = avg_distance
        packet.analysis_data.route_success_rate = 1.0 if not packet.dropped else 0.0

        print(f"Packet ID: {packet.packet_id}")
        print(f"Source: {packet.source_user_id} @ {packet.station_source}")
        print(f"Destination: {packet.destination_user_id} @ {packet.station_dest}")
        print(f"Algorithm: {'RL (DQN)' if packet.use_rl else 'Dijkstra'}")
        print(f"\nMetrics:")
        print(f"  Total Latency: {total_latency:.2f} ms")
        print(f"  Total Distance: {total_distance:.2f} km")
        print(f"  Number of Hops: {num_hops}")
        print(f"  Avg Latency/Hop: {avg_latency:.2f} ms")
        print(f"  Avg Distance/Hop: {avg_distance:.2f} km")
        print(f"  Success Rate: {packet.analysis_data.route_success_rate:.1%}")
        print(f"\nPath Taken:")
        print(f"  {' -> '.join(packet.path_history)}")

        if num_hops > 0:
            print(f"\nHop Details:")
            for i, hop in enumerate(packet.hop_records, 1):
                print(f"  Hop {i}: {hop.from_node_id} -> {hop.to_node_id}")
                print(f"    Distance: {hop.distance_km:.2f} km")
                print(f"    Latency: {hop.latency_ms:.2f} ms")
                if hop.from_node_buffer_state:
                    print(f"    Buffer: Queue={hop.from_node_buffer_state.queue_size}, "
                          f"Util={hop.from_node_buffer_state.bandwidth_utilization:.1%}")

        print("="*60)


if __name__ == '__main__':
    receiver = TCPReceiver('localhost', 65432)
    receiver.listen()
