import socket
import json
import sys
import os
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import time
import random
import uuid

# Add project root to path to allow imports from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.Packet import Packet, QoS, AnalysisData, HopRecord, Position, BufferState, RoutingDecisionInfo, RoutingAlgorithm


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o) if not isinstance(o, type) else None
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


def send_packet(packet: Packet, host: str, port: int):
    """
    Serializes and sends a Packet object to the specified host and port.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))

            packet_json = json.dumps(packet, cls=CustomJSONEncoder, indent=4)
            sock.sendall(packet_json.encode('utf-8'))
            print(f"Successfully sent packet {packet.packet_id} to {host}:{port}")

    except ConnectionRefusedError:
        print(f"Connection refused at {host}:{port}. Is the receiver running?")
    except Exception as e:
        print(f"An error occurred while sending packet: {e}")


def create_test_packet(use_rl: bool) -> Packet:
    """
    Creates a sample packet for testing, configured for RL or Dijkstra.
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


if __name__ == '__main__':
    receiver_host = 'localhost'
    receiver_port = 65432

    # --- Test Case 1: Packet using RL-based routing ---
    print("--- Sending packet with RL routing ---")
    rl_packet = create_test_packet(use_rl=True)
    send_packet(rl_packet, receiver_host, receiver_port)

    print("\n" + "="*40 + "\n")
    time.sleep(2) # Wait a moment before sending the next packet

    # --- Test Case 2: Packet using Dijkstra-based routing ---
    print("--- Sending packet with Dijkstra routing ---")
    dijkstra_packet = create_test_packet(use_rl=False)
    send_packet(dijkstra_packet, receiver_host, receiver_port)
