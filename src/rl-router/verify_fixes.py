
import socket
import json
import time
from datetime import datetime

def create_test_packet(source_user, dest_user, source_station, dest_station, use_rl):
    """Creates a test packet dictionary."""
    return {
        "packet_id": f"pkt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)}",
        "source_user_id": source_user,
        "destination_user_id": dest_user,
        "station_source": source_station,
        "station_dest": dest_station,
        "type": "DATA",
        "time_sent_from_source_ms": time.time() * 1000,
        "payload_data_base64": "VERIFY_FIX_TEST",
        "payload_size_byte": 15,
        "service_qos": {
            "service_type": "REALTIME",
            "default_priority": 1,
            "max_latency_ms": 100.0,
            "max_jitter_ms": 10.0,
            "min_bandwidth_mbps": 50.0,
            "max_loss_rate": 0.01
        },
        "current_holding_node_id": source_station,
        "next_hop_node_id": "",
        "priority_level": 1,
        "max_acceptable_latency_ms": 100.0,
        "max_acceptable_loss_rate": 0.05, # Set higher to avoid initial drop
        "analysis_data": {
            "avg_latency": 0.0,
            "avg_distance_km": 0.0,
            "route_success_rate": 0.0,
            "total_distance_km": 0.0,
            "total_latency_ms": 0.0
        },
        "use_rl": use_rl,
        "ttl": 64,
        "acknowledged_packet_id": None,
        "path_history": [source_station],
        "hop_records": [],
        "accumulated_delay_ms": 0.0,
        "dropped": False,
        "drop_reason": None
    }

def send_packet_to_receiver(packet):
    """Sends a single packet to the TCPReceiver."""
    host = 'localhost'
    port = 10004
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print(f"\n🔌 Connected to TCPReceiver at {host}:{port}")
            
            packet_json = json.dumps(packet)
            s.sendall(packet_json.encode('utf-8'))
            
            print(f"📤 Sent packet {packet['packet_id']} (use_rl={packet['use_rl']})")
            
        print("✅ Connection closed.")
    except ConnectionRefusedError:
        print(f"❌ Connection refused. Is the TCPReceiver running at {host}:{port}?")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    """Main function to send test packets."""
    print("🚀 Starting verification script...")

    # Test Case 1: Dijkstra (use_rl=False)
    # This tests the "No path found by Dijkstra" and the "Loss rate violation" fix.
    # Route: GS_HANOI -> GS_LONDON
    print("\n--- Test Case 1: Dijkstra Routing ---")
    dijkstra_packet = create_test_packet(
        source_user="user-tokyo",
        dest_user="user-london",
        source_station="GS_HANOI",
        dest_station="GS_LONDON",
        use_rl=False
    )
    send_packet_to_receiver(dijkstra_packet)
    
    time.sleep(2) # Wait a moment for the receiver to process

    # Test Case 2: RL (use_rl=True)
    # This tests the "No neighbors found in database" fix for the RL agent.
    # Route: GS_NEWYORK -> GS_HOCHIMINH
    print("\n--- Test Case 2: RL Routing ---")
    rl_packet = create_test_packet(
        source_user="user-ny",
        dest_user="user-sydney",
        source_station="GS_NEWYORK",
        dest_station="GS_HOCHIMINH",
        use_rl=True
    )
    send_packet_to_receiver(rl_packet)

    print("\n✅ Verification script finished.")
    print("Please check the output of the TCPReceiver to confirm the fixes.")

if __name__ == "__main__":
    main()
