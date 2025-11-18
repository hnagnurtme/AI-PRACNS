#!/usr/bin/env python3
"""
Beautiful Sample Data Test for BatchPacketService
Tạo dữ liệu mẫu đẹp với nhiều packets để test visualization
"""

import sys
import os
from datetime import datetime, timezone
import time
import random

# Add rl-router to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/rl-router'))

from python.utils.db_connector import MongoConnector
from model.Packet import Packet, QoS, HopRecord, BufferState, RoutingDecisionInfo, AnalysisData, RoutingAlgorithm
from service.BatchPacketService import BatchPacketService


# ============================================================================
# BEAUTIFUL SAMPLE DATA CONFIGURATION
# ============================================================================

# Asian Cities Network
CITIES = {
    "HANOI": {
        "userId": "USER_HANOI",
        "station": "STATION_HANOI",
        "coords": {"lat": 21.0285, "lon": 105.8542, "alt": 10000}
    },
    "BANGKOK": {
        "userId": "USER_BANGKOK",
        "station": "STATION_BANGKOK",
        "coords": {"lat": 13.7563, "lon": 100.5018, "alt": 10000}
    },
    "SINGAPORE": {
        "userId": "USER_SINGAPORE",
        "station": "STATION_SINGAPORE",
        "coords": {"lat": 1.3521, "lon": 103.8198, "alt": 10000}
    },
    "TOKYO": {
        "userId": "USER_TOKYO",
        "station": "STATION_TOKYO",
        "coords": {"lat": 35.6762, "lon": 139.6503, "alt": 10000}
    },
    "SEOUL": {
        "userId": "USER_SEOUL",
        "station": "STATION_SEOUL",
        "coords": {"lat": 37.5665, "lon": 126.9780, "alt": 10000}
    }
}

# Service Types with realistic QoS
SERVICES = {
    "VIDEO_STREAMING": QoS(
        service_type="VIDEO_STREAMING",
        default_priority=3,
        max_latency_ms=150.0,
        max_jitter_ms=30.0,
        min_bandwidth_mbps=5.0,
        max_loss_rate=0.01
    ),
    "AUDIO_CALL": QoS(
        service_type="AUDIO_CALL",
        default_priority=5,
        max_latency_ms=100.0,
        max_jitter_ms=20.0,
        min_bandwidth_mbps=1.0,
        max_loss_rate=0.005
    ),
    "IMAGE_TRANSFER": QoS(
        service_type="IMAGE_TRANSFER",
        default_priority=2,
        max_latency_ms=200.0,
        max_jitter_ms=50.0,
        min_bandwidth_mbps=2.0,
        max_loss_rate=0.02
    ),
    "TEXT_MESSAGE": QoS(
        service_type="TEXT_MESSAGE",
        default_priority=1,
        max_latency_ms=500.0,
        max_jitter_ms=100.0,
        min_bandwidth_mbps=0.1,
        max_loss_rate=0.05
    )
}

# Network Nodes (Satellites)
SATELLITES = [
    "SAT_LEO_001", "SAT_LEO_002", "SAT_LEO_003", "SAT_LEO_004",
    "SAT_LEO_005", "SAT_LEO_006", "SAT_LEO_007", "SAT_LEO_008"
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_realistic_route(source_station: str, dest_station: str,
                             use_rl: bool = False) -> list:
    """Generate realistic satellite route"""
    # Simulated different routes based on algorithm
    if use_rl:
        # RL might choose more satellites for better load balancing
        num_hops = random.randint(3, 5)
    else:
        # Dijkstra typically chooses shortest path
        num_hops = random.randint(2, 4)

    route = [source_station]
    available_sats = SATELLITES.copy()

    for _ in range(num_hops):
        if available_sats:
            sat = random.choice(available_sats)
            route.append(sat)
            available_sats.remove(sat)

    route.append(dest_station)
    return route


def create_hop_record(from_node: str, to_node: str,
                      timestamp_ms: int, is_congested: bool = False,
                      from_coords: dict = None, to_coords: dict = None) -> HopRecord:
    """Create realistic hop record with position data for visualization"""

    # Realistic latency based on congestion
    base_latency = random.uniform(10, 30)
    if is_congested:
        latency = base_latency * random.uniform(1.5, 3.0)  # Higher latency when congested
        queue_size = random.randint(50, 100)
        bandwidth_util = random.uniform(0.7, 0.95)
        packet_loss_rate = random.uniform(0.01, 0.05)  # Higher loss when congested
    else:
        latency = base_latency
        queue_size = random.randint(0, 30)
        bandwidth_util = random.uniform(0.1, 0.5)
        packet_loss_rate = random.uniform(0.0, 0.01)  # Lower loss normally

    # Create Position objects for route visualization
    from model.Packet import Position
    from_position = None
    to_position = None

    if from_coords:
        from_position = Position(
            latitude=from_coords["lat"],
            longitude=from_coords["lon"],
            altitude=from_coords["alt"]
        )

    if to_coords:
        to_position = Position(
            latitude=to_coords["lat"],
            longitude=to_coords["lon"],
            altitude=to_coords["alt"]
        )

    return HopRecord(
        from_node_id=from_node,
        to_node_id=to_node,
        latency_ms=latency,
        timestamp_ms=timestamp_ms,
        distance_km=random.uniform(500, 2000),
        packet_loss_rate=packet_loss_rate,
        from_node_position=from_position,  # ✅ Added for visualization
        to_node_position=to_position,      # ✅ Added for visualization
        from_node_buffer_state=BufferState(
            queue_size=queue_size,
            bandwidth_utilization=bandwidth_util
        ),
        routing_decision_info=RoutingDecisionInfo(
            algorithm=RoutingAlgorithm.REINFORCEMENT_LEARNING if "RL" in from_node else RoutingAlgorithm.DIJKSTRA,
            metric="Q-Value" if "RL" in from_node else "Distance",
            reward=random.uniform(0.5, 1.0) if "RL" in from_node else None
        )
    )


def create_beautiful_packet(packet_id: str, source_city: str, dest_city: str,
                            service_name: str, use_rl: bool,
                            should_drop: bool = False) -> Packet:
    """Create a beautiful, realistic packet"""

    source = CITIES[source_city]
    dest = CITIES[dest_city]
    service = SERVICES[service_name]

    # Generate route
    route = generate_realistic_route(source["station"], dest["station"], use_rl)

    # Create a map of node coordinates (stations have coords, satellites get random coords)
    node_coords = {}
    node_coords[source["station"]] = source["coords"]
    node_coords[dest["station"]] = dest["coords"]

    # Generate random coordinates for satellite nodes
    for node_id in route:
        if node_id not in node_coords:
            # Generate coordinates between source and destination for satellites
            node_coords[node_id] = {
                "lat": random.uniform(
                    min(source["coords"]["lat"], dest["coords"]["lat"]),
                    max(source["coords"]["lat"], dest["coords"]["lat"])
                ),
                "lon": random.uniform(
                    min(source["coords"]["lon"], dest["coords"]["lon"]),
                    max(source["coords"]["lon"], dest["coords"]["lon"])
                ),
                "alt": random.uniform(500, 1200)  # LEO altitude in km
            }

    # Create hop records
    hop_records = []
    path_history = []
    total_delay = 0.0
    current_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Simulate some congestion in the middle of route
    congestion_points = random.sample(range(1, len(route) - 1),
                                     k=min(2, len(route) - 2)) if len(route) > 2 else []

    for i in range(len(route) - 1):
        is_congested = i in congestion_points
        hop = create_hop_record(
            route[i],
            route[i + 1],
            current_time + int(total_delay),
            is_congested,
            from_coords=node_coords.get(route[i]),     # ✅ Added coordinates for visualization
            to_coords=node_coords.get(route[i + 1])    # ✅ Added coordinates for visualization
        )
        hop_records.append(hop)
        path_history.append(route[i])
        total_delay += hop.latency_ms
        current_time += int(hop.latency_ms)

    path_history.append(route[-1])

    # Decide if packet should be dropped
    dropped = should_drop
    drop_reason = None

    if dropped:
        drop_reason = random.choice([
            "QoS_VIOLATION_LATENCY",
            "BUFFER_OVERFLOW",
            "TTL_EXPIRED",
            "ROUTE_NOT_FOUND"
        ])
    elif total_delay > service.max_latency_ms:
        # Random chance of dropping if QoS violated
        if random.random() < 0.3:
            dropped = True
            drop_reason = "QoS_VIOLATION_LATENCY"

    # Analysis data
    successful_hops = len(hop_records) if not dropped else random.randint(1, len(hop_records) - 1)
    analysis = AnalysisData(
        avg_latency=total_delay / len(hop_records) if hop_records else 0,
        avg_distance_km=sum(h.distance_km for h in hop_records) / len(hop_records) if hop_records else 0,
        route_success_rate=0.0 if dropped else 1.0,
        total_distance_km=sum(h.distance_km for h in hop_records),
        total_latency_ms=total_delay
    )

    return Packet(
        packet_id=packet_id,
        source_user_id=source["userId"],
        destination_user_id=dest["userId"],
        station_source=source["station"],
        station_dest=dest["station"],
        type="DATA",
        time_sent_from_source_ms=current_time,  # ✅ Fixed: Use time_sent_from_source_ms instead of timestamp
        payload_data_base64="SGVsbG8gV29ybGQ=",  # "Hello World"
        payload_size_byte=random.randint(500, 5000),
        service_qos=service,
        current_holding_node_id=route[-1] if not dropped else route[successful_hops],
        next_hop_node_id=route[-1] if not dropped else "DROPPED",
        priority_level=service.default_priority,
        max_acceptable_latency_ms=service.max_latency_ms,
        max_acceptable_loss_rate=service.max_loss_rate,
        analysis_data=analysis,
        use_rl=use_rl,
        ttl=random.randint(50, 100),
        # Optional fields
        path_history=path_history[:successful_hops + 1],
        hop_records=hop_records[:successful_hops],
        accumulated_delay_ms=total_delay,
        dropped=dropped,
        drop_reason=drop_reason
    )


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def test_scenario_1_perfect_comparison():
    """
    Scenario 1: Perfect Comparison
    Cả Dijkstra và RL đều thành công, dễ so sánh
    """
    print("\n" + "=" * 80)
    print("📊 SCENARIO 1: Perfect Comparison - Both Algorithms Succeed")
    print("=" * 80)

    db = MongoConnector()
    service = BatchPacketService(db)

    # Create batch
    pair_id = "HANOI_BANGKOK"
    service.create_batch("USER_HANOI", "USER_BANGKOK", total_pairs=5)

    # Send 5 pairs of packets
    for i in range(5):
        print(f"\n📤 Sending pair {i + 1}/5...")

        # Dijkstra packet
        dijkstra_pkt = create_beautiful_packet(
            f"PKT_DIJKSTRA_{i:03d}",
            "HANOI", "BANGKOK",
            "VIDEO_STREAMING",
            use_rl=False,
            should_drop=False
        )
        service.save_packet(dijkstra_pkt)
        print(f"   ✅ Dijkstra: {dijkstra_pkt.packet_id} | Latency: {dijkstra_pkt.accumulated_delay_ms:.1f}ms")

        time.sleep(0.5)

        # RL packet
        rl_pkt = create_beautiful_packet(
            f"PKT_RL_{i:03d}",
            "HANOI", "BANGKOK",
            "VIDEO_STREAMING",
            use_rl=True,
            should_drop=False
        )
        service.save_packet(rl_pkt)
        print(f"   ✅ RL: {rl_pkt.packet_id} | Latency: {rl_pkt.accumulated_delay_ms:.1f}ms")

        time.sleep(1)

    print("\n✅ Scenario 1 completed!")
    print("💡 Expected: Both algorithms work well, compare metrics")


def test_scenario_2_rl_advantage():
    """
    Scenario 2: RL Advantage
    RL performs better (lower latency, fewer drops)
    """
    print("\n" + "=" * 80)
    print("📊 SCENARIO 2: RL Advantage - RL Outperforms Dijkstra")
    print("=" * 80)

    db = MongoConnector()
    service = BatchPacketService(db)

    service.create_batch("USER_SINGAPORE", "USER_TOKYO", total_pairs=5)

    for i in range(5):
        print(f"\n📤 Sending pair {i + 1}/5...")

        # Dijkstra - higher chance of issues
        dijkstra_pkt = create_beautiful_packet(
            f"PKT_DIJKSTRA_SG_TK_{i:03d}",
            "SINGAPORE", "TOKYO",
            "AUDIO_CALL",
            use_rl=False,
            should_drop=(i % 3 == 0)  # Drop every 3rd packet
        )
        service.save_packet(dijkstra_pkt)
        status = "❌ DROPPED" if dijkstra_pkt.dropped else "✅ SUCCESS"
        print(f"   {status} Dijkstra: {dijkstra_pkt.packet_id} | Latency: {dijkstra_pkt.accumulated_delay_ms:.1f}ms")

        time.sleep(0.5)

        # RL - better performance
        rl_pkt = create_beautiful_packet(
            f"PKT_RL_SG_TK_{i:03d}",
            "SINGAPORE", "TOKYO",
            "AUDIO_CALL",
            use_rl=True,
            should_drop=False  # RL rarely drops
        )
        service.save_packet(rl_pkt)
        print(f"   ✅ SUCCESS RL: {rl_pkt.packet_id} | Latency: {rl_pkt.accumulated_delay_ms:.1f}ms")

        time.sleep(1)

    print("\n✅ Scenario 2 completed!")
    print("💡 Expected: RL shows better success rate and lower latency")


def test_scenario_3_mixed_services():
    """
    Scenario 3: Mixed Services
    Different service types to show QoS handling
    """
    print("\n" + "=" * 80)
    print("📊 SCENARIO 3: Mixed Services - Different QoS Requirements")
    print("=" * 80)

    db = MongoConnector()
    service = BatchPacketService(db)

    service.create_batch("USER_SEOUL", "USER_HANOI", total_pairs=4)

    services_to_test = ["VIDEO_STREAMING", "AUDIO_CALL", "IMAGE_TRANSFER", "TEXT_MESSAGE"]

    for i, service_name in enumerate(services_to_test):
        print(f"\n📤 Sending pair {i + 1}/4 - Service: {service_name}...")

        # Dijkstra
        dijkstra_pkt = create_beautiful_packet(
            f"PKT_DIJKSTRA_{service_name}_{i:03d}",
            "SEOUL", "HANOI",
            service_name,
            use_rl=False,
            should_drop=False
        )
        service.save_packet(dijkstra_pkt)
        print(f"   ✅ Dijkstra: QoS={service_name} | Latency: {dijkstra_pkt.accumulated_delay_ms:.1f}ms")

        time.sleep(0.5)

        # RL
        rl_pkt = create_beautiful_packet(
            f"PKT_RL_{service_name}_{i:03d}",
            "SEOUL", "HANOI",
            service_name,
            use_rl=True,
            should_drop=False
        )
        service.save_packet(rl_pkt)
        print(f"   ✅ RL: QoS={service_name} | Latency: {rl_pkt.accumulated_delay_ms:.1f}ms")

        time.sleep(1)

    print("\n✅ Scenario 3 completed!")
    print("💡 Expected: Different QoS requirements handled appropriately")


def test_scenario_4_high_load():
    """
    Scenario 4: High Load
    Many packets to test congestion visualization
    """
    print("\n" + "=" * 80)
    print("📊 SCENARIO 4: High Load - Stress Test with Many Packets")
    print("=" * 80)

    db = MongoConnector()
    service = BatchPacketService(db)

    service.create_batch("USER_BANGKOK", "USER_SINGAPORE", total_pairs=10)

    for i in range(10):
        print(f"\n📤 Sending pair {i + 1}/10...")

        # Both algorithms under high load
        dijkstra_pkt = create_beautiful_packet(
            f"PKT_DIJKSTRA_HIGHLOAD_{i:03d}",
            "BANGKOK", "SINGAPORE",
            "VIDEO_STREAMING",
            use_rl=False,
            should_drop=(random.random() < 0.2)  # 20% drop rate
        )
        service.save_packet(dijkstra_pkt)

        time.sleep(0.3)

        rl_pkt = create_beautiful_packet(
            f"PKT_RL_HIGHLOAD_{i:03d}",
            "BANGKOK", "SINGAPORE",
            "VIDEO_STREAMING",
            use_rl=True,
            should_drop=(random.random() < 0.1)  # 10% drop rate (better)
        )
        service.save_packet(rl_pkt)

        status_d = "❌" if dijkstra_pkt.dropped else "✅"
        status_r = "❌" if rl_pkt.dropped else "✅"
        print(f"   {status_d} Dijkstra | {status_r} RL")

        time.sleep(0.5)

    print("\n✅ Scenario 4 completed!")
    print("💡 Expected: High congestion, RL handles better")


def show_menu():
    """Show interactive menu"""
    print("\n" + "=" * 80)
    print("🎨 BEAUTIFUL BATCH PACKET SERVICE TESTER")
    print("=" * 80)
    print("\nAvailable Test Scenarios:\n")
    print("1️⃣  Scenario 1: Perfect Comparison (5 pairs, both succeed)")
    print("2️⃣  Scenario 2: RL Advantage (5 pairs, RL outperforms)")
    print("3️⃣  Scenario 3: Mixed Services (4 pairs, different QoS)")
    print("4️⃣  Scenario 4: High Load (10 pairs, stress test)")
    print("5️⃣  Run ALL Scenarios")
    print("0️⃣  Exit")
    print("\n" + "=" * 80)


def main():
    """Main interactive menu"""

    print("\n🚀 Starting Beautiful BatchPacketService Tester...")
    print("💡 This will create beautiful sample data for visualization testing")

    while True:
        show_menu()
        choice = input("\n👉 Select scenario (0-5): ").strip()

        if choice == "0":
            print("\n👋 Goodbye!")
            break
        elif choice == "1":
            test_scenario_1_perfect_comparison()
            input("\n⏸️  Press Enter to continue...")
        elif choice == "2":
            test_scenario_2_rl_advantage()
            input("\n⏸️  Press Enter to continue...")
        elif choice == "3":
            test_scenario_3_mixed_services()
            input("\n⏸️  Press Enter to continue...")
        elif choice == "4":
            test_scenario_4_high_load()
            input("\n⏸️  Press Enter to continue...")
        elif choice == "5":
            print("\n🎯 Running ALL scenarios...")
            test_scenario_1_perfect_comparison()
            time.sleep(2)
            test_scenario_2_rl_advantage()
            time.sleep(2)
            test_scenario_3_mixed_services()
            time.sleep(2)
            test_scenario_4_high_load()
            print("\n🎉 All scenarios completed!")
            input("\n⏸️  Press Enter to continue...")
        else:
            print("❌ Invalid choice. Please select 0-5.")

    print("\n" + "=" * 80)
    print("📊 NEXT STEPS:")
    print("=" * 80)
    print("1. Check Java backend logs for Change Stream events")
    print("2. Open frontend: http://localhost:3000/batch-monitor")
    print("3. Verify visualization shows the test data")
    print("\n💡 TIP: Wait 3-5 seconds after each pair for WebSocket updates")
    print("=" * 80)


if __name__ == "__main__":
    main()
