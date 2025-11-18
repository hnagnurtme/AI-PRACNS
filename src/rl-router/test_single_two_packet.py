#!/usr/bin/env python3
"""
Quick test to send a single TwoPacket with position data
"""

import sys
import os
from datetime import datetime, timezone
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/rl-router'))

from python.utils.db_connector import MongoConnector
from model.Packet import Packet, QoS, HopRecord, BufferState, RoutingDecisionInfo, AnalysisData, RoutingAlgorithm, Position
from model.TwoPacket import TwoPacket

print("=" * 80)
print("🧪 Testing Single TwoPacket with Position Data")
print("=" * 80)

# Sample QoS
qos = QoS(
    service_type="VIDEO_STREAMING",
    default_priority=3,
    max_latency_ms=150.0,
    max_jitter_ms=30.0,
    min_bandwidth_mbps=5.0,
    max_loss_rate=0.01
)

# Sample Analysis Data
analysis = AnalysisData(
    avg_latency=35.0,
    avg_distance_km=1200.0,
    route_success_rate=1.0,
    total_distance_km=3600.0,
    total_latency_ms=105.0
)

# Create hop record with positions
def create_hop(from_id, to_id, from_coords, to_coords):
    return HopRecord(
        from_node_id=from_id,
        to_node_id=to_id,
        latency_ms=random.uniform(20, 40),
        timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        distance_km=random.uniform(800, 1500),
        packet_loss_rate=0.001,
        from_node_position=Position(**from_coords) if from_coords else None,
        to_node_position=Position(**to_coords) if to_coords else None,
        from_node_buffer_state=BufferState(
            queue_size=random.randint(5, 20),
            bandwidth_utilization=random.uniform(0.2, 0.6)
        ),
        routing_decision_info=RoutingDecisionInfo(
            algorithm=RoutingAlgorithm.DIJKSTRA,
            metric="Distance",
            reward=None
        )
    )

# Coordinates
HANOI = {"latitude": 21.0285, "longitude": 105.8542, "altitude": 10000}
SAT1 = {"latitude": 19.5, "longitude": 104.2, "altitude": 850}
SAT2 = {"latitude": 17.8, "longitude": 102.5, "altitude": 920}
BANGKOK = {"latitude": 13.7563, "longitude": 100.5018, "altitude": 10000}

print("\n📦 Creating Dijkstra Packet with positions...")
dijkstra_packet = Packet(
    packet_id="TEST_DIJKSTRA_001",
    source_user_id="USER_HANOI",
    destination_user_id="USER_BANGKOK",
    station_source="STATION_HANOI",
    station_dest="STATION_BANGKOK",
    type="DATA",
    time_sent_from_source_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
    payload_data_base64="SGVsbG8gV29ybGQ=",
    payload_size_byte=1024,
    service_qos=qos,
    current_holding_node_id="STATION_BANGKOK",
    next_hop_node_id="STATION_BANGKOK",
    priority_level=3,
    max_acceptable_latency_ms=150.0,
    max_acceptable_loss_rate=0.01,
    analysis_data=analysis,
    use_rl=False,
    ttl=64,
    path_history=["STATION_HANOI", "SAT_LEO_001", "SAT_LEO_002", "STATION_BANGKOK"],
    hop_records=[
        create_hop("STATION_HANOI", "SAT_LEO_001", HANOI, SAT1),
        create_hop("SAT_LEO_001", "SAT_LEO_002", SAT1, SAT2),
        create_hop("SAT_LEO_002", "STATION_BANGKOK", SAT2, BANGKOK),
    ],
    accumulated_delay_ms=105.0,
    dropped=False,
    drop_reason=None
)

print("✅ Dijkstra packet created")
print(f"   Hop records: {len(dijkstra_packet.hop_records)}")
print(f"   First hop from position: {dijkstra_packet.hop_records[0].from_node_position}")

print("\n📦 Creating RL Packet with positions...")
rl_packet = Packet(
    packet_id="TEST_RL_001",
    source_user_id="USER_HANOI",
    destination_user_id="USER_BANGKOK",
    station_source="STATION_HANOI",
    station_dest="STATION_BANGKOK",
    type="DATA",
    time_sent_from_source_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
    payload_data_base64="SGVsbG8gV29ybGQ=",
    payload_size_byte=1024,
    service_qos=qos,
    current_holding_node_id="STATION_BANGKOK",
    next_hop_node_id="STATION_BANGKOK",
    priority_level=3,
    max_acceptable_latency_ms=150.0,
    max_acceptable_loss_rate=0.01,
    analysis_data=AnalysisData(
        avg_latency=28.0,
        avg_distance_km=1100.0,
        route_success_rate=1.0,
        total_distance_km=3300.0,
        total_latency_ms=84.0
    ),
    use_rl=True,
    ttl=64,
    path_history=["STATION_HANOI", "SAT_LEO_003", "STATION_BANGKOK"],
    hop_records=[
        create_hop("STATION_HANOI", "SAT_LEO_003", HANOI, {"latitude": 17.4, "longitude": 103.0, "altitude": 900}),
        create_hop("SAT_LEO_003", "STATION_BANGKOK", {"latitude": 17.4, "longitude": 103.0, "altitude": 900}, BANGKOK),
    ],
    accumulated_delay_ms=84.0,
    dropped=False,
    drop_reason=None
)

print("✅ RL packet created")
print(f"   Hop records: {len(rl_packet.hop_records)}")
print(f"   First hop from position: {rl_packet.hop_records[0].from_node_position}")

print("\n📊 Creating TwoPacket...")
two_packet = TwoPacket(
    pairId="USER_HANOI_USER_BANGKOK",
    dijkstraPacket=dijkstra_packet,
    rlPacket=rl_packet
)

print("\n💾 Saving to MongoDB...")
db = MongoConnector()
collection = db.db["two_packets"]

# Delete existing
collection.delete_many({"pairId": "USER_HANOI_USER_BANGKOK"})
print("🗑️ Deleted existing TwoPacket")

# Insert new
two_packet_dict = two_packet.to_dict()
result = collection.insert_one(two_packet_dict)
print(f"✅ Inserted TwoPacket: {result.inserted_id}")

# Verify position data
print("\n🔍 Verifying position data in MongoDB...")
saved = collection.find_one({"pairId": "USER_HANOI_USER_BANGKOK"})
if saved:
    dijkstra_hops = saved.get("dijkstraPacket", {}).get("hopRecords", [])
    if dijkstra_hops:
        first_hop = dijkstra_hops[0]
        print(f"✅ First hop fromNodePosition: {first_hop.get('fromNodePosition')}")
        print(f"✅ First hop toNodePosition: {first_hop.get('toNodePosition')}")

        if first_hop.get('fromNodePosition') and first_hop.get('toNodePosition'):
            print("\n🎉 SUCCESS! Position data is saved correctly!")
        else:
            print("\n❌ ERROR: Position data is missing!")
    else:
        print("❌ No hop records found!")
else:
    print("❌ TwoPacket not found in database!")

print("\n" + "=" * 80)
print("✅ Test Complete!")
print("=" * 80)
print("\n💡 Next steps:")
print("1. Backend will detect this change via Change Stream")
print("2. After 3 seconds, it will send to /topic/packets")
print("3. Frontend Monitor page should receive and display route")
print("4. Check: http://localhost:3000/monitor")
print("=" * 80)
