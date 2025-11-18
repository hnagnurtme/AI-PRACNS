"""
Test script for BatchPacketService
Demonstrates automatic packet saving to TwoPacket and BatchPacket collections
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from service.BatchPacketService import BatchPacketService
from python.utils.db_connector import MongoConnector
from model.Packet import Packet, QoS, AnalysisData
import json


def test_batch_packet_service():
    """Test BatchPacketService functionality"""

    print("=" * 80)
    print("🧪 TESTING BATCH PACKET SERVICE")
    print("=" * 80)

    # Initialize service
    db = MongoConnector()
    service = BatchPacketService(db)

    # Create sample QoS and AnalysisData
    sample_qos = QoS(
        service_type="VIDEO_STREAMING",
        default_priority=5,
        max_latency_ms=100.0,
        max_jitter_ms=10.0,
        min_bandwidth_mbps=50.0,
        max_loss_rate=0.01
    )

    sample_analysis = AnalysisData(
        avg_latency=45.2,
        avg_distance_km=2345.6,
        route_success_rate=1.0,
        total_distance_km=7036.8,
        total_latency_ms=135.6
    )

    # Test Case 1: Create and save Dijkstra packet (successful)
    print("\n📦 Test Case 1: Dijkstra Packet (Successful Delivery)")
    print("-" * 80)

    dijkstra_packet = Packet(
        packet_id="test-dijkstra-001",
        source_user_id="user-Singapore",
        destination_user_id="user-Hanoi",
        station_source="GS_SINGAPORE",
        station_dest="GS_HANOI",
        type="DATA",
        time_sent_from_source_ms=1234567890.0,
        payload_data_base64="base64encodeddata",
        payload_size_byte=1024,
        service_qos=sample_qos,
        current_holding_node_id="GS_HANOI",
        next_hop_node_id="",
        priority_level=5,
        max_acceptable_latency_ms=150.0,
        max_acceptable_loss_rate=0.02,
        analysis_data=sample_analysis,
        use_rl=False,
        ttl=50,
        path_history=["GS_SINGAPORE", "LEO-01", "MEO-05", "GS_HANOI"],
        accumulated_delay_ms=135.6,
        dropped=False,
        drop_reason=None
    )

    service.save_packet(dijkstra_packet)
    print("✅ Dijkstra packet saved successfully")

    # Test Case 2: Create and save RL packet (successful)
    print("\n📦 Test Case 2: RL Packet (Successful Delivery)")
    print("-" * 80)

    rl_packet = Packet(
        packet_id="test-rl-001",
        source_user_id="user-Singapore",
        destination_user_id="user-Hanoi",
        station_source="GS_SINGAPORE",
        station_dest="GS_HANOI",
        type="DATA",
        time_sent_from_source_ms=1234567890.0,
        payload_data_base64="base64encodeddata",
        payload_size_byte=1024,
        service_qos=sample_qos,
        current_holding_node_id="GS_HANOI",
        next_hop_node_id="",
        priority_level=5,
        max_acceptable_latency_ms=150.0,
        max_acceptable_loss_rate=0.02,
        analysis_data=AnalysisData(
            avg_latency=38.7,
            avg_distance_km=2234.5,
            route_success_rate=1.0,
            total_distance_km=6703.5,
            total_latency_ms=116.1
        ),
        use_rl=True,
        ttl=50,
        path_history=["GS_SINGAPORE", "LEO-03", "GEO-01", "GS_HANOI"],
        accumulated_delay_ms=116.1,
        dropped=False,
        drop_reason=None
    )

    service.save_packet(rl_packet)
    print("✅ RL packet saved successfully")

    # Test Case 3: Create and save dropped Dijkstra packet
    print("\n📦 Test Case 3: Dijkstra Packet (Dropped)")
    print("-" * 80)

    dropped_dijkstra = Packet(
        packet_id="test-dijkstra-dropped-001",
        source_user_id="user-Tokyo",
        destination_user_id="user-NewYork",
        station_source="GS_TOKYO",
        station_dest="GS_NEWYORK",
        type="DATA",
        time_sent_from_source_ms=1234567890.0,
        payload_data_base64="base64encodeddata",
        payload_size_byte=1024,
        service_qos=sample_qos,
        current_holding_node_id="LEO-15",
        next_hop_node_id="",
        priority_level=5,
        max_acceptable_latency_ms=150.0,
        max_acceptable_loss_rate=0.02,
        analysis_data=sample_analysis,
        use_rl=False,
        ttl=0,
        path_history=["GS_TOKYO", "LEO-10", "LEO-15"],
        accumulated_delay_ms=165.3,
        dropped=True,
        drop_reason="TTL expired"
    )

    service.save_packet(dropped_dijkstra)
    print("✅ Dropped Dijkstra packet saved successfully")

    # Test Case 4: Create and save dropped RL packet
    print("\n📦 Test Case 4: RL Packet (Dropped)")
    print("-" * 80)

    dropped_rl = Packet(
        packet_id="test-rl-dropped-001",
        source_user_id="user-Tokyo",
        destination_user_id="user-NewYork",
        station_source="GS_TOKYO",
        station_dest="GS_NEWYORK",
        type="DATA",
        time_sent_from_source_ms=1234567890.0,
        payload_data_base64="base64encodeddata",
        payload_size_byte=1024,
        service_qos=sample_qos,
        current_holding_node_id="MEO-08",
        next_hop_node_id="",
        priority_level=5,
        max_acceptable_latency_ms=150.0,
        max_acceptable_loss_rate=0.02,
        analysis_data=sample_analysis,
        use_rl=True,
        ttl=0,
        path_history=["GS_TOKYO", "LEO-12", "MEO-08"],
        accumulated_delay_ms=158.7,
        dropped=True,
        drop_reason="Latency violation: 158.70ms > 150.00ms"
    )

    service.save_packet(dropped_rl)
    print("✅ Dropped RL packet saved successfully")

    # Verify TwoPacket collection
    print("\n" + "=" * 80)
    print("🔍 VERIFICATION: TwoPacket Collection")
    print("=" * 80)

    pair_id_1 = "user-Singapore_user-Hanoi"
    two_packet_1 = service.get_two_packet(pair_id_1)

    if two_packet_1:
        print(f"\n✅ TwoPacket found: {pair_id_1}")
        print(f"   📊 Dijkstra Packet: {two_packet_1.get('dijkstraPacket', {}).get('packetId', 'N/A')}")
        print(f"   📊 RL Packet: {two_packet_1.get('rlPacket', {}).get('packetId', 'N/A')}")
    else:
        print(f"❌ TwoPacket not found: {pair_id_1}")

    pair_id_2 = "user-Tokyo_user-NewYork"
    two_packet_2 = service.get_two_packet(pair_id_2)

    if two_packet_2:
        print(f"\n✅ TwoPacket found: {pair_id_2}")
        print(f"   📊 Dijkstra Packet: {two_packet_2.get('dijkstraPacket', {}).get('packetId', 'N/A')}")
        print(f"   📊 RL Packet: {two_packet_2.get('rlPacket', {}).get('packetId', 'N/A')}")
    else:
        print(f"❌ TwoPacket not found: {pair_id_2}")

    # Verify BatchPacket collection
    print("\n" + "=" * 80)
    print("🔍 VERIFICATION: BatchPacket Collection")
    print("=" * 80)

    batch_1 = service.get_batch(pair_id_1)
    if batch_1:
        print(f"\n✅ BatchPacket found: {pair_id_1}")
        print(f"   📦 Total Packets in Batch: {len(batch_1.get('packets', []))}")
        print(f"   📈 Total Pair Packets: {batch_1.get('totalPairPackets', 0)}")
    else:
        print(f"❌ BatchPacket not found: {pair_id_1}")

    batch_2 = service.get_batch(pair_id_2)
    if batch_2:
        print(f"\n✅ BatchPacket found: {pair_id_2}")
        print(f"   📦 Total Packets in Batch: {len(batch_2.get('packets', []))}")
        print(f"   📈 Total Pair Packets: {batch_2.get('totalPairPackets', 0)}")
    else:
        print(f"❌ BatchPacket not found: {pair_id_2}")

    # Get statistics
    print("\n" + "=" * 80)
    print("📊 BATCH STATISTICS")
    print("=" * 80)

    for batch_id in [pair_id_1, pair_id_2]:
        stats = service.get_batch_statistics(batch_id)
        if stats:
            print(f"\n📋 Statistics for {batch_id}:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"\n⚠️ No statistics available for {batch_id}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_batch_packet_service()
