#!/usr/bin/env python3
"""
Test script để verify MongoDB Change Stream hoạt động
Kiểm tra xem Python write operations có trigger change events không
"""

import sys
import os
from datetime import datetime

# Add rl-router to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/rl-router'))

from python.utils.db_connector import MongoConnector
from model.Packet import Packet
from model.TwoPacket import TwoPacket

def test_insert_two_packet():
    """Test insert một TwoPacket vào MongoDB"""
    print("=" * 60)
    print("🧪 TEST: Insert TwoPacket vào MongoDB")
    print("=" * 60)

    # Kết nối MongoDB
    db = MongoConnector()
    collection = db.db["two_packets"]

    # Tạo test packet
    test_packet = Packet(
        packet_id="TEST_PKT_001",
        source_user_id="USER_HANOI",
        destination_user_id="USER_BANGKOK",
        size_bytes=1500,
        timestamp=datetime.now().isoformat(),
        use_rl=False,
        dropped=False
    )

    # Tạo TwoPacket
    pair_id = f"{test_packet.source_user_id}_{test_packet.destination_user_id}"
    two_packet = TwoPacket(pairId=pair_id)
    two_packet.dijkstraPacket = test_packet

    # Insert vào MongoDB
    two_packet_dict = two_packet.to_dict()

    print(f"📝 Inserting TwoPacket:")
    print(f"   - PairId: {pair_id}")
    print(f"   - Dijkstra Packet: {test_packet.packet_id}")
    print(f"   - Database: network")
    print(f"   - Collection: two_packets")

    # Insert
    result = collection.insert_one(two_packet_dict)

    print(f"✅ Insert successful! ID: {result.inserted_id}")
    print(f"\n📊 Document inserted:")
    print(f"   {two_packet_dict}")

    # Verify
    found = collection.find_one({"pairId": pair_id})
    print(f"\n🔍 Verification - Document exists: {found is not None}")

    return pair_id

def test_replace_two_packet(pair_id):
    """Test replace (update) TwoPacket"""
    print("\n" + "=" * 60)
    print("🧪 TEST: Replace (Update) TwoPacket")
    print("=" * 60)

    db = MongoConnector()
    collection = db.db["two_packets"]

    # Lấy document hiện tại
    existing = collection.find_one({"pairId": pair_id})
    if not existing:
        print("❌ Document not found!")
        return

    # Tạo RL packet
    rl_packet = Packet(
        packet_id="TEST_PKT_002_RL",
        source_user_id="USER_HANOI",
        destination_user_id="USER_BANGKOK",
        size_bytes=1500,
        timestamp=datetime.now().isoformat(),
        use_rl=True,
        dropped=False
    )

    # Update TwoPacket
    two_packet = TwoPacket.from_dict(existing)
    two_packet.rlPacket = rl_packet

    print(f"📝 Replacing TwoPacket:")
    print(f"   - PairId: {pair_id}")
    print(f"   - Adding RL Packet: {rl_packet.packet_id}")

    # Replace
    two_packet_dict = two_packet.to_dict()
    result = collection.replace_one(
        {"pairId": pair_id},
        two_packet_dict
    )

    print(f"✅ Replace successful!")
    print(f"   - Matched: {result.matched_count}")
    print(f"   - Modified: {result.modified_count}")

    # Verify
    updated = collection.find_one({"pairId": pair_id})
    has_both = (updated.get("dijkstraPacket") is not None and
                updated.get("rlPacket") is not None)
    print(f"\n🔍 Verification - Has both packets: {has_both}")

def test_cleanup(pair_id):
    """Cleanup test data"""
    print("\n" + "=" * 60)
    print("🧹 CLEANUP: Removing test data")
    print("=" * 60)

    db = MongoConnector()
    collection = db.db["two_packets"]

    result = collection.delete_one({"pairId": pair_id})
    print(f"✅ Deleted {result.deleted_count} document(s)")

def main():
    """Main test flow"""
    print("\n🚀 Starting MongoDB Change Stream Test")
    print("=" * 60)
    print("⚠️  IMPORTANT: Make sure Java PacketChangeStreamService is running!")
    print("=" * 60)

    try:
        # Test 1: Insert
        pair_id = test_insert_two_packet()

        print("\n⏳ Waiting 5 seconds for Java to receive INSERT event...")
        import time
        time.sleep(5)

        # Test 2: Replace (Update)
        test_replace_two_packet(pair_id)

        print("\n⏳ Waiting 5 seconds for Java to receive REPLACE event...")
        time.sleep(5)

        # Cleanup
        test_cleanup(pair_id)

        print("\n" + "=" * 60)
        print("✅ Test completed!")
        print("=" * 60)
        print("\n📋 Next steps:")
        print("   1. Check Java application logs for:")
        print("      - '🔄 [INSERT] TwoPacket received'")
        print("      - '🔄 [REPLACE] TwoPacket received'")
        print("      - '📤 [SENT] TwoPacket to /topic/packets'")
        print("   2. If no logs appear, check:")
        print("      - Java service is running")
        print("      - MessageListenerContainer started successfully")
        print("      - MongoDB Change Streams are enabled (should be OK on Atlas)")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
