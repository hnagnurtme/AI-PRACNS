"""
Service quản lý BatchPacket và TwoPacket
Xử lý việc tạo, lưu và query batch packets
✅ Tự động lưu packet vào 2 collections khi packet đến (drop/success)
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from model.Packet import Packet
from model.TwoPacket import TwoPacket
from model.BatchPacket import BatchPacket
from python.utils.db_connector import MongoConnector

logger = logging.getLogger(__name__)


class BatchPacketService:
    """
    Service quản lý BatchPacket và TwoPacket
    ✅ Tự động lưu packet vào 2 collections khi packet đến (drop/success)
    """

    def __init__(self, db_connector: MongoConnector):
        self.db = db_connector
        self.two_packet_collection = "two_packets"
        self.batch_packet_collection = "batch_packets"

    def save_packet(self, packet: Packet):
        """
        ✅ Lưu packet vào database (drop hoặc success)
        Tự động tạo/update TwoPacket và append vào BatchPacket
        """
        if not packet:
            logger.warning("[BatchPacketService] Cannot save null packet")
            return

        source_user_id = packet.source_user_id
        destination_user_id = packet.destination_user_id
        pair_id = self._generate_pair_id(source_user_id, destination_user_id)

        logger.info(f"[BatchPacketService] 💾 Saving packet: {packet.packet_id} | "
                   f"Algorithm: {'RL' if packet.use_rl else 'Dijkstra'} | "
                   f"Dropped: {packet.dropped} | "
                   f"PairId: {pair_id}")

        # ✅ 1. Lưu/Update TwoPacket (xóa và ghi đè)
        self._save_two_packet(pair_id, packet)

        # ✅ 2. Append vào BatchPacket
        self._append_to_batch(pair_id, source_user_id, destination_user_id, packet)

    def create_batch(self, source_user_id: str, destination_user_id: str,
                    total_pairs: int = 0) -> Optional[Dict[str, Any]]:
        """
        Tạo và lưu batch mới
        ✅ BatchId = sourceUserId_destinationUserId
        ✅ Nếu trùng ID → xóa document cũ
        """
        batch_id = self._generate_batch_id(source_user_id, destination_user_id)

        # ✅ Kiểm tra batch cũ
        collection = self.db.db[self.batch_packet_collection]
        existing_batch = collection.find_one({"batchId": batch_id})

        if existing_batch:
            logger.info(f"[BatchPacketService] Batch {batch_id} already exists. Deleting old batch...")
            collection.delete_one({"batchId": batch_id})

        # Tạo batch mới
        batch = BatchPacket(
            batch_id=batch_id,
            total_pair_packets=total_pairs,
            packets=[]
        )

        batch_dict = batch.to_dict()
        result = collection.insert_one(batch_dict)

        logger.info(f"[BatchPacketService] ✅ Created batch {batch_id} with {total_pairs} pairs")

        return {**batch_dict, "_id": result.inserted_id}

    def add_two_packet_to_batch(self, batch_id: str, two_packet: TwoPacket):
        """
        Thêm cặp packet vào batch
        ✅ TwoPacket: Xóa và ghi đè (chỉ giữ 1 document mới nhất)
        ✅ BatchPacket: Ghi chèn (append) TwoPacket vào array packets[]
        """
        try:
            # ✅ Lưu TwoPacket vào collection riêng (UPSERT - xóa và ghi đè)
            two_packet_collection = self.db.db[self.two_packet_collection]
            two_packet_dict = two_packet.to_dict()

            # Replace if exists, insert if not
            two_packet_collection.replace_one(
                {"pairId": two_packet.pairId},
                two_packet_dict,
                upsert=True
            )
            logger.debug(f"[BatchPacketService] ✅ Saved/Replaced TwoPacket: {two_packet.pairId}")

            # ✅ Cập nhật batch: GHI CHÈN TwoPacket vào array
            batch_collection = self.db.db[self.batch_packet_collection]
            batch = batch_collection.find_one({"batchId": batch_id})

            if batch:
                # ✅ LUÔN LUÔN thêm vào array (không check trùng)
                batch_collection.update_one(
                    {"batchId": batch_id},
                    {
                        "$push": {"packets": two_packet_dict},
                        "$inc": {"totalPairPackets": 1}
                    }
                )

                updated_batch = batch_collection.find_one({"batchId": batch_id})
                total_packets = len(updated_batch.get("packets", []))

                logger.info(f"[BatchPacketService] ✅ Appended TwoPacket to batch {batch_id} | "
                          f"Total packets: {total_packets}")
            else:
                logger.warning(f"[BatchPacketService] Batch {batch_id} not found")

        except Exception as e:
            logger.error(f"[BatchPacketService] Error adding TwoPacket to batch: {e}", exc_info=True)

    def save_batch(self, batch: BatchPacket):
        """
        Lưu toàn bộ batch (bao gồm tất cả TwoPackets)
        """
        try:
            # Lưu từng TwoPacket vào collection riêng
            two_packet_collection = self.db.db[self.two_packet_collection]
            for two_packet in batch.packets:
                two_packet_dict = two_packet.to_dict()
                two_packet_collection.replace_one(
                    {"pairId": two_packet.pairId},
                    two_packet_dict,
                    upsert=True
                )

            # Lưu batch
            batch_collection = self.db.db[self.batch_packet_collection]
            batch_dict = batch.to_dict()
            batch_collection.replace_one(
                {"batchId": batch.batch_id},
                batch_dict,
                upsert=True
            )

            logger.info(f"[BatchPacketService] ✅ Saved batch {batch.batch_id} with "
                       f"{len(batch.packets)} packets")

        except Exception as e:
            logger.error(f"[BatchPacketService] Error saving batch: {e}", exc_info=True)
            raise RuntimeError("Failed to save batch") from e

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy batch theo ID
        """
        try:
            collection = self.db.db[self.batch_packet_collection]
            batch = collection.find_one({"batchId": batch_id})
            return batch
        except Exception as e:
            logger.error(f"[BatchPacketService] Error getting batch: {e}", exc_info=True)
            return None

    def get_two_packet(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy TwoPacket theo pairId
        """
        try:
            collection = self.db.db[self.two_packet_collection]
            two_packet = collection.find_one({"pairId": pair_id})
            return two_packet
        except Exception as e:
            logger.error(f"[BatchPacketService] Error getting TwoPacket: {e}", exc_info=True)
            return None

    def get_batch_statistics(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thống kê của một batch
        """
        batch = self.get_batch(batch_id)
        if not batch:
            return None

        packets = batch.get("packets", [])
        total_packets = len(packets)

        dijkstra_success = 0
        dijkstra_dropped = 0
        rl_success = 0
        rl_dropped = 0

        total_dijkstra_latency = 0.0
        total_rl_latency = 0.0

        for two_packet in packets:
            dijkstra_packet = two_packet.get("dijkstraPacket", {})
            rl_packet = two_packet.get("rlPacket", {})

            # Dijkstra stats
            if dijkstra_packet:
                if dijkstra_packet.get("dropped", False):
                    dijkstra_dropped += 1
                else:
                    dijkstra_success += 1
                    total_dijkstra_latency += dijkstra_packet.get("totalDelay", 0.0)

            # RL stats
            if rl_packet:
                if rl_packet.get("dropped", False):
                    rl_dropped += 1
                else:
                    rl_success += 1
                    total_rl_latency += rl_packet.get("totalDelay", 0.0)

        avg_dijkstra_latency = (total_dijkstra_latency / dijkstra_success
                               if dijkstra_success > 0 else 0.0)
        avg_rl_latency = (total_rl_latency / rl_success
                         if rl_success > 0 else 0.0)

        return {
            "batchId": batch_id,
            "totalPackets": total_packets,
            "dijkstra": {
                "successful": dijkstra_success,
                "dropped": dijkstra_dropped,
                "successRate": dijkstra_success / (dijkstra_success + dijkstra_dropped)
                              if (dijkstra_success + dijkstra_dropped) > 0 else 0.0,
                "avgLatencyMs": avg_dijkstra_latency
            },
            "rl": {
                "successful": rl_success,
                "dropped": rl_dropped,
                "successRate": rl_success / (rl_success + rl_dropped)
                              if (rl_success + rl_dropped) > 0 else 0.0,
                "avgLatencyMs": avg_rl_latency
            }
        }

    def _generate_batch_id(self, source_user_id: str, destination_user_id: str) -> str:
        """
        Tạo batchId từ sourceUserId và destinationUserId
        ✅ Format: sourceUserId_destinationUserId
        """
        return f"{source_user_id}_{destination_user_id}"

    def _generate_pair_id(self, source_user_id: str, destination_user_id: str) -> str:
        """
        Tạo pairId từ sourceUserId và destinationUserId
        ✅ Format: sourceUserId_destinationUserId
        """
        return f"{source_user_id}_{destination_user_id}"

    def _save_two_packet(self, pair_id: str, packet: Packet):
        """
        ✅ Lưu/Update TwoPacket (xóa và ghi đè)
        """
        try:
            collection = self.db.db[self.two_packet_collection]

            # Tìm TwoPacket hiện tại
            existing = collection.find_one({"pairId": pair_id})

            if existing:
                # Cập nhật TwoPacket hiện có
                two_packet = TwoPacket.from_dict(existing)

                if packet.use_rl:
                    two_packet.rlPacket = packet
                    logger.debug(f"[BatchPacketService] Updated RL packet in TwoPacket: {pair_id}")
                else:
                    two_packet.dijkstraPacket = packet
                    logger.debug(f"[BatchPacketService] Updated Dijkstra packet in TwoPacket: {pair_id}")
            else:
                # Tạo TwoPacket mới
                two_packet = TwoPacket(pairId=pair_id)

                if packet.use_rl:
                    two_packet.rlPacket = packet
                else:
                    two_packet.dijkstraPacket = packet

                logger.debug(f"[BatchPacketService] Created new TwoPacket: {pair_id}")

            # ✅ Lưu/Ghi đè vào collection two_packets
            two_packet_dict = two_packet.to_dict()
            collection.replace_one(
                {"pairId": pair_id},
                two_packet_dict,
                upsert=True
            )

            logger.info(f"[BatchPacketService] ✅ Saved TwoPacket: {pair_id} | "
                       f"Algorithm: {'RL' if packet.use_rl else 'Dijkstra'} | "
                       f"Dropped: {packet.dropped}")

        except Exception as e:
            logger.error(f"[BatchPacketService] Error saving TwoPacket: {e}", exc_info=True)

    def _append_to_batch(self, pair_id: str, source_user_id: str,
                        destination_user_id: str, packet: Packet):
        """
        ✅ Append TwoPacket vào BatchPacket
        """
        try:
            batch_id = self._generate_batch_id(source_user_id, destination_user_id)

            # Tìm hoặc tạo batch
            batch_collection = self.db.db[self.batch_packet_collection]
            existing_batch = batch_collection.find_one({"batchId": batch_id})

            if not existing_batch:
                # Tạo batch mới nếu chưa có
                batch = BatchPacket(
                    batch_id=batch_id,
                    total_pair_packets=0,
                    packets=[]
                )
                batch_dict = batch.to_dict()
                batch_collection.insert_one(batch_dict)
                logger.debug(f"[BatchPacketService] Created new BatchPacket: {batch_id}")

            # Lấy TwoPacket hiện tại
            two_packet_collection = self.db.db[self.two_packet_collection]
            current_two_packet_dict = two_packet_collection.find_one({"pairId": pair_id})

            if current_two_packet_dict:
                # ✅ LUÔN LUÔN append vào array (ghi chèn)
                batch_collection.update_one(
                    {"batchId": batch_id},
                    {
                        "$push": {"packets": current_two_packet_dict},
                        "$set": {"totalPairPackets": len(existing_batch.get("packets", [])) + 1
                                if existing_batch else 1}
                    }
                )

                updated_batch = batch_collection.find_one({"batchId": batch_id})
                total_packets = len(updated_batch.get("packets", []))

                logger.info(f"[BatchPacketService] ✅ Appended to BatchPacket: {batch_id} | "
                          f"Total packets: {total_packets} | "
                          f"Algorithm: {'RL' if packet.use_rl else 'Dijkstra'}")

        except Exception as e:
            logger.error(f"[BatchPacketService] Error appending to batch: {e}", exc_info=True)
