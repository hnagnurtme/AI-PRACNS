from bson import ObjectId
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from model.TwoPacket import TwoPacket

@dataclass
class BatchPacket:
    """
    Model đại diện cho 1 LÔ gói tin chứa nhiều CẶP (Dijkstra + RL)
    Lưu trong collection: batch_packets
    ✅ BatchId format: sourceUserId_destinationUserId
    ✅ Nếu trùng batchId → xóa document cũ, tạo mới
    """

    # Sử dụng Optional vì MongoDB sẽ tự generate ID mới
    id: Optional[ObjectId] = None

    # ID của batch - Format: "sourceUserId_destinationUserId"
    batch_id: str = None

    # Tổng số cặp packet trong batch
    total_pair_packets: int = 0

    # Danh sách các cặp packets (TwoPacket) - Embedded documents
    packets: List = field(default_factory=list)
    
    # Phương thức để thiết lập ID từ string (nếu cần)
    def set_id(self, id_str: str):
        """Thiết lập ID từ string"""
        if id_str:
            self.id = ObjectId(id_str)
    
    # Phương thức để lấy ID dạng string (tiện cho API responses)
    def get_id_str(self) -> str:
        """Lấy ID dạng string"""
        return str(self.id) if self.id else None
    
    def __post_init__(self):
        """Tự động tính total_pair_packets nếu chưa set"""
        if self.total_pair_packets == 0 and self.packets:
            self.total_pair_packets = len(self.packets)
    
    def to_dict(self):
        """Chuyển object thành dictionary để lưu vào MongoDB"""
        data = {
            "batchId": self.batch_id,
            "totalPairPackets": self.total_pair_packets,
            "packets": [packet.to_dict() if hasattr(packet, 'to_dict') else packet
                       for packet in self.packets] if self.packets else []
        }

        # Chỉ thêm _id nếu có giá trị
        if self.id:
            data["_id"] = self.id

        return data

    @classmethod
    def from_dict(cls, data):
        """Tạo BatchPacket từ dictionary (khi đọc từ MongoDB)"""
        # Lazy import to avoid circular dependency
        from model.TwoPacket import TwoPacket

        return cls(
            id=data.get("_id"),
            batch_id=data.get("batchId"),
            total_pair_packets=data.get("totalPairPackets", 0),
            packets=[TwoPacket.from_dict(packet) if isinstance(packet, dict) else packet
                    for packet in data.get("packets", [])]
        )