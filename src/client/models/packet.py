# client_app/models/packet.py
from dataclasses import dataclass, field
from typing import List, Optional
import time

class PacketType:
    DATA = 'DATA'
    ACK = 'ACK'

@dataclass
class Packet:
    # 1. Nhận dạng và Phân loại
    packetId: str
    type: str = PacketType.DATA
    sourceUserId: str = ''
    destinationUserId: str = ''
    
    # 2. Dữ liệu & QoS
    payloadSizeByte: int = 1500
    serviceType: str = 'BASIC_DATA'
    maxAcceptableLatencyMs: float = 1000.0
    
    # 3. Dữ liệu Lý thuyết (Tính tại Client)
    theoryPath: List[str] = field(default_factory=list)
    theoryDelayMs: float = 0.0
    
    # 4. Định tuyến và Theo dõi RL
    TTL: int = 10
    currentHoldingNodeId: str = field(init=False, default='') 
    nextHopNodeId: Optional[str] = None
    
    rlPathHistory: List[str] = field(default_factory=list) 
    accumulatedDelayMs: float = 0.0                       
    rlCostAccumulation: float = 0.0
    priorityLevel: int = 3
    
    # 5. Trạng thái và Thời gian
    timeSentFromSourceMs: int = field(default_factory=lambda: int(time.time() * 1000))
    acknowledgedPacketId: Optional[str] = None
    dropped: bool = False
    dropReason: Optional[str] = None

    def __post_init__(self):
        if self.sourceUserId and not self.currentHoldingNodeId:
            self.currentHoldingNodeId = self.sourceUserId
        if self.sourceUserId and not self.rlPathHistory:
            self.rlPathHistory.append(self.sourceUserId)

    def to_dict(self):
        """Trả về Dictionary cho JSON serialization, loại bỏ None."""
        return {k: v for k, v in self.__dict__.items() if v is not None}