# from datetime import datetime, timezone
# from typing import Dict, Any, Optional
# import json
# from bson import ObjectId

# from model.Packet import Packet


# class TwoPacket:
#     """
#     Lớp đại diện cho cặp packet (Dijkstra và RL)
#     ✅ Format pairId: "sourceUserId_destinationUserId"
#     ✅ Mỗi cặp user CHỈ có 1 document
#     """
    
#     def __init__(self, pairId: str = None, dijkstraPacket: Packet = None, 
#                  rlPacket: Packet = None, id: ObjectId = None):
        
#         self.id = id or ObjectId()
#         self.pairId = pairId
#         self.dijkstraPacket = dijkstraPacket or Packet()
#         self.rlPacket = rlPacket or Packet()
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Chuyển đối tượng TwoPacket thành dictionary (phù hợp với MongoDB)"""
#         result = {
#             "_id": self.id,
#             "pairId": self.pairId,
#             "dijkstraPacket": self.dijkstraPacket.to_dict(),
#             "rlPacket": self.rlPacket.to_dict()
#         }
#         return result
    
#     def to_json_dict(self) -> Dict[str, Any]:
#         """Chuyển đối tượng TwoPacket thành dictionary (phù hợp với JSON serialization)"""
#         result = {
#             "_id": str(self.id),
#             "pairId": self.pairId,
#             "dijkstraPacket": self.dijkstraPacket.to_dict(),
#             "rlPacket": self.rlPacket.to_dict()
#         }
#         return result
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'TwoPacket':
#         """Tạo đối tượng TwoPacket từ dictionary"""
#         # Xử lý _id có thể là string hoặc ObjectId
#         _id = data.get("_id")
#         if _id and isinstance(_id, str):
#             _id = ObjectId(_id)
        
#         # Xử lý các packet
#         dijkstra_data = data.get("dijkstraPacket", {})
#         rl_data = data.get("rlPacket", {})
        
#         return cls(
#             id=_id or ObjectId(),
#             pairId=data.get("pairId"),
#             dijkstraPacket=Packet.from_dict(dijkstra_data),
#             rlPacket=Packet.from_dict(rl_data)
#         )
    
#     def generate_pair_id(self, sourceUserId: str, destinationUserId: str) -> str:
#         """Tạo pairId theo format: sourceUserId_destinationUserId"""
#         self.pairId = f"{sourceUserId}_{destinationUserId}"
#         return self.pairId
    
#     def update_dijkstra_packet(self, sourceUserId: str, destinationUserId: str, 
#                              path: list, totalDelay: float, totalPacketLoss: float,
#                              bandwidth: float):
#         """Cập nhật packet Dijkstra"""
#         self.dijkstraPacket = Packet(
#             packetId=f"dijkstra_{self.pairId}",
#             sourceUserId=sourceUserId,
#             destinationUserId=destinationUserId,
#             path=path,
#             totalDelay=totalDelay,
#             totalPacketLoss=totalPacketLoss,
#             bandwidth=bandwidth,
#             algorithm="DIJKSTRA",
#             status="SENT"
#         )
    
#     def update_rl_packet(self, sourceUserId: str, destinationUserId: str,
#                        path: list, totalDelay: float, totalPacketLoss: float,
#                        bandwidth: float):
#         """Cập nhật packet RL"""
#         self.rlPacket = Packet(
#             packetId=f"rl_{self.pairId}",
#             sourceUserId=sourceUserId,
#             destinationUserId=destinationUserId,
#             path=path,
#             totalDelay=totalDelay,
#             totalPacketLoss=totalPacketLoss,
#             bandwidth=bandwidth,
#             algorithm="RL",
#             status="SENT"
#         )
    
#     def compare_performance(self) -> Dict[str, Any]:
#         """So sánh hiệu năng giữa hai thuật toán"""
#         dijkstra = self.dijkstraPacket
#         rl = self.rlPacket
        
#         comparison = {
#             "pairId": self.pairId,
#             "delay_comparison": {
#                 "dijkstra": dijkstra.totalDelay,
#                 "rl": rl.totalDelay,
#                 "winner": "DIJKSTRA" if dijkstra.totalDelay < rl.totalDelay else "RL" if rl.totalDelay < dijkstra.totalDelay else "TIE"
#             },
#             "packet_loss_comparison": {
#                 "dijkstra": dijkstra.totalPacketLoss,
#                 "rl": rl.totalPacketLoss,
#                 "winner": "DIJKSTRA" if dijkstra.totalPacketLoss < rl.totalPacketLoss else "RL" if rl.totalPacketLoss < dijkstra.totalPacketLoss else "TIE"
#             },
#             "bandwidth_comparison": {
#                 "dijkstra": dijkstra.bandwidth,
#                 "rl": rl.bandwidth,
#                 "winner": "DIJKSTRA" if dijkstra.bandwidth > rl.bandwidth else "RL" if rl.bandwidth > dijkstra.bandwidth else "TIE"
#             },
#             "path_length_comparison": {
#                 "dijkstra": len(dijkstra.path),
#                 "rl": len(rl.path),
#                 "winner": "DIJKSTRA" if len(dijkstra.path) < len(rl.path) else "RL" if len(rl.path) < len(dijkstra.path) else "TIE"
#             }
#         }
        
#         # Tính điểm tổng
#         dijkstra_score = 0
#         rl_score = 0
        
#         if comparison["delay_comparison"]["winner"] == "DIJKSTRA":
#             dijkstra_score += 1
#         elif comparison["delay_comparison"]["winner"] == "RL":
#             rl_score += 1
            
#         if comparison["packet_loss_comparison"]["winner"] == "DIJKSTRA":
#             dijkstra_score += 1
#         elif comparison["packet_loss_comparison"]["winner"] == "RL":
#             rl_score += 1
            
#         if comparison["bandwidth_comparison"]["winner"] == "DIJKSTRA":
#             dijkstra_score += 1
#         elif comparison["bandwidth_comparison"]["winner"] == "RL":
#             rl_score += 1
            
#         if comparison["path_length_comparison"]["winner"] == "DIJKSTRA":
#             dijkstra_score += 1
#         elif comparison["path_length_comparison"]["winner"] == "RL":
#             rl_score += 1
        
#         comparison["overall_winner"] = "DIJKSTRA" if dijkstra_score > rl_score else "RL" if rl_score > dijkstra_score else "TIE"
#         comparison["scores"] = {"dijkstra": dijkstra_score, "rl": rl_score}
        
#         return comparison
    
#     def __str__(self) -> str:
#         return f"TwoPacket(pairId={self.pairId}, dijkstra={self.dijkstraPacket.status}, rl={self.rlPacket.status})"

# class TwoPacketManager:
#     """Quản lý collection TwoPacket"""
    
#     def __init__(self):
#         self.packets = {}  # pairId -> TwoPacket
    
#     def create_two_packet(self, sourceUserId: str, destinationUserId: str) -> TwoPacket:
#         """Tạo mới một TwoPacket cho cặp user"""
#         pairId = f"{sourceUserId}_{destinationUserId}"
        
#         if pairId in self.packets:
#             return self.packets[pairId]
        
#         two_packet = TwoPacket()
#         two_packet.generate_pair_id(sourceUserId, destinationUserId)
#         self.packets[pairId] = two_packet
        
#         return two_packet
    
#     def get_two_packet(self, pairId: str) -> Optional[TwoPacket]:
#         """Lấy TwoPacket theo pairId"""
#         return self.packets.get(pairId)
    
#     def get_by_user_pair(self, sourceUserId: str, destinationUserId: str) -> Optional[TwoPacket]:
#         """Lấy TwoPacket theo cặp user"""
#         pairId = f"{sourceUserId}_{destinationUserId}"
#         return self.get_two_packet(pairId)
    
#     def save_to_json(self, filename: str = "two_packets.json"):
#         """Lưu tất cả TwoPackets vào file JSON"""
#         data = [packet.to_json_dict() for packet in self.packets.values()]
#         with open(filename, "w") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#         print(f"✅ Saved {len(data)} TwoPackets to {filename}")
    
#     def load_from_json(self, filename: str = "two_packets.json"):
#         """Tải TwoPackets từ file JSON"""
#         try:
#             with open(filename, "r") as f:
#                 data = json.load(f)
            
#             self.packets = {}
#             for item in data:
#                 two_packet = TwoPacket.from_dict(item)
#                 self.packets[two_packet.pairId] = two_packet
            
#             print(f"✅ Loaded {len(self.packets)} TwoPackets from {filename}")
#         except FileNotFoundError:
#             print(f"⚠️ File {filename} not found, starting with empty collection")
#         except Exception as e:
#             print(f"❌ Error loading TwoPackets: {e}")

# # Ví dụ sử dụng
# if __name__ == "__main__":
#     # Tạo manager
#     manager = TwoPacketManager()
    
#     # Tạo TwoPacket cho cặp user
#     two_packet = manager.create_two_packet("user-Singapore", "user-Hanoi")
    
#     # Cập nhật thông tin packet Dijkstra
#     two_packet.update_dijkstra_packet(
#         sourceUserId="user-Singapore",
#         destinationUserId="user-Hanoi",
#         path=["GS_SINGAPORE", "LEO-01", "MEO-05", "GS_HANOI"],
#         totalDelay=45.2,
#         totalPacketLoss=0.02,
#         bandwidth=150.5
#     )
    
#     # Cập nhật thông tin packet RL
#     two_packet.update_rl_packet(
#         sourceUserId="user-Singapore",
#         destinationUserId="user-Hanoi",
#         path=["GS_SINGAPORE", "LEO-03", "GEO-01", "GS_HANOI"],
#         totalDelay=38.7,
#         totalPacketLoss=0.015,
#         bandwidth=165.8
#     )
    
#     # So sánh hiệu năng
#     comparison = two_packet.compare_performance()
#     print("📊 Performance Comparison:")
#     print(json.dumps(comparison, indent=2))
    
#     # Lưu vào file
#     manager.save_to_json()