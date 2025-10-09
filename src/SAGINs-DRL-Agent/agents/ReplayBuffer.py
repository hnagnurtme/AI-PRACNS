# agents/ReplayBuffer.py (Đã sửa chữa)
from pymongo.collection import Collection
import numpy as np
from typing import Dict, Any, List, Tuple
import random
import time

class ReplayBuffer:
    
    def __init__(self, mongo_collection: Collection, capacity: int = 100000):
        self.collection = mongo_collection
        self.capacity = capacity

    def store_experience(self, experience: Dict):
        """Lưu kinh nghiệm mới vào MongoDB và quản lý dung lượng."""
        self.collection.insert_one(experience)
        
        # 1. Kiểm tra kích thước buffer bằng phương pháp hiện đại
        if self.collection.estimated_document_count() > self.capacity: 
            try:
                # 2. Tìm tài liệu cũ nhất (dựa trên _id thấp nhất)
                # Sử dụng next() trên cursor để lấy tài liệu đầu tiên
                oldest_doc_cursor = self.collection.find({}).sort("_id", 1).limit(1)
                
                # 3. Kiểm tra và thực hiện xóa
                oldest_doc = next(oldest_doc_cursor, None) # Lấy tài liệu đầu tiên, hoặc None
                
                if oldest_doc is not None:
                    oldest_doc_id = oldest_doc["_id"]
                    # Xóa tài liệu bằng ID
                    self.collection.delete_one({"_id": oldest_doc_id})
                    
            except Exception as e:
                print(f"Lỗi khi xóa kinh nghiệm cũ nhất: {e}")

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, str, float, np.ndarray]]:
        # ... (Phần này vẫn giữ nguyên vì sử dụng aggregation $sample) ...
        pipeline = [{ '$sample': { 'size': batch_size } }]
        experiences = list(self.collection.aggregate(pipeline))
        # ... (logic chuyển đổi sang NumPy) ...
        
        samples = []
        for exp in experiences:
            s = np.array(exp['stateVectorS'], dtype=np.float32)
            a = exp['actionTakenA'] # Node ID (string)
            r = exp['rewardR']
            s_prime = np.array(exp['nextStateVectorSPrime'], dtype=np.float32)
            samples.append((s, a, r, s_prime))
            
        return samples

    def get_size(self) -> int:
        """Trả về số lượng kinh nghiệm hiện tại."""
        # Sử dụng phương thức hiện đại để đếm tài liệu
        return self.collection.estimated_document_count()