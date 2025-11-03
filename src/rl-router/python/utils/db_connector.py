# utils/db_connector.py

from pymongo import MongoClient
from typing import List, Dict, Optional, Any
import logging
import os

# try to use python-dotenv to load a local .env in development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional in some deployments; environment variables may be provided externally
    pass

logger = logging.getLogger(__name__)

# Định nghĩa kiểu dữ liệu cho một document Node
NodeData = Dict[str, Any]

class MongoConnector:
    """Quản lý kết nối và truy vấn dữ liệu Node từ MongoDB."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: str = "sagsin_network",
        nodes_collection_name: str = "network_nodes",
    ):
        """
        Initialize MongoConnector. The MongoDB URI is taken in this order:
        1) the `uri` parameter if provided,
        2) the `MONGODB_URI` environment variable (loaded from .env by python-dotenv if present),
        3) a sensible default used previously.
        """

        # resolve URI from parameter or environment (support both MONGODB_URI and MONGO_URI)
        resolved_uri = (
            uri
            or os.getenv("MONGODB_URI")
            or os.getenv("MONGO_URI")
            or "mongodb://user:password123@localhost:27017/?authSource=admin"
        )

        self.client = MongoClient(resolved_uri)
        self.db = self.client[db_name]
        self.nodes_collection = self.db[nodes_collection_name]
        # avoid logging credentials; log that connection was established (host part may contain creds)
        logger.info("Connected to MongoDB, DB: %s, Collection: %s", db_name, nodes_collection_name)

    def get_node(self, node_id: str, projection: Optional[Dict[str, int]] = None) -> Optional[NodeData]:
        """
        Lấy 1 Node theo nodeId.
        :param node_id: ID của Node
        :param projection: Optional, dict các field muốn lấy {field: 1}
        """
        return self.nodes_collection.find_one({"nodeId": node_id}, projection)

    def get_all_nodes(self, projection: Optional[Dict[str, int]] = None) -> List[NodeData]:
        """
        Lấy tất cả các Node trong Collection.
        :param projection: Optional, dict các field muốn lấy {field: 1}
        """
        return list(self.nodes_collection.find({}, projection))

    def get_nodes_by_type(self, node_type: str, projection: Optional[Dict[str, int]] = None) -> List[NodeData]:
        """
        Lấy tất cả Node có nodeType cụ thể.
        :param node_type: Kiểu Node (ví dụ "SAT_LEO", "GROUND_STATION")
        :param projection: Optional, dict các field muốn lấy {field: 1}
        """
        return list(self.nodes_collection.find({"nodeType": node_type}, projection))
    
    def get_nodes_by_status(self, operational: bool, projection: Optional[Dict[str, int]] = None) -> List[NodeData]:
        """
        Lấy tất cả Node có trạng thái hoạt động (isOperational) cụ thể.
        :param operational: True nếu muốn lấy các node đang hoạt động, False nếu không.
        :param projection: Optional, dict các field muốn lấy {field: 1}
        """
        # Đã sửa lỗi chính tả và sử dụng trường 'isOperational'
        return list(self.nodes_collection.find({"operational": operational}, projection))
    
    # --- Hàm Lấy Node Lân cận và Batch (Tối ưu cho RL) ---

    def get_neighbor_status_batch(self, neighbor_ids: List[str], projection: Optional[Dict[str, int]] = None) -> Dict[str, NodeData]:
        """
        Lấy trạng thái Node chi tiết (batch) cho một danh sách các Node ID.
        Sử dụng $in để tối ưu hóa truy vấn cho tất cả các node lân cận.
        
        :param neighbor_ids: Danh sách các nodeId cần fetch trạng thái.
        :param projection: Optional, dict các field muốn lấy {field: 1}.
        :return: Dict chứa dữ liệu Node, với nodeId là key.
        """
        neighbors_data = {}
        
        # Chỉ truy vấn các node có ID nằm trong danh sách neighbor_ids
        results = self.nodes_collection.find({"nodeId": {"$in": neighbor_ids}}, projection)
        
        for doc in results:
            neighbors_data[doc["nodeId"]] = doc
            
        return neighbors_data

    def get_node_neighbors(
        self, node_id: str, projection: Optional[Dict[str, int]] = None
    ) -> Dict[str, NodeData]:
        """
        Lấy danh sách Node neighbors DƯỚI DẠNG DICT dựa trên field 'neighbors' trong document 
        và fetch trạng thái của chúng trong một batch duy nhất.
        
        :param node_id: ID của node nguồn.
        :param projection: Optional, dict các field muốn lấy {field: 1} cho các node lân cận.
        :return: Dict các NodeData của Neighbors, với nodeId là key.
        """
        # 1. Lấy danh sách ID Neighbors từ Node hiện tại
        node = self.get_node(node_id, projection={"neighbors": 1})
        if not node or "neighbors" not in node:
            return {}

        neighbor_ids = node.get("neighbors", [])
        
        # 2. Fetch trạng thái chi tiết của tất cả Neighbors trong 1 batch
        return self.get_neighbor_status_batch(neighbor_ids, projection)
    