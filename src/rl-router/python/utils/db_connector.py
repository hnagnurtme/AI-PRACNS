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

# Define MongoDB URIs
LOCAL_MONGO_URI = "mongodb://user:password123@localhost:27017"

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
        3) the CLOUD_MONGO_URI if use_cloud_db is True,
        4) the LOCAL_MONGO_URI otherwise.
        """

        # resolve URI from parameter or environment
        if uri:
            resolved_uri = uri
        elif os.getenv("MONGODB_URI"):
            resolved_uri = os.getenv("MONGODB_URI")
        elif os.getenv("MONGO_URI"):
            resolved_uri = os.getenv("MONGO_URI")
        else:
            resolved_uri = LOCAL_MONGO_URI

        self.client = MongoClient(resolved_uri)
        self.db = self.client[db_name]
        self.nodes_collection = self.db[nodes_collection_name]
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
        # Sử dụng trường 'isOperational' để truy vấn
        return list(self.nodes_collection.find({"isOperational": operational}, projection))

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

    def get_nodes(self, node_ids: List[str], projection: Optional[Dict[str, int]] = None) -> List[NodeData]:
        """
        Lấy nhiều nodes theo danh sách IDs.
        :param node_ids: Danh sách các nodeId cần fetch.
        :param projection: Optional, dict các field muốn lấy {field: 1}.
        :return: List các NodeData.
        """
        if not node_ids:
            return []
        results = self.nodes_collection.find({"nodeId": {"$in": node_ids}}, projection)
        return list(results)

    def clear_and_insert_nodes(self, nodes_data: List[NodeData]):
        """
        Xóa tất cả các document trong collection và insert list các node mới.
        :param nodes_data: List các dictionary data của nodes.
        """
        self.nodes_collection.delete_many({})
        if nodes_data:
            self.nodes_collection.insert_many(nodes_data)
        logger.info(f"Cleared collection and inserted {len(nodes_data)} new nodes.")

    # --- User Management Methods ---

    def get_user(self, user_id: str, projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        """
        Lấy 1 User theo userId.
        :param user_id: ID của User
        :param projection: Optional, dict các field muốn lấy {field: 1}
        """
        users_collection = self.db["users"]
        return users_collection.find_one({"userId": user_id}, projection)

    def get_user_by_city(self, city_name: str, projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        """
        Lấy User theo tên thành phố.
        :param city_name: Tên thành phố
        :param projection: Optional, dict các field muốn lấy {field: 1}
        """
        users_collection = self.db["users"]
        return users_collection.find_one({"cityName": city_name}, projection)

    def get_all_users(self, projection: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Lấy tất cả Users trong Collection.
        :param projection: Optional, dict các field muốn lấy {field: 1}
        """
        users_collection = self.db["users"]
        return list(users_collection.find({}, projection))

    def insert_user(self, user_data: Dict[str, Any]) -> str:
        """
        Thêm một User mới vào database.
        :param user_data: Dictionary chứa thông tin user
        :return: ID của user được insert
        """
        users_collection = self.db["users"]
        result = users_collection.insert_one(user_data)
        return str(result.inserted_id)

    def clear_and_insert_users(self, users_data: List[Dict[str, Any]]):
        """
        Xóa tất cả users và insert danh sách users mới.
        :param users_data: List các dictionary data của users.
        """
        users_collection = self.db["users"]
        users_collection.delete_many({})
        if users_data:
            users_collection.insert_many(users_data)
        logger.info(f"Cleared collection and inserted {len(users_data)} new users.")