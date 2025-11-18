# utils/db_connector.py

from pymongo import MongoClient, UpdateOne
from typing import List, Dict, Optional, Any
import logging
import os
from datetime import datetime, timezone
from bson import ObjectId

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
        db_name: str = "network",
        nodes_collection_name: str = "network_nodes",
    ):
        """
        Initialize MongoConnector. The MongoDB URI is taken in this order:
        1) the `uri` parameter if provided,
        2) the `MONGODB_URI` environment variable,
        3) the `MONGO_URI` environment variable.

        Raises:
            ValueError: If no MongoDB URI is provided via parameter or environment variables.
        """

        # resolve URI from parameter or environment
        if uri:
            resolved_uri = uri
        elif os.getenv("MONGODB_URI"):
            resolved_uri = os.getenv("MONGODB_URI")
        elif os.getenv("MONGO_URI"):
            resolved_uri = os.getenv("MONGO_URI")
        else:
            raise ValueError(
                "No MongoDB URI provided. Please set MONGODB_URI or MONGO_URI environment variable, "
                "or provide the uri parameter to MongoConnector."
            )

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

    # --- Node Update Methods ---

    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """
        Cập nhật thông tin của một Node.
        :param node_id: ID của Node cần cập nhật
        :param updates: Dictionary chứa các field cần cập nhật
        :return: True nếu thành công, False nếu thất bại
        """
        try:
            # Thêm timestamp cập nhật
            updates["lastUpdated"] = datetime.now(timezone.utc).isoformat()
            
            result = self.nodes_collection.update_one(
                {"nodeId": node_id},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated node: {node_id}")
                return True
            else:
                logger.warning(f"No node found with ID: {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating node {node_id}: {e}")
            return False

    def update_node_status(self, node_id: str, is_operational: bool, healthy: bool = None) -> bool:
        """
        Cập nhật trạng thái hoạt động của Node.
        :param node_id: ID của Node
        :param is_operational: Trạng thái hoạt động
        :param healthy: Trạng thái sức khỏe (optional)
        :return: True nếu thành công
        """
        updates = {"isOperational": is_operational}
        if healthy is not None:
            updates["healthy"] = healthy
            
        return self.update_node(node_id, updates)

    def update_node_metrics(self, node_id: str, battery: int = None, delay: float = None, 
                           loss_rate: float = None, utilization: float = None, 
                           packet_count: int = None, weather: str = None) -> bool:
        """
        Cập nhật các metrics của Node.
        :param node_id: ID của Node
        :param battery: Phần trăm pin (0-100)
        :param delay: Độ trễ xử lý (ms)
        :param loss_rate: Tỷ lệ mất gói
        :param utilization: Mức độ sử dụng tài nguyên (0.0-1.0)
        :param packet_count: Số lượng packet hiện tại
        :param weather: Điều kiện thời tiết
        :return: True nếu thành công
        """
        updates = {}
        
        if battery is not None:
            updates["batteryChargePercent"] = max(0, min(100, battery))
        if delay is not None:
            updates["nodeProcessingDelayMs"] = delay
        if loss_rate is not None:
            updates["packetLossRate"] = loss_rate
        if utilization is not None:
            updates["resourceUtilization"] = utilization
        if packet_count is not None:
            updates["currentPacketCount"] = packet_count
        if weather is not None:
            updates["weather"] = weather
            
        return self.update_node(node_id, updates) if updates else True

    def update_node_neighbors(self, node_id: str, neighbors: List[str]) -> bool:
        """
        Cập nhật danh sách neighbors của Node.
        :param node_id: ID của Node
        :param neighbors: Danh sách nodeId của neighbors
        :return: True nếu thành công
        """
        return self.update_node(node_id, {"neighbors": neighbors})

    def add_node_neighbor(self, node_id: str, neighbor_id: str) -> bool:
        """
        Thêm một neighbor vào danh sách neighbors của Node.
        :param node_id: ID của Node
        :param neighbor_id: ID của neighbor cần thêm
        :return: True nếu thành công
        """
        try:
            result = self.nodes_collection.update_one(
                {"nodeId": node_id},
                {
                    "$addToSet": {"neighbors": neighbor_id},
                    "$set": {"lastUpdated": datetime.now(timezone.utc).isoformat()}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error adding neighbor {neighbor_id} to node {node_id}: {e}")
            return False

    def remove_node_neighbor(self, node_id: str, neighbor_id: str) -> bool:
        """
        Xóa một neighbor khỏi danh sách neighbors của Node.
        :param node_id: ID của Node
        :param neighbor_id: ID của neighbor cần xóa
        :return: True nếu thành công
        """
        try:
            result = self.nodes_collection.update_one(
                {"nodeId": node_id},
                {
                    "$pull": {"neighbors": neighbor_id},
                    "$set": {"lastUpdated": datetime.now(timezone.utc).isoformat()}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error removing neighbor {neighbor_id} from node {node_id}: {e}")
            return False

    def bulk_update_nodes(self, updates: List[Dict[str, Any]]) -> bool:
        """
        Cập nhật nhiều nodes cùng lúc.
        :param updates: List các dict chứa node_id và các field cần cập nhật
                      [{"node_id": "node1", "updates": {"field": value}}, ...]
        :return: True nếu thành công
        """
        try:
            bulk_operations = []
            
            for update_data in updates:
                node_id = update_data["node_id"]
                updates_dict = update_data["updates"]
                updates_dict["lastUpdated"] = datetime.now(timezone.utc).isoformat()
                
                bulk_operations.append(
                    UpdateOne(
                        {"nodeId": node_id},
                        {"$set": updates_dict}
                    )
                )
            
            if bulk_operations:
                result = self.nodes_collection.bulk_write(bulk_operations)
                logger.info(f"Bulk updated {result.modified_count} nodes")
                return True
            return True
            
        except Exception as e:
            logger.error(f"Error in bulk update: {e}")
            return False

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
        # Thêm timestamp
        user_data["lastUpdated"] = datetime.now(timezone.utc).isoformat()
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
            # Thêm timestamp cho mỗi user
            for user_data in users_data:
                user_data["lastUpdated"] = datetime.now(timezone.utc).isoformat()
            users_collection.insert_many(users_data)
        logger.info(f"Cleared collection and inserted {len(users_data)} new users.")

    # --- User Update Methods ---

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Cập nhật thông tin của một User.
        :param user_id: ID của User cần cập nhật
        :param updates: Dictionary chứa các field cần cập nhật
        :return: True nếu thành công, False nếu thất bại
        """
        try:
            users_collection = self.db["users"]
            
            # Thêm timestamp cập nhật
            updates["lastUpdated"] = datetime.now(timezone.utc).isoformat()
            updates["lastActivity"] = datetime.now(timezone.utc).isoformat()
            
            result = users_collection.update_one(
                {"userId": user_id},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated user: {user_id}")
                return True
            else:
                logger.warning(f"No user found with ID: {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            return False

    def update_user_connection_status(self, user_id: str, connection_status: str, 
                                    connected_node_id: str = None) -> bool:
        """
        Cập nhật trạng thái kết nối của User.
        :param user_id: ID của User
        :param connection_status: Trạng thái kết nối ("CONNECTED", "DISCONNECTED", etc.)
        :param connected_node_id: ID của node đang kết nối (optional)
        :return: True nếu thành công
        """
        updates = {"connectionStatus": connection_status}
        if connected_node_id is not None:
            updates["connectedNodeId"] = connected_node_id
            
        return self.update_user(user_id, updates)

    def update_user_network_metrics(self, user_id: str, latency_ms: float = None, 
                                  bandwidth_mbps: float = None, packet_loss_rate: float = None) -> bool:
        """
        Cập nhật các metrics mạng của User.
        :param user_id: ID của User
        :param latency_ms: Độ trễ (ms)
        :param bandwidth_mbps: Băng thông (Mbps)
        :param packet_loss_rate: Tỷ lệ mất gói
        :return: True nếu thành công
        """
        updates = {}
        
        if latency_ms is not None:
            updates["latencyMs"] = latency_ms
        if bandwidth_mbps is not None:
            updates["bandwidthMbps"] = bandwidth_mbps
        if packet_loss_rate is not None:
            updates["packetLossRate"] = packet_loss_rate
            
        return self.update_user(user_id, updates) if updates else True

    def update_user_session(self, user_id: str, session_duration: float = None, 
                          data_consumed_mb: float = None) -> bool:
        """
        Cập nhật thông tin session của User.
        :param user_id: ID của User
        :param session_duration: Thời gian session (giây)
        :param data_consumed_mb: Dữ liệu đã tiêu thụ (MB)
        :return: True nếu thành công
        """
        updates = {}
        
        if session_duration is not None:
            updates["sessionDuration"] = session_duration
        if data_consumed_mb is not None:
            updates["dataConsumedMB"] = data_consumed_mb
            
        return self.update_user(user_id, updates) if updates else True

    def connect_user_to_node(self, user_id: str, node_id: str, 
                           latency_ms: float = None, bandwidth_mbps: float = None) -> bool:
        """
        Kết nối User đến một Node.
        :param user_id: ID của User
        :param node_id: ID của Node
        :param latency_ms: Độ trễ kết nối (ms)
        :param bandwidth_mbps: Băng thông kết nối (Mbps)
        :return: True nếu thành công
        """
        updates = {
            "connectedNodeId": node_id,
            "connectionStatus": "CONNECTED",
            "isActive": True
        }
        
        if latency_ms is not None:
            updates["latencyMs"] = latency_ms
        if bandwidth_mbps is not None:
            updates["bandwidthMbps"] = bandwidth_mbps
            
        return self.update_user(user_id, updates)

    def disconnect_user(self, user_id: str) -> bool:
        """
        Ngắt kết nối User.
        :param user_id: ID của User
        :return: True nếu thành công
        """
        updates = {
            "connectedNodeId": None,
            "connectionStatus": "DISCONNECTED"
        }
        
        return self.update_user(user_id, updates)

    def bulk_update_users(self, updates: List[Dict[str, Any]]) -> bool:
        """
        Cập nhật nhiều users cùng lúc.
        :param updates: List các dict chứa user_id và các field cần cập nhật
                      [{"user_id": "user1", "updates": {"field": value}}, ...]
        :return: True nếu thành công
        """
        try:
            users_collection = self.db["users"]
            bulk_operations = []
            
            for update_data in updates:
                user_id = update_data["user_id"]
                updates_dict = update_data["updates"]
                updates_dict["lastUpdated"] = datetime.now(timezone.utc).isoformat()
                updates_dict["lastActivity"] = datetime.now(timezone.utc).isoformat()
                
                bulk_operations.append(
                    UpdateOne(
                        {"userId": user_id},
                        {"$set": updates_dict}
                    )
                )
            
            if bulk_operations:
                result = users_collection.bulk_write(bulk_operations)
                logger.info(f"Bulk updated {result.modified_count} users")
                return True
            return True
            
        except Exception as e:
            logger.error(f"Error in bulk user update: {e}")
            return False

    # --- Utility Methods ---

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về collections trong database.
        :return: Dictionary chứa thông tin thống kê
        """
        stats = {}
        
        # Node collection stats
        nodes_count = self.nodes_collection.count_documents({})
        operational_nodes = self.nodes_collection.count_documents({"isOperational": True})
        healthy_nodes = self.nodes_collection.count_documents({"healthy": True})
        
        stats["nodes"] = {
            "total": nodes_count,
            "operational": operational_nodes,
            "healthy": healthy_nodes
        }
        
        # User collection stats
        users_collection = self.db["users"]
        users_count = users_collection.count_documents({})
        connected_users = users_collection.count_documents({"connectionStatus": "CONNECTED"})
        active_users = users_collection.count_documents({"isActive": True})
        
        stats["users"] = {
            "total": users_count,
            "connected": connected_users,
            "active": active_users
        }
        
        return stats

    def close_connection(self):
        """Đóng kết nối MongoDB."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()

# Example usage
if __name__ == "__main__":
    # Test the connector
    with MongoConnector() as connector:
        # Get all nodes
        nodes = connector.get_all_nodes()
        print(f"Total nodes: {len(nodes)}")
        
        # Update a node
        if nodes:
            node_id = nodes[0]["nodeId"]
            success = connector.update_node_metrics(
                node_id, 
                battery=85,
                delay=5.2,
                utilization=0.7
            )
            print(f"Node update successful: {success}")
        
        # Get collection stats
        stats = connector.get_collection_stats()
        print(f"Collection stats: {stats}")