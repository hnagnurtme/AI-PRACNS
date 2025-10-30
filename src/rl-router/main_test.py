import logging
import os
# Đảm bảo đường dẫn import này là chính xác
from python.utils.db_connector import MongoConnector 
from typing import Dict, Any

# --- 1. Cấu hình Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. Dữ liệu Kết nối (Cần có dữ liệu thực trong DB) ---
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://user:password123@localhost:27017/?authSource=admin"
)

# --- 3. ID Kiểm thử (Phải tồn tại trong DB của bạn) ---
TEST_NODE_ID = "SAT-LEO-2"
TEST_DEST_ID = "GS_HANOI"
TEST_NODE_TYPE = "LEO_SATELLITE"  # nếu muốn test theo type
TEST_OPERATIONAL_STATUS = True # Test trạng thái hoạt động


def print_node_data(data: Dict[str, Any]):
    """Hàm helper để in dữ liệu node gọn gàng hơn."""
    print({k: v for k, v in data.items() if k != '_id'})


def test_mongo_connector_load_only():
    """Kiểm tra tất cả các hàm truy vấn trên dữ liệu hiện có."""
    
    logger.info("Starting MongoDB connection test...")
    connector = None
    try:
        # 1. KHỞI TẠO KẾT NỐI
        connector = MongoConnector(uri=MONGO_URI)

        # 2. TEST get_node()
        print("\n--- TEST: 2. get_node (Cơ bản) ---")
        node_status = connector.get_node(TEST_NODE_ID)
        if node_status:
            print(f"✅ Trạng thái Node {TEST_NODE_ID} đã được fetch.")
            print_node_data(node_status)
        else:
            print(f"❌ LỖI: Không tìm thấy Node ID '{TEST_NODE_ID}'. Vui lòng kiểm tra dữ liệu DB.")
            assert False, "Node ID không tồn tại"

        # 3. TEST get_all_nodes()
        print("\n--- TEST: 3. get_all_nodes (Tổng quan) ---")
        all_nodes = connector.get_all_nodes(projection={"nodeId": 1}) # Chỉ lấy nodeId
        print(f"✅ Tổng số Nodes trong mạng: {len(all_nodes)}")
        assert isinstance(all_nodes, list)

        # 4. TEST get_nodes_by_type()
        print(f"\n--- TEST: 4. get_nodes_by_type ({TEST_NODE_TYPE}) ---")
        type_nodes = connector.get_nodes_by_type(TEST_NODE_TYPE, projection={"nodeId": 1, "nodeType": 1})
        print(f"✅ Tổng số Nodes type '{TEST_NODE_TYPE}': {len(type_nodes)}")
        for n in type_nodes[:2]:  # chỉ in 2 node đầu
            print_node_data(n)
            
        # 5. TEST get_nodes_by_status()
        print(f"\n--- TEST: 5. get_nodes_by_status (isOperational={TEST_OPERATIONAL_STATUS}) ---")
        op_nodes = connector.get_nodes_by_status(TEST_OPERATIONAL_STATUS, projection={"nodeId": 1, "isOperational": 1})
        print(f"✅ Tổng số Nodes đang hoạt động: {len(op_nodes)}")
        for n in op_nodes[:2]:  # chỉ in 2 node đầu
            print_node_data(n)

        # 6. TEST get_node_neighbors()
        print(f"\n--- TEST: 6. get_node_neighbors ({TEST_NODE_ID}) ---")
        neighbors = connector.get_node_neighbors(TEST_NODE_ID)
        print(f"✅ Node {TEST_NODE_ID} có {len(neighbors)} neighbors.")
        if neighbors:
             # Lấy 1 neighbor đầu tiên để in
            first_neighbor_id = next(iter(neighbors.keys())) 
            print(f"  - Dữ liệu Neighbor mẫu ({first_neighbor_id}):")
            print_node_data(neighbors[first_neighbor_id])
        else:
            print("⚠️ CẢNH BÁO: Không tìm thấy neighbors. Kiểm tra field 'neighbors' trong document Node.")
        
        logger.info("\n✅ TẤT CẢ CÁC TRUY VẤN ĐÃ CHẠY THÀNH CÔNG!")

    except Exception as e:
        logger.error(f"❌ TEST THẤT BẠI: {e}", exc_info=True)

    finally:
        if connector:
            connector.client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    test_mongo_connector_load_only()