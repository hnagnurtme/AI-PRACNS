import logging
from python.utils.db_connector import MongoConnector
from python.utils.state_builder import convert_to_ecef, calculate_distance_km

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CÁC NODE CẦN KIỂM TRA ---
NODE_A_ID = "GS_HANOI"
NODE_B_ID = "GS_SYDNEY"     # Node đi lang thang
NODE_DEST_ID = "GS_HOCHIMINH" # Đích đến

def run_debug():
    logger.info("--- BẮT ĐẦU KIỂM TRA TỌA ĐỘ DATABASE ---")
    
    try:
        mongo_conn = MongoConnector()
        logger.info("✅ Kết nối MongoDB thành công.")
    except Exception as e:
        logger.error(f"❌ Lỗi kết nối MongoDB: {e}")
        return

    # 1. Lấy dữ liệu position từ DB
    node_a = mongo_conn.get_node(NODE_A_ID, projection={"position": 1, "nodeId": 1})
    node_b = mongo_conn.get_node(NODE_B_ID, projection={"position": 1, "nodeId": 1})
    node_dest = mongo_conn.get_node(NODE_DEST_ID, projection={"position": 1, "nodeId": 1})

    if not node_a or not node_b or not node_dest:
        logger.error("❌ LỖI: Không tìm thấy 1 trong 3 node (HANOI, SYDNEY, HOCHIMINH).")
        logger.error("Hãy đảm bảo các Node ID này tồn tại trong DB.")
        return

    pos_a = node_a.get("position", {})
    pos_b = node_b.get("position", {})
    pos_dest = node_dest.get("position", {})

    logger.info(f"Dữ liệu Lat/Lon/Alt (từ DB):")
    logger.info(f"  [{NODE_A_ID}]: {pos_a}")
    logger.info(f"  [{NODE_B_ID}]: {pos_b}")
    logger.info(f"  [{NODE_DEST_ID}]: {pos_dest}")
    print("-" * 30)

    # 2. Chuyển đổi sang ECEF (x,y,z)
    try:
        ecef_a = convert_to_ecef(pos_a)
        ecef_b = convert_to_ecef(pos_b)
        ecef_dest = convert_to_ecef(pos_dest)
        
        logger.info("Tọa độ ECEF (x,y,z) (đã chuyển đổi):")
        logger.info(f"  [{NODE_A_ID}]: {ecef_a}")
        logger.info(f"  [{NODE_B_ID}]: {ecef_b}")
        logger.info(f"  [{NODE_DEST_ID}]: {ecef_dest}")
        print("-" * 30)
    except Exception as e:
        logger.error(f"❌ LỖI khi chuyển đổi ECEF: {e}")
        logger.error("Kiểm tra lại hàm 'convert_to_ecef' hoặc dữ liệu đầu vào (Lat/Lon).")
        return

    # 3. Tính toán khoảng cách
    # Đây là 2 con số mà Agent dùng để so sánh
    
    # Khoảng cách từ (A) Hanoi -> (Đích) HCMC
    dist_a_to_dest = calculate_distance_km(ecef_a, ecef_dest)
    
    # Khoảng cách từ (B) Sydney -> (Đích) HCMC
    dist_b_to_dest = calculate_distance_km(ecef_b, ecef_dest)

    logger.info("--- KẾT QUẢ TÍNH TOÁN KHOẢNG CÁCH ---")
    logger.info(f"Khoảng cách từ '{NODE_A_ID}' đến '{NODE_DEST_ID}': {dist_a_to_dest:,.2f} km")
    logger.info(f"Khoảng cách từ '{NODE_B_ID}' đến '{NODE_DEST_ID}': {dist_b_to_dest:,.2f} km")
    print("-" * 30)

    # 4. Chẩn đoán
    if dist_b_to_dest < dist_a_to_dest:
        logger.error("❌❌❌ VẤN ĐỀ ĐÃ ĐƯỢC XÁC NHẬN ❌❌❌")
        logger.error("Dữ liệu tọa độ trong DB của bạn BỊ SAI.")
        logger.error(f"Theo tính toán, '{NODE_B_ID}' (Sydney) đang ở GẦN '{NODE_DEST_ID}' (HCMC) hơn là '{NODE_A_ID}' (Hanoi).")
        logger.error("Đây là lý do Agent đi lang thang. Hãy sửa lại tọa độ (Lat/Lon) trong Database.")
    else:
        logger.info("✅ Dữ liệu tọa độ có vẻ hợp lý (Sydney xa hơn Hanoi).")
        logger.info("Nếu kết quả này đúng, thì vấn đề 100% là do bạn đang dùng model CŨ.")
        logger.info("Hãy XÓA file '..._latest.pth' và huấn luyện lại từ đầu.")

if __name__ == "__main__":
    run_debug()