import streamlit as st
import base64
from datetime import datetime, timezone
import time
import sys
import queue
import math
import socket
import logging
from typing import List, Dict, Any, Optional

# ==============================================================================
# PHẦN 1: CẤU HÌNH BAN ĐẦU VÀ IMPORTS
# ==============================================================================

# Cấu hình hệ thống ghi log (logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # In log ra terminal
)
logger = logging.getLogger(__name__)

# Thêm đường dẫn dự án để import các module
sys.path.insert(0, '.')
try:
    from models.node import Node, Position, Communication as NodeCommunication, Status as NodeStatus, NodeType, Velocity, Metadata
    from models.user import User, UserPosition
    from models.packet import Packet, ServiceQoS ,get_qos_profile,ServiceType
    from service.TCP_Listener import TCPListener
    from service.TCP_Sender import send_packet_via_tcp
    from service.incoming_queue import GLOBAL_INCOMING_QUEUE
    from config.mongo_config import get_collection
except ImportError as e:
    # Lỗi nghiêm trọng, dừng ứng dụng nếu không import được
    st.error(f"LỖI IMPORT NGHIÊM TRỌNG: {e}")
    st.error("Hãy đảm bảo cấu trúc thư mục đúng và các file model tồn tại (ví dụ: 'models/node.py').")
    st.stop()

from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# PHẦN 2: CẤU HÌNH VÀ HẰNG SỐ
# ==============================================================================
REFRESH_INTERVAL_MS = 2500
SERVICE_TYPES = ["VIDEO_STREAM", "AUDIO_CALL", "IMAGE_TRANSFER", "TEXT_MESSAGE"]
DEFAULT_LISTEN_PORT = 9001

# ==============================================================================
# PHẦN 3: CÁC HÀM XỬ LÝ LOGIC VÀ TƯƠNG TÁC DATABASE
# ==============================================================================

def update_user_status_in_db(user_id: str, host: str, port: int, is_active: bool):
    """
    Cập nhật trạng thái và thông tin kết nối của một user lên MongoDB.

    Args:
        user_id (str): ID của user cần cập nhật.
        host (str): Địa chỉ IP mới.
        port (int): Cổng mới.
        is_active (bool): Trạng thái online (True) hay offline (False).
    """
    try:
        users_collection = get_collection("users")
        update_data = {
            "$set": {
                "status.active": is_active,
                "communication.ipAddress": host,
                "communication.port": port,
                "status.lastSeen": datetime.now(timezone.utc).isoformat()
            }
        }
        result = users_collection.update_one({"userId": user_id}, update_data)
        status_text = "ONLINE" if is_active else "OFFLINE"
        
        if result.modified_count > 0 or result.matched_count > 0:
            log_msg = f"Đã cập nhật User '{user_id}' thành {status_text} (Port: {port}) trên DB."
            logger.info(log_msg)
            st.toast(f"✅ {log_msg}")
        else:
            logger.warning(f"Không tìm thấy User '{user_id}' trên DB để cập nhật.")

    except Exception as e:
        logger.error(f"Lỗi khi cập nhật trạng thái User '{user_id}': {e}")
        st.error(f"Lỗi DB: Không thể cập nhật User '{user_id}'.")

def haversine_distance(pos1: Any, pos2: Any) -> float:
    """
    Tính khoảng cách Haversine (km) giữa hai điểm tọa độ địa lý.

    Args:
        pos1 (Any): Đối tượng có thuộc tính latitude và longitude (ví dụ: Position).
        pos2 (Any): Đối tượng có thuộc tính latitude và longitude.

    Returns:
        float: Khoảng cách giữa hai điểm (km).
    """
    R = 6371  # Bán kính Trái Đất
    lat1, lon1 = math.radians(pos1.latitude), math.radians(pos1.longitude)
    lat2, lon2 = math.radians(pos2.latitude), math.radians(pos2.longitude)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_nearest_ground_station(user_pos: UserPosition, stations: List[Node]) -> Optional[Node]:
    """
    Tìm trạm mặt đất (Node) gần nhất với vị trí của một user.

    Args:
        user_pos (UserPosition): Vị trí của user.
        stations (List[Node]): Danh sách các trạm mặt đất để tìm kiếm.

    Returns:
        Optional[Node]: Trạm mặt đất gần nhất, hoặc None nếu danh sách rỗng.
    """
    if not stations:
        return None
    return min(stations, key=lambda s: haversine_distance(user_pos, s.position))

@st.cache_data(ttl=10)
def load_from_db(collection_name: str, model_class: Any) -> List[Any]:
    """
    Hàm chung để tải dữ liệu từ MongoDB, chuyển đổi thành đối tượng và cache lại.

    Args:
        collection_name (str): Tên của collection ('users' hoặc 'nodes').
        model_class (Any): Class của model để chuyển đổi (User hoặc Node).

    Returns:
        List[Any]: Danh sách các đối tượng đã được tải và chuyển đổi.
    """
    logger.info(f"Đang tải dữ liệu từ collection: '{collection_name}'...")
    collection = get_collection(collection_name)
    query = {}
    if collection_name == "network_nodes":
        query = {"status.active": True, "type": NodeType.GROUND_STATION.value}
    
    items = []
    for doc in collection.find(query):
        try:
            items.append(model_class.from_dict(doc))
        except Exception as e:
            id_key = "nodeId" if hasattr(model_class, "nodeId") else "userId"
            logger.warning(f"Bỏ qua mục {doc.get(id_key)} do lỗi dữ liệu: {e}")
    logger.info(f"Tải thành công {len(items)} mục từ '{collection_name}'.")
    return items

# ==============================================================================
# PHẦN 4: CÁC HÀM XỬ LÝ SỰ KIỆN VÀ TƯƠNG TÁC NGƯỜI DÙNG
# ==============================================================================

def _create_packet_from_ui(source_user: User, dest_user: User, source_station: Node, dest_station: Node) -> Optional[Packet]:
    """
    Tạo một đối tượng Packet từ dữ liệu nhập trên giao diện.

    Args:
        source_user (User): Đối tượng user gửi.
        dest_user (User): Đối tượng user nhận.
        source_station (Node): Trạm mặt đất nguồn.
        dest_station (Node): Trạm mặt đất đích.

    Returns:
        Optional[Packet]: Đối tượng Packet nếu hợp lệ, ngược lại là None.
    """
    message = st.session_state.message_input
    if not message:
        st.error("Vui lòng nhập nội dung tin nhắn!"); return None

    payload_raw_bytes = message.encode('utf-8')
    payload_base64 = base64.b64encode(payload_raw_bytes).decode('utf-8')
    service_type = st.session_state.service_type_select
    qos = get_qos_profile(ServiceType(service_type))  

    return Packet(
        packetId=f"PKT-{int(time.time() * 1000)}", sourceUserId=source_user.userId,
        destinationUserId=dest_user.userId, stationSource=source_station.nodeId,
        stationDest=dest_station.nodeId, type="DATA", acknowledgedPacketId=None,
        timeSentFromSourceMs=int(time.time() * 1000), payloadDataBase64=payload_base64,
        payloadSizeByte=len(payload_raw_bytes), serviceQoS=qos,
        TTL=st.session_state.ttl_value, currentHoldingNodeId=source_station.nodeId,
        nextHopNodeId="", pathHistory=[source_station.nodeId], hopRecords=[],
        priorityLevel=st.session_state.priority_level_value, isUseRL=st.session_state.is_use_rl_toggle
    )

def handle_send_button_click():
    """
    Xử lý sự kiện khi người dùng nhấn nút "Gửi Gói tin".
    """
    logger.info("Nút 'Gửi Gói tin' được nhấn.")
    # 1. Lấy dữ liệu từ session_state
    source_user_id = st.session_state.active_user_id
    dest_user_id = st.session_state.dest_user_id
    if source_user_id == dest_user_id:
        st.error("Người gửi và người nhận không được trùng nhau!"); return

    # 2. Tìm các đối tượng tương ứng
    users_map = {u.userId: u for u in st.session_state.users}
    source_user, dest_user = users_map.get(source_user_id), users_map.get(dest_user_id)
    ground_stations = st.session_state.ground_stations

    if not source_user or not dest_user or not ground_stations:
        st.error("Thiếu thông tin người dùng hoặc trạm mặt đất."); return

    # 3. Tính toán trạm gần nhất
    source_station = find_nearest_ground_station(source_user.position, ground_stations)
    dest_station = find_nearest_ground_station(dest_user.position, ground_stations)
    if not source_station or not dest_station:
        st.error("Không thể xác định trạm mặt đất."); return

    # 4. Tạo và gửi gói tin
    new_packet = _create_packet_from_ui(source_user, dest_user, source_station, dest_station)
    if not new_packet: return
    
    print(new_packet.to_json())

    with st.spinner(f"Đang gửi gói tin đến trạm nguồn {source_station.nodeName}..."):
        error_message = send_packet_via_tcp(source_station.communication.ipAddress, source_station.communication.port, new_packet)
    
    # 5. Ghi log và thông báo kết quả
    log_entry: Dict[str, Any] = { "target": source_station.nodeName, "time": datetime.now().strftime("%H:%M:%S"), "packet_id": new_packet.packetId, "content": f"From {source_user.userName} to {dest_user.userName}" }
    if error_message:
        log_entry.update({"type": "error", "content": f"Gửi thất bại: {error_message}"})
        logger.error(f"Gửi gói tin {new_packet.packetId} thất bại: {error_message}")
        st.error(f"Gửi thất bại: {error_message}")
    else:
        log_entry.update({"type": "sent", "is_use_rl": new_packet.isUseRL, "ttl": new_packet.TTL})
        logger.info(f"Gói tin {new_packet.packetId} đã được gửi thành công đến trạm nguồn.")
        st.success(f"Gói tin đã gửi thành công đến trạm nguồn! ID: {new_packet.packetId}")
    
    st.session_state.chat_history.append(log_entry)
    st.session_state.message_input = ""

def background_thread_handler(received_data: Dict):
    """
    Callback function được gọi bởi TCPListener khi có gói tin đến.
    Nó đưa gói tin vào hàng đợi chung để luồng chính xử lý.
    """
    GLOBAL_INCOMING_QUEUE.put(received_data)

def check_for_incoming_messages():
    """
    Kiểm tra hàng đợi và xử lý các gói tin đến.
    """
    while not GLOBAL_INCOMING_QUEUE.empty():
        try:
            received_data = GLOBAL_INCOMING_QUEUE.get_nowait()
            addr, packet_json = received_data['source_addr'], received_data['packet_data']
            received_packet = Packet.from_json(packet_json) if isinstance(packet_json, str) else Packet(**packet_json)
            logger.info(f"Nhận được gói tin {received_packet.packetId} từ {addr}")
            
            log_entry = { "type": "received", "target": f"{addr[0]}:{addr[1]}", "content": received_packet.get_decoded_payload(), "packet_type": received_packet.type, "packet_id": received_packet.packetId, "time": datetime.now().strftime("%H:%M:%S") }
            st.session_state.chat_history.append(log_entry)
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Lỗi xử lý gói tin từ hàng đợi: {e}")
            st.session_state.chat_history.append({"type": "system_error", "content": f"Lỗi xử lý gói tin đến: {e}", "time": datetime.now().strftime("%H:%M:%S")})

# ==============================================================================
# PHẦN 5: KHỞI TẠO VÀ QUẢN LÝ TRẠNG THÁI STREAMLIT
# ==============================================================================

def initialize_session_state():
    """
    Khởi tạo các giá trị cần thiết trong st.session_state cho lần chạy đầu tiên.
    """
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'listener_status' not in st.session_state: st.session_state.listener_status = "Chưa khởi động"
    if 'listen_port' not in st.session_state: st.session_state.listen_port = DEFAULT_LISTEN_PORT
    if 'previous_user_id' not in st.session_state: st.session_state.previous_user_id = None

def manage_tcp_listener():
    """
    Quản lý vòng đời của TCP Listener: khởi động, dừng, và khởi động lại khi cần.
    Cập nhật trạng thái user lên DB khi listener hoạt động.
    """
    current_port = st.session_state.listen_port
    active_user_id = st.session_state.get("active_user_id")
    previous_user_id = st.session_state.get("previous_user_id")

    # 1. Cập nhật user cũ thành OFFLINE nếu có sự thay đổi user
    if previous_user_id and previous_user_id != active_user_id:
        logger.info(f"User thay đổi từ '{previous_user_id}' sang '{active_user_id}'. Cập nhật user cũ thành offline.")
        update_user_status_in_db(previous_user_id, "", 0, False)
    
    # 2. Lưu lại user hiện tại cho lần so sánh tiếp theo
    st.session_state.previous_user_id = active_user_id

    if not active_user_id:
        st.sidebar.warning("Vui lòng chọn một User để bắt đầu.")
        return

    # 3. Quản lý luồng Listener
    listener_running = 'tcp_listener' in st.session_state and st.session_state.tcp_listener.is_alive()
    needs_restart = not listener_running or st.session_state.tcp_listener.port != current_port

    if needs_restart and listener_running:
        logger.info(f"Port thay đổi, yêu cầu dừng listener cũ trên port {st.session_state.tcp_listener.port}")
        st.session_state.tcp_listener.stop()
        st.session_state.tcp_listener.join(timeout=2)
    
    if needs_restart:
        try:
            logger.info(f"Đang khởi động listener mới trên port {current_port}...")
            listener = TCPListener(host='0.0.0.0', port=current_port, handler=background_thread_handler)
            listener.start()
            st.session_state.tcp_listener = listener
            st.session_state.listener_status = f"Đang chạy (port {current_port})"
            
            try:
                my_ip = socket.gethostbyname(socket.gethostname())
            except socket.gaierror:
                my_ip = "127.0.0.1"
            
            # 4. Cập nhật user MỚI thành ONLINE
            update_user_status_in_db(user_id=active_user_id, host=my_ip, port=current_port, is_active=True)
            
        except Exception as e:
            logger.error(f"Không thể khởi động listener trên port {current_port}: {e}")
            st.session_state.listener_status = f"Lỗi trên port {current_port}: {e}"
            st.error(st.session_state.listener_status)

# ==============================================================================
# PHẦN 6: GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)
# ==============================================================================

def _display_chat_entry(entry: Dict[str, Any], chat_box):
    """
    Hiển thị một mục trong lịch sử chat.
    """
    type = entry.get('type')
    if type == 'sent':
        rl_tag = ":green[RL]" if entry.get('is_use_rl') else ":gray[No RL]"
        html = f"<div style='text-align: right; margin-bottom: 8px;'><div style='display: inline-block; text-align: left; border-radius: 10px; padding: 10px; background-color: #dcf8c6;'><span style='color: #556B2F; font-weight: bold;'>GỬI ► {entry['target']}</span> <span style='font-size: 0.8em; color: gray;'> {entry['time']}</span><p style='margin: 5px 0 0 0;'><b>Payload</b>: {entry['content']}</p><span style='font-size: 0.8em; color: teal;'>[ID: {entry['packet_id']}] [TTL: {entry['ttl']}, {rl_tag}]</span></div></div>"
        chat_box.markdown(html, unsafe_allow_html=True)
    elif type == 'received':
        html = f"<div style='text-align: left; margin-bottom: 8px;'><div style='display: inline-block; text-align: left; border-radius: 10px; padding: 10px; background-color: #f7f7ff;'><span style='color: blue; font-weight: bold;'>◄ NHẬN TỪ {entry['target']}</span> <span style='font-size: 0.8em; color: gray;'> {entry['time']}</span><p style='margin: 5px 0 0 0;'><b>Payload</b>: {entry['content']}</p><span style='font-size: 0.8em; color: #6A5ACD;'>[ID: {entry['packet_id']}]</span></div></div>"
        chat_box.markdown(html, unsafe_allow_html=True)
    elif type == 'error':
        chat_box.error(f"**[{entry['time']}] LỖI GỬI** đến `{entry['target']}`: {entry['content']}")
    elif type == 'system_error':
        chat_box.warning(f"**[{entry['time']}] LỖI HỆ THỐNG**: {entry['content']}")

def draw_main_interface():
    """
    Vẽ toàn bộ giao diện chính của ứng dụng Streamlit.
    """
    st.set_page_config(page_title="SAGSIN Simulator", layout="wide")
    st.title("🛰️ Bảng điều khiển Gửi Gói tin Mô phỏng")
    
    # --- Tải dữ liệu ---
    try:
        st.session_state.users = load_from_db("users", User)
        st.session_state.ground_stations = load_from_db("nodes", Node)
    except Exception as e:
        st.error(f"Lỗi kết nối hoặc tải dữ liệu từ DB: {e}")
        st.session_state.users, st.session_state.ground_stations = [], []

    all_users_options = {user.userId: user.userName for user in st.session_state.users}
    
    # --- THANH BÊN (SIDEBAR) ---
    with st.sidebar:
        st.header("Cấu hình Bảng điều khiển")
        st.selectbox("👤 Giao diện này đại diện cho User:", options=list(all_users_options.keys()), format_func=lambda x: all_users_options.get(x, ""), key="active_user_id", help="Chọn User để 'online'. Thay đổi Port sẽ cập nhật cho User này.")
        with st.expander("Cấu hình Listener (Nhận phản hồi)", expanded=True):
            st.number_input("Port Lắng nghe", 1024, 65535, key='listen_port', help="Thay đổi port sẽ khởi động lại Listener và cập nhật lên DB cho user đã chọn.")
            status_color = "green" if "Đang chạy" in st.session_state.get('listener_status', '') else "red"
            st.markdown(f"**Trạng thái Listener:** :{status_color}[{st.session_state.get('listener_status', 'N/A')}]")

    # --- NỘI DUNG CHÍNH ---
    is_send_disabled = not st.session_state.ground_stations or not st.session_state.users
    col1, col2 = st.columns([1, 2])
    with col1:
        # <<< KHÔI PHỤC PHẦN UI BỊ THIẾU >>>
        st.subheader("1. Gửi Gói tin")
        if is_send_disabled: st.warning("Không tìm thấy trạm mặt đất hoặc người dùng.")
        
        active_user_id = st.session_state.get("active_user_id")
        recipient_options = {uid: uname for uid, uname in all_users_options.items() if uid != active_user_id}

        st.selectbox(
            "👤 Gửi đến Người Nhận (Đích):", 
            options=list(recipient_options.keys()) if recipient_options else [], 
            format_func=lambda x: recipient_options.get(x, ""), 
            key="dest_user_id", 
            disabled=is_send_disabled or not recipient_options
        )
        
        # --- Phần code cũ của bạn bắt đầu từ đây ---
        st.subheader("2. Dữ liệu & Cấu hình Gói tin")

        st.selectbox("Loại Dịch vụ", SERVICE_TYPES, key="service_type_select")

        try:
            selected_service_str = st.session_state.service_type_select
            selected_service_enum = ServiceType(selected_service_str)
            qos_profile = get_qos_profile(selected_service_enum)
            
            with st.expander(f"🔍 Xem chi tiết QoS cho **{selected_service_str}**", expanded=True):
                # <<< SỬA LỖI: Dùng tên biến khác cho các cột con >>>
                qos_col1, qos_col2, qos_col3 = st.columns(3)
                qos_col1.metric("Độ trễ tối đa", f"{qos_profile.maxLatencyMs} ms")
                qos_col2.metric("Tỷ lệ mất gói", f"{qos_profile.maxLossRate * 100:.1f}%")
                qos_col3.metric("Độ ưu tiên", qos_profile.defaultPriority)
                
                qos_col4, qos_col5 = st.columns(2)
                qos_col4.metric("Jitter tối đa", f"{qos_profile.maxJitterMs} ms")
                qos_col5.metric("Băng thông tối thiểu", f"{qos_profile.minBandwidthMbps} Mbps")

        except (ValueError, KeyError) as e:
            st.warning("Không tìm thấy cấu hình QoS cho loại dịch vụ đã chọn.")
            logger.error(f"Lỗi khi lấy QoS profile: {e}")

        st.text_area("Nội dung tin nhắn", key="message_input", height=100)
        net_col1, net_col2 = st.columns(2)
        net_col1.slider("TTL", 1, 30, 10, key="ttl_value")
        net_col1.slider("Mức ưu tiên", 1, 5, 1, key="priority_level_value")
        net_col2.checkbox("Sử dụng RL Routing", value=True, key="is_use_rl_toggle")
        st.button("🚀 Gửi Gói tin", on_click=handle_send_button_click, use_container_width=True, type="primary", disabled=is_send_disabled)

    with col2:
        st.subheader("Lịch sử Giao tiếp")
        if st.button("Xóa Lịch sử", use_container_width=True):
            st.session_state.chat_history = []; st.rerun()
        chat_box = st.container(height=500, border=True)
        if not st.session_state.get('chat_history', []):
            chat_box.info("Chưa có gói tin nào được gửi hoặc nhận.")
        else:
            for entry in reversed(st.session_state.chat_history):
                _display_chat_entry(entry, chat_box)

# ==============================================================================
# PHẦN 7: ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ==============================================================================

def main():
    """
    Hàm chính, điều phối toàn bộ ứng dụng Streamlit.
    """
    initialize_session_state()
    draw_main_interface() # Vẽ UI trước để st.selectbox 'active_user_id' có giá trị
    manage_tcp_listener() # Sau đó mới quản lý listener dựa trên user đã chọn
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="auto_refresher")
    check_for_incoming_messages()

if __name__ == "__main__":
    main()