# pages/1_ChatPage.py

import streamlit as st
import base64
from datetime import datetime
import time
import socket
import sys
from typing import List, Dict, Any
from collections import deque
import queue

# ------------------- PHẦN 1: CÀI ĐẶT VÀ IMPORT -------------------
sys.path.insert(0, '.')
try:
    from models.app_models import Packet, ServiceQoS, Communication, Node
    from service.TCP_Listener import TCPListener
    from service.TCP_Sender import send_packet_via_tcp
    from service.incoming_queue import GLOBAL_INCOMING_QUEUE
    from config.mongo_config import get_collection
except ImportError as e:
    st.error(f"Lỗi Import: {e}. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

from streamlit_autorefresh import st_autorefresh

# ------------------- PHẦN 2: CÁC HÀM XỬ LÝ DỮ LIỆU VÀ LOGIC -------------------

@st.cache_data(ttl=60)
def load_peers_from_db(current_node_id: str) -> List[Node]:
    """Tải danh sách các node đang hoạt động từ MongoDB."""
    peers = []
    nodes_collection = get_collection("nodes")
    query = {"status.active": True, "nodeId": {"$ne": current_node_id}}
    projection = {"nodeId": 1, "nodeName": 1, "communication": 1}
    
    for node_doc in nodes_collection.find(query, projection):
        comm_data = node_doc.get("communication", {})
        ip = comm_data.get("ipAddress")
        port = comm_data.get("port")
        if ip and port:
            peers.append(
                Node(
                    nodeId=node_doc.get("nodeId"),
                    nodeName=node_doc.get("nodeName"),
                    communication=Communication(ipAddress=ip, port=int(port))
                )
            )
    return peers

def background_thread_handler(received_data: Dict):
    GLOBAL_INCOMING_QUEUE.put(received_data)

def define_qos(service_type: str) -> ServiceQoS:
    qos_map = {
        "VIDEO_STREAM": ServiceQoS(maxLatencyMs=150.0, maxLossRate=0.01, defaultPriority=1),
        "AUDIO_CALL": ServiceQoS(maxLatencyMs=80.0, maxLossRate=0.005, defaultPriority=2),
        "IMAGE_TRANSFER": ServiceQoS(maxLatencyMs=500.0, maxLossRate=0.02, defaultPriority=3),
    }
    return qos_map.get(service_type, ServiceQoS(maxLatencyMs=1500.0, maxLossRate=0.05, defaultPriority=4))

def handle_send_button_click():
    target_node_id = st.session_state.target_peer
    target_node = next((p for p in st.session_state.connected_peers if p.nodeId == target_node_id), None)
    if not target_node:
        st.error(f"Không tìm thấy thông tin cho Node đích: {target_node_id}")
        return

    message = st.session_state.message_input
    uploaded_file = st.session_state.get('uploaded_file')
    packet_type = st.session_state.packet_type_select
    
    if not message and not uploaded_file and packet_type == "DATA":
        st.error("Vui lòng nhập nội dung hoặc tải tệp lên!")
        return
    
    file_name, payload_base64, payload_size = None, "", 0
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        payload_base64 = base64.b64encode(file_bytes).decode('utf-8')
        payload_size = len(file_bytes)
        file_name = uploaded_file.name
    elif message:
        payload_raw_bytes = message.encode('utf-8')
        payload_base64 = base64.b64encode(payload_raw_bytes).decode('utf-8')
        payload_size = len(payload_raw_bytes)

    new_packet = Packet(
        packetId=f"PKT-{int(time.time() * 1000)}",
        sourceUserId=st.session_state.my_node.nodeId,
        destinationUserId=target_node_id,
        type=packet_type,
        serviceQoS=define_qos(st.session_state.service_type_select),
        payloadDataBase64=payload_base64, payloadFileName=file_name, payloadSizeByte=payload_size,
        TTL=st.session_state.ttl_value, priorityLevel=st.session_state.priority_level_value,
        isUseRL=st.session_state.is_use_rl_toggle
    )
    
    with st.spinner(f"Đang gửi gói tin đến {target_node.nodeName}..."):
        error_message = send_packet_via_tcp(target_node.communication.ipAddress, target_node.communication.port, new_packet)
    
    content_preview = new_packet.payloadFileName or new_packet.get_decoded_payload_preview()
    log_entry: Dict[str, Any] = { "target": f"{target_node.communication.ipAddress}:{target_node.communication.port}", "time": datetime.now().strftime("%H:%M:%S"), "packet_id": new_packet.packetId, "content": content_preview, }
    if error_message is None:
        log_entry.update({"type": "sent", "packet_type": new_packet.type, "is_use_rl": new_packet.isUseRL, "ttl": new_packet.TTL,})
        st.success(f"Gói tin đã gửi thành công! ID: {new_packet.packetId}")
    else:
        log_entry.update({"type": "error", "content": f"Gửi thất bại: {error_message}"})
        st.error(f"Gửi thất bại: {error_message}")
    
    st.session_state.chat_history.append(log_entry)
    st.session_state.message_input = ""

def check_for_incoming_messages():
    while not GLOBAL_INCOMING_QUEUE.empty():
        try:
            received_data = GLOBAL_INCOMING_QUEUE.get_nowait()
            addr, packet_dict = received_data['source_addr'], received_data['packet_data']
            qos_dict = packet_dict.pop('serviceQoS', {})
            received_packet = Packet(**packet_dict, serviceQoS=ServiceQoS(**qos_dict))
            st.session_state.chat_history.append({"type": "received", "target": f"{addr[0]}:{addr[1]}", "content": received_packet.payloadFileName or received_packet.get_decoded_payload_preview(), "packet_type": received_packet.type, "packet_id": received_packet.packetId, "time": datetime.now().strftime("%H:%M:%S")})
        except queue.Empty: break
        except Exception as e:
            st.session_state.chat_history.append({"type": "system_error", "target": "N/A", "content": f"Lỗi xử lý gói tin từ hàng đợi: {e}", "time": datetime.now().strftime("%H:%M:%S")})

# ------------------- PHẦN 3: KHỞI TẠO VÀ CẤU HÌNH -------------------
st.set_page_config(page_title="P2P Packet Sender", layout="wide")

if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# <<< THÊM MỚI: Khởi tạo listener_status nếu chưa có >>>
if 'listener_status' not in st.session_state:
    st.session_state.listener_status = "Chưa khởi động"

# <<< SỬA ĐỔI: Khởi tạo port từ session_state, nếu chưa có thì dùng giá trị mặc định/tham số >>>
DEFAULT_LISTEN_PORT = 50001
if 'listen_port' not in st.session_state:
    # Ưu tiên tham số dòng lệnh, sau đó mới đến mặc định
    try:
        port_from_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else DEFAULT_LISTEN_PORT
        st.session_state.listen_port = port_from_arg
    except IndexError:
        st.session_state.listen_port = DEFAULT_LISTEN_PORT

if 'my_node' not in st.session_state:
    st.session_state.my_node = Node(nodeId="STREAMLIT_CLIENT", nodeName="Streamlit P2P Client", communication=Communication(ipAddress='0.0.0.0', port=st.session_state.listen_port))

try:
    st.toast("Đang làm mới danh sách peer từ database...")
    st.session_state.connected_peers = load_peers_from_db(st.session_state.my_node.nodeId)
except Exception as e:
    st.error(f"Lỗi khi tải danh sách peer từ DB: {e}")
    st.session_state.connected_peers = []

# <<< SỬA ĐỔI: Logic khởi động và khởi động lại listener linh hoạt hơn >>>
listener_needs_restart = False
if 'tcp_listener' not in st.session_state or not st.session_state.tcp_listener.is_alive():
    listener_needs_restart = True
else:
    # Kiểm tra xem listener hiện tại có đang chạy ở cổng đã cấu hình không
    if st.session_state.tcp_listener.port != st.session_state.listen_port:
        st.toast(f"Cổng thay đổi. Đang khởi động lại listener trên port {st.session_state.listen_port}...")
        st.session_state.tcp_listener.stop()
        st.session_state.tcp_listener.join(timeout=2) # Chờ luồng cũ dừng
        listener_needs_restart = True

if listener_needs_restart:
    try:
        listener = TCPListener(
            host=st.session_state.my_node.communication.ipAddress,
            port=st.session_state.listen_port, # Dùng cổng từ session_state
            handler=background_thread_handler
        )
        listener.start()
        st.session_state.tcp_listener = listener
        st.session_state.listener_status = f"Đang chạy (port {st.session_state.listen_port})"
        # Cập nhật lại port của node chính
        st.session_state.my_node.communication.port = st.session_state.listen_port
    except Exception as e:
        st.session_state.listener_status = f"Lỗi khởi tạo trên port {st.session_state.listen_port}: {e}"
        st.error(st.session_state.listener_status)


# ------------------- PHẦN 4: GIAO DIỆN STREAMLIT -------------------
st.title("🛰️ Giao diện Gửi Gói tin P2P")
st.markdown("Gửi và nhận các gói tin trực tiếp với các node khác trong mạng lưới.")

with st.sidebar:
    st.header("Thông tin Node Của Bạn")
    st.markdown(f"**ID Node:** `{st.session_state.my_node.nodeId}`")
    
    # <<< THÊM MỚI: Giao diện cấu hình listener >>>
    with st.expander("Cấu hình Listener", expanded=True):
        st.number_input(
            "Port Lắng nghe",
            min_value=1024,
            max_value=65535,
            key='listen_port', # Trực tiếp cập nhật st.session_state.listen_port
            help="Thay đổi cổng và listener sẽ tự động khởi động lại."
        )
    
    status_color = "green" if "Đang chạy" in st.session_state.get('listener_status', '') else "red"
    st.markdown(f"**Trạng thái Listener:** :{status_color}[{st.session_state.get('listener_status', 'Không xác định')}]")
    st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Cấu hình Gửi tin")
    
    is_send_disabled = not st.session_state.connected_peers
    if is_send_disabled:
        st.warning("Không tìm thấy node nào đang hoạt động để kết nối.")
    
    peer_options = {node.nodeId: f"{node.nodeName} ({node.communication.ipAddress}:{node.communication.port})" for node in st.session_state.connected_peers}
    st.selectbox("Chọn Node Đích", options=list(peer_options.keys()), format_func=lambda x: peer_options.get(x, "N/A"), key="target_peer", disabled=is_send_disabled)
    st.selectbox("Loại Dịch vụ", ["VIDEO_STREAM", "AUDIO_CALL", "IMAGE_TRANSFER", "TEXT_MESSAGE"], key="service_type_select")
    
    st.subheader("2. Dữ liệu Gửi (Payload)")
    st.file_uploader("Tải lên Tệp", key="uploaded_file")
    st.text_area("Hoặc nhập nội dung văn bản", key="message_input", height=100)
    
    st.subheader("3. Cấu hình Gói tin Mạng")
    net_col1, net_col2 = st.columns(2)
    net_col1.selectbox("Loại Gói tin", ["DATA", "ACK"], key="packet_type_select")
    net_col1.slider("TTL", 1, 30, 10, key="ttl_value")
    net_col2.checkbox("Sử dụng RL Routing", value=False, key="is_use_rl_toggle")
    net_col2.slider("Mức ưu tiên", 1, 5, 1, key="priority_level_value")
    
    st.divider()
    st.button("🚀 Gửi Gói tin", on_click=handle_send_button_click, use_container_width=True, type="primary", disabled=is_send_disabled)

with col2:
    st.subheader("Lịch sử Giao tiếp")
    if st.button("Xóa Lịch sử", key="clear_history"):
        st.session_state.chat_history = []
        st.rerun()
    
    chat_box = st.container(height=500, border=True)
    if not st.session_state.chat_history:
        chat_box.info("Chưa có gói tin nào được gửi hoặc nhận.")
    else:
        for entry in reversed(st.session_state.chat_history):
            if entry['type'] == 'sent':
                rl_tag = ":green[RL: ON]" if entry.get('is_use_rl') else ":gray[RL: OFF]"
                chat_box.markdown(f"""<div style='text-align: right; margin-left: 20%; margin-bottom: 8px;'><div style='display: inline-block; text-align: left; border: 1px solid #dcf8c6; padding: 10px; border-radius: 10px; background-color: #dcf8c6;'><span style='color: #556B2F; font-weight: bold;'>GỬI ► [{entry['packet_type']}]</span> <span style='font-size: 0.8em; color: gray;'> {entry['time']} đến {entry['target']}</span><p style='margin: 5px 0 0 0; color: black;'><b>Payload</b>: {entry['content']}</p><span style='font-size: 0.8em; color: teal;'>[ID: {entry['packet_id']}] [TTL: {entry['ttl']}, {rl_tag}]</span></div></div>""", unsafe_allow_html=True)
            elif entry['type'] == 'received':
                chat_box.markdown(f"""<div style='text-align: left; margin-right: 20%; margin-bottom: 8px;'><div style='display: inline-block; text-align: left; border: 1px solid #e0e0e0; padding: 10px; border-radius: 10px; background-color: #f7f7ff;'><span style='color: blue; font-weight: bold;'>◄ NHẬN [{entry['packet_type']}]</span> <span style='font-size: 0.8em; color: gray;'> {entry['time']} từ {entry['target']}</span><p style='margin: 5px 0 0 0; color: black;'><b>Payload</b>: {entry['content']}</p><span style='font-size: 0.8em; color: #6A5ACD;'>[ID: {entry['packet_id']}]</span></div></div>""", unsafe_allow_html=True)
            elif entry['type'] == 'error':
                chat_box.error(f"**[{entry['time']}] LỖI GỬI** đến `{entry['target']}`: {entry['content']}")
            elif entry['type'] == 'system_error':
                chat_box.warning(f"**[{entry['time']}] LỖI HỆ THỐNG**: {entry['content']}")

# ------------------- PHẦN 5: CƠ CHẾ TỰ ĐỘNG CẬP NHẬT -------------------
st_autorefresh(interval=1500, key="auto_refresher")
check_for_incoming_messages()