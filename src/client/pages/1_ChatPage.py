# 1_ChatPage.py

import streamlit as st
import base64
from datetime import datetime
import time
import socket
import threading
import json
import sys
from typing import Dict
from collections import deque
import queue
# Giả sử bạn có file này để chia sẻ queue
from service.incoming_queue import GLOBAL_INCOMING_QUEUE
from streamlit_autorefresh import st_autorefresh

# --- 1. Imports ---
try:
    sys.path.insert(0, '.')
    from models.app_models import Packet, ServiceQoS, Communication, Node
except ImportError as e:
    st.error(f"Lỗi Import Model: {e}. Đảm bảo app_models.py nằm ở thư mục gốc.")
    st.stop()
try:
    from service.TCP_Listener import TCPListener
    from service.TCP_Sender import send_packet_via_tcp
except ImportError as e:
    st.error(f"Lỗi Import Service: {e}. Vui lòng kiểm tra cấu trúc thư mục service/.")
    st.stop()

# --- 2. Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

DEFAULT_LISTEN_PORT = 50001
if 'listen_ip' not in st.session_state:
    st.session_state.listen_ip = '0.0.0.0'
if 'listen_port' not in st.session_state:
    st.session_state.listen_port = DEFAULT_LISTEN_PORT

# ... (Toàn bộ phần mã định cấu hình cổng và sidebar vẫn giữ nguyên) ...
import os
env_port = int(os.environ.get('LISTEN_PORT')) if os.environ.get('LISTEN_PORT') and os.environ.get('LISTEN_PORT').isdigit() else None
cli_port = next((int(a) for a in sys.argv if a.isdigit() and 1024 <= int(a) <= 65535), None)
query_params = st.query_params
query_port_val = query_params.get('listen_port') or query_params.get('port')
query_port = int(query_port_val) if query_port_val and query_port_val.isdigit() else None
resolved_port = env_port or cli_port or query_port or st.session_state.listen_port
st.session_state.listen_port = resolved_port
with st.sidebar.expander("Listener Configuration", expanded=False):
    st.text_input("Listen IP", key="listen_ip")
    st.text_input("Listen Port", value=str(st.session_state.listen_port), key="listen_port_str")
    st.markdown("---")
    st.caption("Debug: port resolution (env > cli > query > sidebar)")
    st.text(f"Env LISTEN_PORT: {env_port or '<none>'}")
    st.text(f"CLI forwarded port: {cli_port or '<none>'}")
    st.text(f"URL query port: {query_port or '<none>'}")
    st.markdown(f"**Resolved listen port:** {st.session_state.listen_port}")
try:
    port_val = int(st.session_state.listen_port_str)
    if not (1024 <= port_val <= 65535):
        raise ValueError("Port out of range")
    st.session_state.listen_port = port_val
except (ValueError, KeyError):
    pass
if 'my_node' not in st.session_state:
    st.session_state.my_node = Node(nodeId="USER_A_SW", nodeName="Streamlit User A", communication=Communication(ipAddress=st.session_state.listen_ip, port=st.session_state.listen_port))
else:
    st.session_state.my_node.communication.ipAddress = st.session_state.listen_ip
    st.session_state.my_node.communication.port = st.session_state.listen_port
if 'connected_peers' not in st.session_state:
    st.session_state.connected_peers = [
        Node(nodeId="USER_B_SW", nodeName="User B (127.0.0.1:2000)", communication=Communication("localhost", 2000)),
        Node(nodeId="LEO-001", nodeName="Vệ tinh LEO-001 (10.0.0.12:7000)", communication=Communication("localhost", 7000)),
    ]
# ----------------------------------------------------------------------

def background_thread_handler(received_data: Dict):
    GLOBAL_INCOMING_QUEUE.put(received_data)

listener_needs_restart = False
if 'tcp_listener' not in st.session_state or not st.session_state.tcp_listener.is_alive():
    listener_needs_restart = True
else:
    existing_listener = st.session_state.tcp_listener
    if getattr(existing_listener, 'port', None) != st.session_state.listen_port or \
       getattr(existing_listener, 'host', None) != st.session_state.listen_ip:
        try:
            existing_listener.stop()
            existing_listener.join(timeout=2)
        except Exception as e:
            st.warning(f"Lỗi khi dừng listener cũ: {e}")
        listener_needs_restart = True

if listener_needs_restart:
    try:
        listener = TCPListener(
            host=st.session_state.listen_ip,
            port=st.session_state.listen_port,
            handler=background_thread_handler
        )
        listener.start()
        st.session_state.tcp_listener = listener
        st.session_state.listener_status = f"Đang chạy (port {st.session_state.listen_port})"
    except Exception as e:
        st.session_state.listener_status = f"Lỗi khởi tạo: {e}"
        st.error(f"Không thể khởi động Listener trên port {st.session_state.listen_port}. Lỗi: {e}")

# ... (toàn bộ phần mã thêm peer, define_qos, handle_send_button_click giữ nguyên) ...
with st.sidebar:
    st.markdown("---")
    st.header("➕ Quản lý Peer Đích")
    new_peer_id = st.text_input("Peer ID (Ví dụ: USER_C)", key="new_peer_id_input")
    new_peer_name = st.text_input("Tên hiển thị", key="new_peer_name_input")
    new_peer_ip = st.text_input("IP đích", value="127.0.0.1", key="new_peer_ip_input")
    new_peer_port = st.number_input("Port đích", min_value=1024, max_value=65535, value=50003, key="new_peer_port_input")
    def add_new_peer():
        if new_peer_id and new_peer_name and new_peer_ip and new_peer_port:
            try:
                new_comm = Communication(ipAddress=new_peer_ip, port=int(new_peer_port))
                display_name = f"{new_peer_name} ({new_peer_ip}:{new_peer_port})"
                new_node = Node(nodeId=new_peer_id, nodeName=display_name, communication=new_comm)
                if new_peer_id in [n.nodeId for n in st.session_state.connected_peers]:
                    st.error("Peer ID đã tồn tại!")
                    return
                st.session_state.connected_peers.append(new_node)
                st.success(f"Đã thêm Peer '{new_peer_name}' thành công!")
            except Exception as e:
                st.error(f"Lỗi thêm Peer: {e}")
        else:
            st.warning("Vui lòng điền đầy đủ thông tin Peer.")
    st.button("Thêm Peer Mới", on_click=add_new_peer, use_container_width=True)
    if st.session_state.connected_peers:
        st.markdown("---")
        st.subheader("Peers Đã Thêm")
        for node in st.session_state.connected_peers:
            st.markdown(f"- **{node.nodeId}**: `{node.communication.ipAddress}:{node.communication.port}`")
def define_qos(service_type: str) -> ServiceQoS:
    qos_map = {"VIDEO_STREAM": ServiceQoS(maxLatencyMs=150.0, maxLossRate=0.01, defaultPriority=1), "AUDIO_CALL": ServiceQoS(maxLatencyMs=80.0, maxLossRate=0.005, defaultPriority=2), "IMAGE_TRANSFER": ServiceQoS(maxLatencyMs=500.0, maxLossRate=0.02, defaultPriority=3),}
    return qos_map.get(service_type, ServiceQoS(maxLatencyMs=1500.0, maxLossRate=0.05, defaultPriority=4))
def handle_send_button_click():
    target_node_id = st.session_state.target_peer
    message = st.session_state.message_input
    service_type = st.session_state.service_type_select
    uploaded_file = st.session_state.get('uploaded_file')
    packet_type = st.session_state.packet_type_select
    ttl = st.session_state.ttl_value
    is_use_rl = st.session_state.is_use_rl_toggle
    priority_level = st.session_state.priority_level_value
    if not message and not uploaded_file and packet_type == "DATA":
        st.error("Vui lòng nhập nội dung hoặc tải tệp lên để gửi dữ liệu!")
        return
    target_node = next((node for node in st.session_state.connected_peers if node.nodeId == target_node_id), None)
    if not target_node:
        st.error("Không tìm thấy Node đích.")
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
    new_packet = Packet(packetId=f"PKT-{int(time.time() * 1000)}", sourceUserId=st.session_state.my_node.nodeId, destinationUserId=target_node_id, type=packet_type, serviceQoS=define_qos(service_type), payloadDataBase64=payload_base64, payloadFileName=file_name, payloadSizeByte=payload_size, TTL=ttl, priorityLevel=priority_level, isUseRL=is_use_rl)
    with st.spinner(f"Đang gửi gói tin..."):
        error_message = send_packet_via_tcp(target_node.communication.ipAddress, target_node.communication.port, new_packet)
    content_preview = new_packet.payloadFileName or new_packet.get_decoded_payload_preview()
    if error_message is None:
        st.session_state.chat_history.append({"type": "sent", "target": f"{target_node.communication.ipAddress}:{target_node.communication.port}", "content": content_preview, "packet_type": new_packet.type, "is_use_rl": new_packet.isUseRL, "ttl": new_packet.TTL, "packet_id": new_packet.packetId, "time": datetime.now().strftime("%H:%M:%S")})
        st.success(f"Gói tin đã gửi thành công! ID: {new_packet.packetId}")
    else:
        st.session_state.chat_history.append({"type": "error", "target": f"{target_node.communication.ipAddress}:{target_node.communication.port}", "content": f"Gửi thất bại: {error_message}", "time": datetime.now().strftime("%H:%M:%S")})
        st.error(f"Gửi thất bại: {error_message}")
    st.session_state.message_input = ""
# ----------------------------------------------------------------------
## 4. Cập nhật UI từ Listener
# ----------------------------------------------------------------------

def check_for_incoming_messages():
    """Kiểm tra hàng đợi chung và cập nhật UI."""
    while not GLOBAL_INCOMING_QUEUE.empty():
        try:
            received_data = GLOBAL_INCOMING_QUEUE.get_nowait()
            addr = received_data['source_addr']
            packet_dict = received_data['packet_data']

            qos_dict = packet_dict.pop('serviceQoS', {})
            qos_object = ServiceQoS(**qos_dict)

            received_packet = Packet(
                **packet_dict,
                serviceQoS=qos_object
            )
            content_preview = received_packet.payloadFileName or received_packet.get_decoded_payload_preview()

            st.session_state.chat_history.append({
                "type": "received",
                "target": f"{addr[0]}:{addr[1]}",
                "content": content_preview,
                "packet_type": received_packet.type,
                "packet_id": received_packet.packetId,
                "time": datetime.now().strftime("%H:%M:%S")
            })
        except queue.Empty:
            break
        except Exception as e:
            st.session_state.chat_history.append({
                "type": "system_error",
                "target": "N/A",
                "content": f"Lỗi xử lý gói tin từ hàng đợi: {e}",
                "time": datetime.now().strftime("%H:%M:%S")
            })

# ----------------------------------------------------------------------
## 5. GIAO DIỆN STREAMLIT
# ----------------------------------------------------------------------

st.set_page_config(page_title="P2P Chat/Packet Sender", layout="wide")
st.title("🛰️ Giao diện Gửi Gói tin P2P (Mô phỏng Mạng)")
st.markdown("---")

# Sidebar
st.sidebar.header("Thông tin Node Của Bạn")
st.sidebar.markdown(f"**ID Node:** `{st.session_state.my_node.nodeId}`")
st.sidebar.markdown(f"**IP/Port Lắng nghe:** `{st.session_state.my_node.communication.ipAddress}:{st.session_state.my_node.communication.port}`")
status_color = "green" if "Đang chạy" in st.session_state.get('listener_status', '') else "red"
st.sidebar.markdown(f"**Trạng thái Listener:** :{status_color}[{st.session_state.get('listener_status', 'Không xác định')}]")
st.sidebar.markdown("---")

# Main columns
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("1. Cấu hình Cơ bản")
    peer_options = {node.nodeId: node.nodeName for node in st.session_state.connected_peers}
    st.selectbox("Chọn Node Đích (Destination)", options=list(peer_options.keys()), format_func=lambda x: peer_options[x], key="target_peer")
    st.selectbox("Loại Dịch vụ (Service Type)", options=["VIDEO_STREAM", "AUDIO_CALL", "IMAGE_TRANSFER", "TEXT_MESSAGE"], key="service_type_select")
    current_qos = define_qos(st.session_state.service_type_select)
    with st.expander("Chi tiết QoS Yêu cầu"):
        st.markdown(f"- Max Latency: `{current_qos.maxLatencyMs} ms`\n- Max Loss Rate: `{current_qos.maxLossRate}`")
    st.subheader("2. Dữ liệu Gửi (Payload)")
    st.file_uploader("Tải lên Tệp để Gửi", type=None, key="uploaded_file")
    st.text_area("Hoặc: Nhập Nội dung Văn bản/Mã", key="message_input", height=100)
    st.subheader("3. Cấu hình Gói tin Mạng")
    col_net1, col_net2 = st.columns(2)
    with col_net1:
        st.selectbox("Loại Gói tin", options=["DATA", "ACK"], key="packet_type_select")
        st.slider("TTL (Time-To-Live)", min_value=1, max_value=30, value=10, key="ttl_value")
    with col_net2:
        st.checkbox("isUseRL (Routing Học Tăng Cường)", value=False, key="is_use_rl_toggle")
        st.slider("Priority Level", min_value=1, max_value=5, value=1, key="priority_level_value")
    st.markdown("---")
    st.button("🚀 Gửi Gói tin", on_click=handle_send_button_click, use_container_width=True, type="primary")

with col2:
    st.subheader("Lịch sử Giao tiếp (Log)")
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

# --- Auto-refresh mechanism (CORRECTED) ---
# <<< SỬA LỖI: Bỏ đi câu lệnh if/st.rerun() không cần thiết >>>
st_autorefresh(interval=1000, key="auto_refresher")
check_for_incoming_messages()