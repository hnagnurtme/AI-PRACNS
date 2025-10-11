# client_app/streamlit_app.py
import streamlit as st
from engine.shared_ack_listener import start_listening_thread
from data.network_config import get_config
# Removed: from models.packet import Packet (Packet is not directly used here)

CONFIG = get_config()

# --- Khởi tạo và chạy Listener trong Session State ---
# Logic được bọc để chỉ chạy một lần khi session bắt đầu
if 'listener_thread' not in st.session_state:
    
    # Nhận cả server instance và thread instance
    server_instance, thread_instance = start_listening_thread(CONFIG['client_listen_port'])
    
    # Lưu trữ kết quả vào Session State
    st.session_state.listener_server = server_instance
    st.session_state.listener_thread = thread_instance
    
    # Kiểm tra lỗi khởi tạo
    if st.session_state.listener_server is None:
        st.error("FATAL ERROR: Could not start TCP Listener Server on Port 5001. Check for port conflicts.")

# --- Cài đặt UI và Tiêu đề ---
st.set_page_config(layout="wide", page_title="SAGSINs RL Simulator")
st.title("🛰️ SAGSINs RL Simulator Dashboard")

# --- Kiểm tra Trạng thái Listener (Robust Check) ---
listener_thread = st.session_state.listener_thread

# Kiểm tra an toàn: Đảm bảo thread tồn tại và đang hoạt động
if listener_thread and listener_thread.is_alive():
    st.success(f"TCP Listener Server is running on Port {CONFIG['client_listen_port']}. Status: READY")
else:
    # Lỗi xảy ra nếu thread là None hoặc đã chết
    st.error("Lỗi FATAL: Listener failed to start or thread has died. Port 5001 có thể đang được sử dụng.")

st.markdown("""
Chào mừng đến với hệ thống mô phỏng Định tuyến Mạng Tích hợp Không gian-Mặt đất (SAGSINs).

**1. Khởi chạy:** Đảm bảo **Server Java Đích** đang chạy và lắng nghe trên **Port 5000**.
**2. Điều khiển:** Sử dụng menu bên trái để chuyển đến trang **Performance Analysis** và bắt đầu kiểm tra tải.
""")