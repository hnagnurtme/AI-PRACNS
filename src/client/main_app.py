# main_app.py
import streamlit as st
import socket
import sys
import random

# --- 1. XỬ LÝ ĐỐI SỐ DÒNG LỆNH VÀ LƯU VÀO SESSION STATE ---

# Logic để lấy cổng lắng nghe từ dòng lệnh (sys.argv[3])
LISTEN_PORT = random.randint(55000, 60000) # Cổng ngẫu nhiên mặc định
try:
    # sys.argv[0]='streamlit', [1]='run', [2]='main_app.py', [3]=<port>
    if len(sys.argv) > 3:
        # Kiểm tra nếu giá trị là số
        if sys.argv[3].isdigit() and 1024 <= int(sys.argv[3]) <= 65535:
            LISTEN_PORT = int(sys.argv[3])
except Exception:
    pass

# Lưu cổng lắng nghe vào Session State để các trang (pages) có thể truy cập
if 'target_listen_port' not in st.session_state:
    st.session_state.target_listen_port = LISTEN_PORT

# --- 2. THIẾT LẬP CẤU HÌNH VÀ GIAO DIỆN TRANG CHỦ ---

st.set_page_config(
    page_title="P2P Mesh Network Simulator",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ P2P Mesh Network Simulator")
st.markdown("---")

st.markdown(f"""
Chào mừng đến với ứng dụng mô phỏng mạng P2P (Peer-to-Peer) trong môi trường Mesh/Vệ tinh.

### ⚙️ Cấu hình Hiện tại:

1.  **Cổng Lắng nghe Của Node Này:** **`{st.session_state.target_listen_port}`** *(Giá trị này được truyền qua dòng lệnh và được sử dụng bởi `1_ChatPage.py`.)*
2.  **IP Cục bộ:** `{socket.gethostbyname(socket.gethostname())}`

### 🚀 Cách sử dụng:

1.  **Giao diện Gửi Gói tin:** Sử dụng menu bên trái để điều hướng đến trang **"1 P2P Chat/Packet Sender"**.
2.  **Kết nối Thật:** Để kiểm tra việc nhận gói tin, bạn cần chạy một ứng dụng Client (ví dụ: một phiên bản khác của `TCP_Sender.py`) và gửi dữ liệu đến IP của bạn tại cổng **`{st.session_state.target_listen_port}`**.
""".format(socket.gethostbyname(socket.gethostname())))

st.info("Sử dụng menu bên trái để điều hướng.")
st.markdown("---")

st.subheader("Mô hình Dữ liệu (Backend)")
st.code("""
from dataclasses import dataclass
class Packet: ...
class Node: ...
""", language="python")