# client_app/pages/2_🚦_Traffic_Monitor.py (Thay thế cho logic Listener thực tế)
import streamlit as st
import pandas as pd
import time
import random
from ui_components.stats_display import display_incoming_data_table
from data.mock_data import generate_mock_data_traffic # <--- THÊM IMPORT

st.title("🚦 Incoming Traffic Monitor (MOCK DATA)")

# Khởi tạo dữ liệu giả lập trong Session State nếu chưa có
if 'mock_incoming_data' not in st.session_state:
    st.session_state.mock_incoming_data = generate_mock_data_traffic(5) # 5 gói tin ban đầu

# --- Hiển thị Giao diện ---
st.markdown("### Gói DATA và ACK đến Node này")
st.write("Dữ liệu dưới đây được giả lập để kiểm tra UI (Không cần kết nối Socket).")

# Giả lập thêm dữ liệu mới theo thời gian
if st.button("Simulate New Incoming DATA"):
    new_data = generate_mock_data_traffic(random.randint(2, 5))
    st.session_state.mock_incoming_data.extend(new_data)
    
# Hiển thị bảng dữ liệu đến (Sử dụng hàm đã định nghĩa)
display_incoming_data_table(st.session_state.mock_incoming_data) 

# Bỏ Auto-Refresh khi dùng Mock
# st.experimental_rerun() # Không cần thiết vì không cần lắng nghe socket