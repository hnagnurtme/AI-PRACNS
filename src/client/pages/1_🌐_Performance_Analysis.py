import streamlit as st
import pandas as pd
from collections import Counter
import time
from data.network_config import get_config
from engine.rl_theory_calculator import calculate_theory_path
from engine.simulation_runner import run_load_test
from ui_components.stats_display import display_delay_comparison_chart
from ui_components.network_map_viz import draw_network_map # Đã thêm import


CONFIG = get_config()

st.title("🌐 Performance Analysis: Reinforcement Learning vs Theory")

# --- UI Sidebar: Cấu hình Input (Khu vực 1) ---
with st.sidebar:
    st.header("1. Packet Configuration")
    node_ids = [n.id for n in CONFIG['nodes']]
    # Các input này luôn trả về str do cấu hình Node ID
    source = st.selectbox("Source Node", node_ids, index=node_ids.index('G01'))
    destination = st.selectbox("Destination Node", node_ids, index=node_ids.index('G02'))
    service_type = st.selectbox("Service Type (QoS)", ["VIDEO_STREAMING", "FILE_TRANSFER", "BASIC_DATA"])
    payload_size = st.slider("Payload Size (Bytes)", 100, 1500, 1024)
    
    st.header("2. Load Test (Concurrency)")
    request_count = st.number_input("Number of Requests", min_value=1, max_value=50, value=10)
    max_workers = st.number_input("Max Thread Workers", min_value=1, max_value=10, value=5)

    run_button = st.button("RUN LOAD TEST & SEND TCP")
    
# --- Xử lý Logic (Gọi Engine) ---
if run_button:
    # 1. KIỂM TRA INPUT AN TOÀN (Giải quyết lỗi Pylance/NoneType)
    if source is None or destination is None:
        st.error("Lỗi: Vui lòng chọn Node Nguồn và Node Đích hợp lệ.")
        st.stop()
        
    # 2. Kiểm tra Listener
    if not hasattr(st.session_state, 'listener_thread') or not st.session_state.listener_thread.is_alive():
        st.error("Lỗi: Listener Server không hoạt động. Vui lòng kiểm tra lại trang chính.")
    else:
        listener_instance = st.session_state.listener_server
        
        # 3. Tính toán Lý thuyết (Sử dụng source/destination đã được kiểm tra)
        theory_result = calculate_theory_path(source, destination, CONFIG)
        
        sim_config = {
            'source': source, 'destination': destination,
            'dest_ip': CONFIG['dest_server']['ip'], 'dest_port': CONFIG['dest_server']['port'],
            'client_listen_port': CONFIG['client_listen_port'],
            'theory_delay': theory_result['delay_ms'],
            'theory_path': theory_result['path'],
            'service_type': service_type, 'payload_size': payload_size
        }
        
        with st.spinner(f"Running {request_count} concurrent simulations..."):
            # 4. Chạy đa luồng và chờ kết quả
            all_results = run_load_test(request_count, max_workers, sim_config, listener_instance)
            
            # Lưu kết quả
            st.session_state['results_df'] = pd.DataFrame(all_results)
            st.session_state['theory_path'] = theory_result['path']
            
        st.success("Load test completed! Analysis below.")
        time.sleep(1) 
        st.rerun() # Đã sửa: Sử dụng st.rerun()

# --- Khu vực 2: Hiển thị Kết quả và Biểu đồ ---
if 'results_df' in st.session_state:
    df = st.session_state['results_df']
    theory_delay = df['theory_delay_ms'].iloc[0] if not df.empty else 0
    
    col_map, col_stats = st.columns([1, 1])

    # TÍNH ĐƯỜNG ĐI PHỔ BIẾN NHẤT
    rl_paths = [tuple(p) for p in df[df['status'] == 'SUCCESS']['path']]
    if rl_paths:
        most_common_path = list(Counter(rl_paths).most_common(1)[0][0])
    else:
        # Sử dụng đường đi lý thuyết nếu không có gói nào thành công
        most_common_path = st.session_state.get('theory_path', [])
        
    
    with col_map:
        st.subheader("Network Map & Optimal RL Path")
        # GỌI HÀM VẼ BIỂU ĐỒ MẠNG
        draw_network_map(
            config=CONFIG,
            rl_path_history=most_common_path,
            theory_path=st.session_state.get('theory_path', [])
        )

        st.info(f"Theory Path ({theory_delay:.2f}ms): {' -> '.join(st.session_state.get('theory_path', []))}")
        st.success(f"Most Frequent RL Path: {' -> '.join(most_common_path)}")
        st.write(f"RL Success Rate: {len(df[df['status'] == 'SUCCESS'])}/{len(df)}")


    with col_stats:
        # GỌI HÀM VẼ BIỂU ĐỒ SO SÁNH ĐỘ TRỄ
        display_delay_comparison_chart(df, theory_delay)