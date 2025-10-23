# pages/2_VisualizationPage.py

import streamlit as st
import pandas as pd
import sys
import random
import time
import altair as alt

# ------------------- PHẦN 1: CÀI ĐẶT VÀ IMPORT -------------------
sys.path.insert(0, '.')
try:
    from config.mongo_config import get_collection
except ImportError as e:
    st.error(f"Lỗi Import: {e}. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

st.set_page_config(page_title="Routing Performance Analysis", layout="wide")

# ------------------- PHẦN 2: CÁC HÀM MÔ PHỎNG VÀ XỬ LÝ -------------------

@st.cache_data(ttl=60)
def load_nodes_from_db():
    """Tải và cache danh sách các node từ MongoDB."""
    try:
        nodes_collection = get_collection("nodes")
        nodes_list = list(nodes_collection.find({}, {"nodeId": 1, "nodeName": 1, "status": 1, "_id": 0}))
        nodes_df = pd.DataFrame(nodes_list)
        if not nodes_df.empty:
            nodes_df['active'] = nodes_df['status'].apply(lambda s: s.get('active', False) if isinstance(s, dict) else False)
        return nodes_df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu node: {e}")
        return pd.DataFrame()

def _simulate_hop_metric():
    """Hàm phụ: Mô phỏng LinkMetric cho một chặng duy nhất."""
    return {
        "latencyMs": random.uniform(15, 40),
        "availableBandwidthMbps": random.uniform(200, 1500),
        "packetLossRate": random.uniform(0.001, 0.015)
    }

def simulate_routing_analysis(source: str, destination: str, all_nodes: list):
    """Nâng cấp: Tạo dữ liệu giả chi tiết cho từng chặng, bao gồm cả dữ liệu tích lũy."""
    from math import prod
    
    def process_algorithm(path_nodes):
        hop_metrics = [_simulate_hop_metric() for _ in range(len(path_nodes) - 1)]
        
        # Tính toán các chỉ số tích lũy
        cumulative_latency = [0.0]
        cumulative_loss = [0.0]
        current_latency = 0.0
        
        for i, hop in enumerate(hop_metrics):
            current_latency += hop['latencyMs']
            cumulative_latency.append(current_latency)
            # Tính loss tích lũy tại mỗi hop
            current_loss_rate = 1 - prod(1 - h['packetLossRate'] for h in hop_metrics[:i+1])
            cumulative_loss.append(current_loss_rate)

        # Tính toán băng thông thấp nhất tại mỗi chặng
        bottleneck_bw_at_hop = []
        min_bw = float('inf')
        for hop in hop_metrics:
            min_bw = min(min_bw, hop['availableBandwidthMbps'])
            bottleneck_bw_at_hop.append(min_bw)
        
        return {
            "path": path_nodes,
            "cumulative_latency": cumulative_latency,
            "cumulative_loss_rate": cumulative_loss,
            "bottleneck_bandwidth_path": [None] + bottleneck_bw_at_hop
        }

    path_dijkstra = [source] + random.sample([n for n in all_nodes if n not in [source, destination]], k=2) + [destination]
    path_rl = [source] + random.sample([n for n in all_nodes if n not in [source, destination]], k=3) + [destination]
    
    return {"dijkstra": process_algorithm(path_dijkstra), "rl": process_algorithm(path_rl)}

def create_single_algorithm_chart(data: dict, chart_color: str):
    """Tạo biểu đồ kết hợp cho MỘT thuật toán."""
    # Tạo DataFrame từ dữ liệu của một thuật toán
    records = []
    for i, node_id in enumerate(data['path']):
        records.append({
            'Node': node_id,
            'Latency (Cumulative)': data['cumulative_latency'][i],
            'Bandwidth (Bottleneck)': data['bottleneck_bandwidth_path'][i],
            'Packet Loss (Cumulative)': data['cumulative_loss_rate'][i] * 100 # Chuyển sang %
        })
    df = pd.DataFrame(records)

    # Biểu đồ cột cho Băng thông
    bar = alt.Chart(df).mark_bar(color=chart_color, opacity=0.7).encode(
        x=alt.X('Node:N', sort=None, title='Các Node trong Lộ trình'),
        y=alt.Y('Bandwidth (Bottleneck):Q', title='Băng thông (Mbps)', axis=alt.Axis(titleColor=chart_color)),
        tooltip=['Node', 'Bandwidth (Bottleneck)']
    )

    # Biểu đồ đường cho Độ trễ
    line = alt.Chart(df).mark_line(point=True, color=chart_color, strokeWidth=3).encode(
        x=alt.X('Node:N', sort=None),
        y=alt.Y('Latency (Cumulative):Q', title='Độ trễ (ms)', axis=alt.Axis(titleColor=chart_color)),
        tooltip=['Node', 'Latency (Cumulative)']
    )

    # Kết hợp hai biểu đồ
    combined_chart = alt.layer(bar, line).resolve_scale(y='independent')
    return combined_chart

# ------------------- PHẦN 3: GIAO DIỆN STREAMLIT -------------------

st.title("📊 Dashboard Phân tích Hiệu suất Định tuyến")

nodes_df = load_nodes_from_db()
active_nodes = nodes_df[nodes_df['active'] == True]['nodeId'].tolist() if not nodes_df.empty else []

with st.sidebar:
    st.header("1. Cấu hình Yêu cầu")
    if not active_nodes:
        st.warning("Không có node nào hoạt động trong DB.")
        st.stop()
    source_node = st.selectbox("Chọn Node Nguồn", active_nodes, index=0, key="source")
    dest_index = min(1, len(active_nodes) - 1)
    destination_node = st.selectbox("Chọn Node Đích", active_nodes, index=dest_index, key="destination")
    st.header("2. Khởi chạy Phân tích")
    run_button = st.button("PHÂN TÍCH HIỆU SUẤT", use_container_width=True, type="primary")

if run_button:
    if source_node == destination_node:
        st.error("Lỗi: Node Nguồn và Node Đích phải khác nhau.")
    else:
        with st.spinner(f"Đang mô phỏng định tuyến từ {source_node} đến {destination_node}..."):
            st.session_state.analysis_results = simulate_routing_analysis(source_node, destination_node, active_nodes)
            time.sleep(1)
        st.success("Phân tích hoàn tất!")
        st.rerun()

# --- KHU VỰC HIỂN THỊ KẾT QUẢ ---
if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    dijkstra = results['dijkstra']
    rl = results['rl']

    st.subheader(f"So sánh Hiệu suất Lộ trình: `{st.session_state.source}` → `{st.session_state.destination}`")
    
    col1, col2 = st.columns(2)

    # --- Cột 1: Phân tích Dijkstra ---
    with col1:
        with st.container(border=True):
            st.markdown("#### Dijkstra (Tối ưu Độ trễ)")
            
            # Chỉ số tổng
            total_latency = dijkstra['cumulative_latency'][-1]
            bottleneck_bw = min(b for b in dijkstra['bottleneck_bandwidth_path'] if b is not None)
            total_loss = dijkstra['cumulative_loss_rate'][-1]
            
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_col1.metric("Tổng Độ trễ", f"{total_latency:.1f} ms")
            sub_col2.metric("Băng thông", f"{bottleneck_bw:.0f} Mbps")
            sub_col3.metric("Mất gói", f"{total_loss:.3%}")
            
            # Biểu đồ
            dijkstra_chart = create_single_algorithm_chart(dijkstra, '#00BFFF')
            st.altair_chart(dijkstra_chart, use_container_width=True)
            
            # Lộ trình
            st.code(f"Path: {' -> '.join(dijkstra['path'])}")

    # --- Cột 2: Phân tích RL ---
    with col2:
        with st.container(border=True):
            st.markdown("#### Reinforcement Learning (Tối ưu Cân bằng)")

            # Chỉ số tổng và so sánh
            total_latency_rl = rl['cumulative_latency'][-1]
            bottleneck_bw_rl = min(b for b in rl['bottleneck_bandwidth_path'] if b is not None)
            total_loss_rl = rl['cumulative_loss_rate'][-1]

            latency_diff = total_latency_rl - total_latency
            bw_diff = bottleneck_bw_rl - bottleneck_bw
            loss_diff = total_loss_rl - total_loss
            
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_col1.metric("Tổng Độ trễ", f"{total_latency_rl:.1f} ms", f"{latency_diff:+.1f} ms", delta_color="inverse")
            sub_col2.metric("Băng thông", f"{bottleneck_bw_rl:.0f} Mbps", f"{bw_diff:+.0f} Mbps")
            sub_col3.metric("Mất gói", f"{total_loss_rl:.3%}", f"{loss_diff:+.3%}", delta_color="inverse")
            
            # Biểu đồ
            rl_chart = create_single_algorithm_chart(rl, '#2E8B57')
            st.altair_chart(rl_chart, use_container_width=True)
            
            # Lộ trình
            st.code(f"Path: {' -> '.join(rl['path'])}")