import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
import time

# --- 1. DỮ LIỆU CẤU HÌNH VÀ MÔ PHỎNG ---

# Định nghĩa công suất băng thông tối đa (Mbps) cho mỗi loại node
NODE_MAX_BANDWIDTH = {
    "USER": 1000, "DEST": 1000, "UAV": 50, "LEO": 200, "MEO": 500, "GEO": 1000
}

# Dữ liệu so sánh 2 gói tin/thuật toán
routing_comparison_data = {
    "RL_Packet": {
        "path": ["USER_A", "UAV_1", "LEO_1", "MEO_1", "GEO_1", "GEO_2", "MEO_5", "LEO_9", "DEST_ROUTER_1"],
        "hops": [{"latencyMs": 5.2}, {"latencyMs": 10.8}, {"latencyMs": 25.5}, {"latencyMs": 60.1}, {"latencyMs": 1.5}, {"latencyMs": 62.3}, {"latencyMs": 24.9}, {"latencyMs": 11.2}],
        "color": "#00a8e8", # Xanh dương (RL)
        "bandwidthMbps": 5.0
    },
    "Dijkstra_Packet": {
        "path": ["USER_A", "UAV_3", "LEO_3", "MEO_2", "MEO_3", "MEO_6", "LEO_10", "UAV_12", "DEST_ROUTER_1"],
        "hops": [{"latencyMs": 4.1}, {"latencyMs": 9.2}, {"latencyMs": 28.1}, {"latencyMs": 14.5}, {"latencyMs": 15.1}, {"latencyMs": 26.8}, {"latencyMs": 10.1}, {"latencyMs": 5.5}],
        "color": "#f72585", # Hồng (Dijkstra)
        "bandwidthMbps": 5.0
    }
}

# Dữ liệu định nghĩa toàn bộ cấu trúc mạng nền
full_network = {
    "nodes": {"USER": ["USER_A"], "DEST": ["DEST_ROUTER_1"], "UAV": [f"UAV_{i}" for i in range(1, 13)], "LEO": [f"LEO_{i}" for i in range(1, 13)], "MEO": [f"MEO_{i}" for i in range(1, 7)], "GEO": ["GEO_1", "GEO_2"]},
    "links": [("GEO_1", "GEO_2"), ("MEO_1","MEO_3"), ("MEO_2","MEO_3"), ("MEO_5","MEO_6")]
}


# --- 2. CÁC HÀM TIỆN ÍCH ---

def create_manual_layout(topology_nodes, scale_x=350, scale_y=150):
    """Tính toán và trả về tọa độ (x, y) cố định cho tất cả các node."""
    positions = {}
    layer_y_coords = {"USER": 4, "DEST": 4, "UAV": 3, "LEO": 2, "MEO": 1, "GEO": 0}
    user_nodes = topology_nodes.get("USER", [])
    dest_nodes = topology_nodes.get("DEST", [])
    all_nodes_flat = [node for sublist in topology_nodes.values() for node in sublist]

    for node_id in all_nodes_flat:
        node_type = node_id.split('_')[0]
        node_suffix_str = node_id.split('_')[-1]
        y = layer_y_coords.get(node_type, 0) * scale_y
        
        if node_type == "USER":
            idx = user_nodes.index(node_id)
            x = -4 * scale_x + idx * scale_x
        elif node_type == "DEST":
            idx = dest_nodes.index(node_id)
            x = 2 * scale_x + idx * scale_x
        elif node_id == "GEO_1":
            x = -1 * scale_x
        elif node_id == "GEO_2":
            x = 1 * scale_x
        elif node_type == "MEO":
            node_num = int(node_suffix_str)
            if node_num <= 3:
                x = -2 * scale_x + (node_num - 1) * scale_x
            else:
                x = 0 * scale_x + (node_num - 4) * scale_x
        else: # UAV và LEO
            node_num = int(node_suffix_str)
            if node_num <= 6:
                x = -3.5 * scale_x + (node_num - 1) * (scale_x / 1.5)
            else:
                x = 0.5 * scale_x + (node_num - 7) * (scale_x / 1.5)
        positions[node_id] = {'x': x, 'y': y}
    return positions

def create_network_visualization(current_time_step, positions):
    """Tạo đối tượng đồ thị Pyvis dựa trên bước thời gian hiện tại."""
    net = Network(height='700px', width='950px', directed=True, notebook=True, cdn_resources='in_line')
    layer_properties = {
        "USER": {"color": "#e07a5f", "shape": "dot", "size": 15}, "DEST": {"color": "#e07a5f", "shape": "dot", "size": 15},
        "UAV": {"color": "#3d405b", "shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf0fb", "size": 30}},
        "LEO": {"color": "#81b29a", "shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf7c2", "size": 30}},
        "MEO": {"color": "#f2cc8f", "shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf7c2", "size": 35}},
        "GEO": {"color": "#5e2a20", "shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf7c2", "size": 40}},
    }

    all_nodes_list = [node for sublist in full_network["nodes"].values() for node in sublist]
    node_states = {node_id: {"packets_arriving": [], "total_bw_required": 0.0} for node_id in all_nodes_list}

    # Tính toán trạng thái động của các node
    for pkt_id, pkt_data in routing_comparison_data.items():
        if current_time_step < len(pkt_data["hops"]):
            next_node = pkt_data["path"][current_time_step + 1]
            node_states[next_node]["packets_arriving"].append(pkt_id)
            node_states[next_node]["total_bw_required"] += pkt_data["bandwidthMbps"]

    # Thêm các node vào đồ thị
    for node_id in all_nodes_list:
        node_type = node_id.split('_')[0]
        props = layer_properties.get(node_type, {}).copy()
        pos = positions[node_id]

        max_bw = NODE_MAX_BANDWIDTH.get(node_type, 1)
        current_bw = node_states[node_id]["total_bw_required"]
        utilization = current_bw / max_bw if max_bw > 0 else 0

        border_color = '#666666'
        if utilization >= 0.75: border_color = 'crimson'  # Tải cao
        elif utilization > 0: border_color = 'orange'     # Đang bận

        props['color'] = {'background': props.get('color', 'grey'), 'border': border_color}
        net.add_node(node_id, label=node_id, x=pos['x'], y=pos['y'], physics=False, borderWidth=4 if utilization > 0 else 1.5, **props)

    # Thêm các liên kết nền
    for u, v in full_network["links"]:
        net.add_edge(u, v, color='lightgray', dashes=True, width=1)

    # Thêm các liên kết của gói tin
    for pkt_id, pkt_data in routing_comparison_data.items():
        path, path_color = pkt_data["path"], pkt_data["color"]
        # Vẽ đường đã đi qua
        edges_traveled = list(zip(path, path[1:]))[:current_time_step]
        for u, v in edges_traveled:
            net.add_edge(u, v, color=path_color, dashes=False, width=2, alpha=0.8)
        # Vẽ chặng đang hoạt động
        if current_time_step < len(pkt_data["hops"]):
            u, v = path[current_time_step], path[current_time_step + 1]
            latency = pkt_data["hops"][current_time_step]["latencyMs"]
            net.add_edge(u, v, color='springgreen', dashes=False, width=5, title=f"{pkt_id} Latency: {latency} ms")

    net.set_options(""" const options = { "physics": { "enabled": false } } """)
    return net, node_states


# --- 3. GIAO DIỆN STREAMLIT ---
st.set_page_config(layout="wide")

# Khởi tạo session state
if 'time_step' not in st.session_state: st.session_state.time_step = 0
if 'running' not in st.session_state: st.session_state.running = False
if 'positions' not in st.session_state:
    st.session_state.positions = create_manual_layout(full_network["nodes"])
max_animation_steps = max(len(p["hops"]) for p in routing_comparison_data.values())

# Bảng điều khiển
col1, col2, col3 = st.columns([1, 1, 6])
if col1.button("▶️ Run/Pause", use_container_width=True): st.session_state.running = not st.session_state.running
if col2.button("🔄 Reset", use_container_width=True):
    st.session_state.running = False
    st.session_state.time_step = 0

st.markdown("---")

# Bố cục chính
main_cols = st.columns([2, 1])
graph_placeholder = main_cols[0].empty()
info_placeholder = main_cols[1].empty()

def update_display(current_step):
    """Cập nhật toàn bộ giao diện dựa trên bước thời gian."""
    net_viz, node_states = create_network_visualization(current_step, st.session_state.positions)

    with graph_placeholder:
        components.html(net_viz.generate_html(), height=710, width=960)

    # Cột thông tin bên phải với khung cuộn
    with info_placeholder:
        with st.container(height=710, border=False):
            st.markdown(f"#### 📊 Phân tích tại Bước {current_step}")
            st.markdown("---")

            display_order = ["Dijkstra_Packet", "RL_Packet"]

            for pkt_id in display_order:
                pkt_data = routing_comparison_data[pkt_id]
                path, color = pkt_data["path"], pkt_data["color"]

                with st.container(border=True):
                    st.markdown(f"<h5 style='color:{color};'> Gói tin: {pkt_id} </h5>", unsafe_allow_html=True)

                    if current_step < len(pkt_data["hops"]):
                        next_node = path[current_step + 1]
                        total_latency = sum(h["latencyMs"] for h in pkt_data["hops"][:current_step + 1])

                        st.metric("Tổng Độ trễ Tích lũy", f"{total_latency:.1f} ms")
                        st.markdown(f"**Node Tiếp theo:** `{next_node}`")

                        # Chỉ hiển thị thông tin chi tiết cho gói tin RL
                        if pkt_id == "RL_Packet":
                            next_node_state = node_states[next_node]
                            node_type = next_node.split('_')[0]
                            max_bw = NODE_MAX_BANDWIDTH.get(node_type, 1)
                            used_bw = next_node_state["total_bw_required"]
                            available_bw = max_bw - used_bw

                            st.markdown("**Trạng thái Node Tiếp theo:**")
                            c1, c2 = st.columns(2)
                            c1.metric("Gói tin đang xử lý", f"{len(next_node_state['packets_arriving'])}")
                            c2.metric("Tải Hiện tại", f"{used_bw:.1f}/{max_bw} Mbps")
                            st.progress(used_bw / max_bw if max_bw > 0 else 0)
                            st.metric("Băng thông Khả dụng", f"{available_bw:.1f} Mbps", delta=f"{-used_bw:.1f} Mbps", delta_color="inverse")
                        else:
                            st.markdown("<br>", unsafe_allow_html=True) # Thêm khoảng trống để căn chỉnh

                            st.markdown("<br>", unsafe_allow_html=True)
                    else:
                        st.success("✅ Đã đến đích!")
                        total_latency = sum(h["latencyMs"] for h in pkt_data["hops"])
                        st.metric("Tổng Độ trễ Toàn trình", f"{total_latency:.1f} ms")
                        st.markdown("<br><br><br>", unsafe_allow_html=True) # Căn chỉnh chiều cao khi hoàn thành

# Vòng lặp hoạt ảnh
while st.session_state.running:
    current_step = st.session_state.time_step
    if current_step >= max_animation_steps:
        st.session_state.running = False
        st.balloons()
        break
    update_display(current_step)
    st.session_state.time_step += 1
    time.sleep(1.5)
    st.rerun()

# Hiển thị tĩnh khi không chạy
if not st.session_state.running:
    update_display(st.session_state.time_step - 1 if st.session_state.time_step > 0 else 0)