# pages/2_VisualizationPage.py

import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import sys

# ------------------- PHẦN 1: CÀI ĐẶT VÀ IMPORT -------------------
sys.path.insert(0, '.')
try:
    from config.mongo_config import get_collection
    from service.LinkMetric_Calculator import calculate_single_link_metric, update_all_link_metrics_in_db
except ImportError as e:
    st.error(f"Lỗi Import: {e}. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

st.set_page_config(page_title="Network Visualization", layout="wide")

# ------------------- PHẦN 2: CÁC HÀM XỬ LÝ -------------------

@st.cache_data(ttl=10)
def load_network_data():
    """Tải đồng thời dữ liệu về Nodes và Links từ MongoDB."""
    try:
        nodes_collection = get_collection("nodes")
        links_collection = get_collection("link_metrics")
        nodes_list = list(nodes_collection.find({}))
        links_list = list(links_collection.find({"isLinkActive": True}))
        nodes_df = pd.DataFrame(nodes_list)
        links_df = pd.DataFrame(links_list)
        if not nodes_df.empty:
            nodes_df['active'] = nodes_df['status'].apply(lambda s: s.get('active', False) if isinstance(s, dict) else False)
        return nodes_df, links_df
    except Exception as e:
        st.error(f"Lỗi kết nối hoặc lấy dữ liệu từ MongoDB: {e}")
        return pd.DataFrame(), pd.DataFrame()

def generate_network_graph(nodes_df: pd.DataFrame, links_df: pd.DataFrame) -> str:
    """Tạo đồ thị mạng tương tác và trả về chuỗi HTML."""
    net = Network(height="750px", width="100%", bgcolor="#0E1117", font_color=True, notebook=True, cdn_resources='in_line')
    net.set_options("""
    var options = { "physics": { "enabled": true, "barnesHut": { "gravitationalConstant": -20000, "centralGravity": 0.15, "springLength": 200 }, "solver": "barnesHut" } }
    """)
    
    node_styles = {
        "GROUND_STATION": {"shape": "triangle", "color": "#00BFFF", "size": 30},
        "LEO_SATELLITE": {"shape": "star", "color": "#FFD700", "size": 25},
        "MEO_SATELLITE": {"shape": "ellipse", "color": "#90EE90", "size": 35},
        "GEO_SATELLITE": {"shape": "square", "color": "#FF6347", "size": 40},
        "INACTIVE": {"color": "#4A4A4A"}
    }

    # **Vẽ các NODE (Đỉnh)**
    active_node_ids = set(nodes_df[nodes_df['active'] == True]['nodeId'])
    for _, node in nodes_df.iterrows():
        node_id, is_active, node_type = node["nodeId"], node["active"], node["type"]
        
        # --- SỬA LỖI TẠI ĐÂY ---
        # 1. Lấy style gốc. Dùng .copy() để không sửa đổi dict gốc.
        style = node_styles.get(node_type, {"shape": "dot", "color": "grey", "size": 20}).copy()
        
        # 2. Ghi đè màu nếu node không hoạt động
        if not is_active:
            style['color'] = node_styles["INACTIVE"]["color"]

        title = f"""
        <b>ID:</b> {node_id}<br>
        <b>Tên:</b> {node['nodeName']}<br>
        <b>Loại:</b> {node_type}
        """
        
        # 3. Giờ đây, chỉ cần unpack dict 'style' đã được cập nhật hoàn chỉnh
        net.add_node(node_id, label=node["nodeName"], title=title, **style)
        # --- KẾT THÚC SỬA LỖI ---

    # **Vẽ các LINK (Cạnh)**
    for _, link in links_df.iterrows():
        src, dst = link["sourceNodeId"], link["destinationNodeId"]
        if src not in active_node_ids or dst not in active_node_ids:
            continue
        score = link.get("linkScore", 0)
        if score > 80: color, width = "#00FF00", 3.5
        elif score > 60: color, width = "#ADFF2F", 2.5
        elif score > 40: color, width = "#FFA500", 1.5
        else: color, width = "#FF0000", 1
        link_title = f"""<b>Link:</b> {src} ↔ {dst}<hr><b>Điểm chất lượng:</b> {score:.1f}<br><b>Độ trễ:</b> {link.get('latencyMs', 0):.2f} ms<br><b>Băng thông:</b> {link.get('currentAvailableBandwidthMbps', 0):.0f} Mbps"""
        net.add_edge(src, dst, title=link_title, color=color, width=width)

    try:
        file_path = "network_graph_live.html"
        net.save_graph(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<p>Lỗi khi tạo đồ thị: {e}</p>"

# ------------------- PHẦN 3: GIAO DIỆN CHÍNH STREAMLIT -------------------
st_autorefresh(interval=10000, key="visual_refresher")

st.title("🛰️ Dashboard Trực quan hóa Mạng lưới")
st.markdown("Đồ thị biểu diễn các **Node** và chất lượng kết nối (**LinkMetric**) theo thời gian thực.")

with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    if st.button("Cập nhật Toàn bộ Link Metrics vào DB", use_container_width=True):
        with st.spinner("Đang tính toán lại tất cả các liên kết mạng..."):
            update_all_link_metrics_in_db()
            st.cache_data.clear()
        st.success("Đã cập nhật thành công dữ liệu Link Metrics vào DB!")
        st.rerun()
    st.divider()
    st.header("🔬 Kiểm tra Liên kết Trực tiếp")
    nodes_df, links_df = load_network_data()
    active_nodes_list = sorted(nodes_df[nodes_df['active'] == True]['nodeId'].tolist()) if not nodes_df.empty else []
    if not active_nodes_list:
        st.warning("Không có node nào đang hoạt động.")
    else:
        source_id = st.selectbox("Chọn Node Nguồn", options=active_nodes_list)
        dest_id = st.selectbox("Chọn Node Đích", options=active_nodes_list, index=min(1, len(active_nodes_list)-1))
        if st.button("Kiểm tra Chất lượng", use_container_width=True, type="primary"):
            if source_id == dest_id:
                st.warning("Vui lòng chọn hai node khác nhau.")
            elif source_id is None or dest_id is None:
                st.warning("Vui lòng chọn cả hai node.")
            else:
                result = calculate_single_link_metric(source_id, dest_id)
                st.session_state.link_check_result = result
    if 'link_check_result' in st.session_state and st.session_state.link_check_result:
        result = st.session_state.link_check_result
        st.divider()
        st.subheader(f"Kết quả: {result['sourceNodeId']} ↔ {result['destinationNodeId']}")
        if not result.get("isLinkActive", True):
            st.error(f"❌ Liên kết không hoạt động: {result.get('reason')}")
        else:
            score = result.get('linkScore', 0)
            st.metric("Điểm chất lượng", f"{score:.1f} / 100", f"{'Tốt' if score > 60 else 'Yếu'}", delta_color=("normal" if score > 60 else "inverse"))
            col1, col2 = st.columns(2)
            col1.metric("Độ trễ", f"{result.get('latencyMs', 0):.1f} ms")
            col2.metric("Khoảng cách", f"{result.get('distanceKm', 0):.0f} km")

if nodes_df.empty:
    st.warning("Không tìm thấy dữ liệu node. Vui lòng chạy script `init_Node.py`.")
else:
    active_count = int(nodes_df['active'].sum())
    st.info(f"Đang hiển thị **{len(nodes_df)}** node ({active_count} hoạt động) và **{len(links_df)}** liên kết.")
    if links_df.empty:
        st.warning("Không có dữ liệu LinkMetric. Hãy nhấn nút 'Cập nhật Toàn bộ...' ở sidebar.")
    graph_html = generate_network_graph(nodes_df, links_df)
    with st.container(border=True):
        components.html(graph_html, height=800, scrolling=True)