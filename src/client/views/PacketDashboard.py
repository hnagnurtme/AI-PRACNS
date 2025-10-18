import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Import thêm graph_objects
import streamlit.components.v1 as components
import sys

# --- 1. IMPORT VÀ CÀI ĐẶT ---
sys.path.insert(0, '.')
try:
    from config.mongo_config import get_collection
except ImportError as e:
    st.error(f"Lỗi Import: {e}. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=60)
def load_app_data():
    """Tải dữ liệu nodes từ DB và tạo dữ liệu packets mẫu đã được đồng bộ hóa."""
    try:
        nodes_collection = get_collection("nodes")
        nodes = list(nodes_collection.find({}, {'_id': 0}))
        
        if not nodes:
            st.warning("Không tìm thấy dữ liệu node trong database.")
            return [], []

        packets = [
            {"packetId": "PKT-VIDEO-CALL-001", "serviceType": "VIDEO_CALL", "accumulatedDelayMs": 81.1, "pathHistory": ["GS-01", "LEO-001", "MEO-001", "LEO-002", "GS-02"], "hopRecords": [{"fromNodeId": "GS-01", "toNodeId": "LEO-001", "latencyMs": 15.2, "linkType": "RF", "bandwidthUsedMbps": 8.5}, {"fromNodeId": "LEO-001", "toNodeId": "MEO-001", "latencyMs": 25.0, "linkType": "LASER", "bandwidthUsedMbps": 8.5}, {"fromNodeId": "MEO-001", "toNodeId": "LEO-002", "latencyMs": 24.8, "linkType": "LASER", "bandwidthUsedMbps": 8.5}, {"fromNodeId": "LEO-002", "toNodeId": "GS-02", "latencyMs": 16.1, "linkType": "RF", "bandwidthUsedMbps": 8.5}]},
            {"packetId": "PKT-IOT-DATA-002", "serviceType": "IOT_DATA", "accumulatedDelayMs": 20.4, "pathHistory": ["GS-01", "LEO-001", "LEO-002"], "hopRecords": [{"fromNodeId": "GS-01", "toNodeId": "LEO-001", "latencyMs": 14.9, "linkType": "RF", "bandwidthUsedMbps": 0.2}, {"fromNodeId": "LEO-001", "toNodeId": "LEO-002", "latencyMs": 5.5, "linkType": "LASER", "bandwidthUsedMbps": 0.2}]}
        ]
        
        return nodes, packets
    except Exception as e:
        st.error(f"Lỗi kết nối hoặc truy vấn database: {e}")
        return [], []

# --- 3. CÁC HÀM VẼ BIỂU ĐỒ MỚI ---

def create_resource_usage_chart(packet, nodes_data):
    """Tạo biểu đồ cột hiển thị tỷ lệ tài nguyên băng thông gói tin sử dụng trên mỗi node."""
    hop_records = packet.get('hopRecords', [])
    if not hop_records: return go.Figure().update_layout(title_text="Không có dữ liệu chặng")

    usage_data = []
    # Bắt đầu từ node nguồn
    path = [hop_records[0]['fromNodeId']] + [hop['toNodeId'] for hop in hop_records]

    for node_id in path:
        node_info = next((n for n in nodes_data if n.get('nodeId') == node_id), None)
        if node_info:
            total_bandwidth = node_info.get('communication', {}).get('bandwidthMHz', 0)
            
            # Tìm tất cả các hop đi ra từ node này để tính tổng băng thông đã dùng
            used_bw = sum(h.get('bandwidthUsedMbps', 0) for h in hop_records if h.get('fromNodeId') == node_id)
            
            percentage = (used_bw / total_bandwidth) * 100 if total_bandwidth > 0 else 0
            usage_data.append({
                "Node ID": node_id,
                "Percentage": percentage,
                "Usage Info": f"{used_bw:.1f} / {total_bandwidth:.1f} Mbps"
            })
            
    if not usage_data: return go.Figure().update_layout(title_text="Thiếu thông tin băng thông")

    df = pd.DataFrame(usage_data)
    fig = px.bar(df, x="Node ID", y="Percentage", title="Tỷ lệ Băng thông Gói tin sử dụng trên mỗi Node", labels={"Percentage": "Tỷ lệ sử dụng (%)", "Node ID": "Node trên Đường đi"}, text="Usage Info", color="Percentage", color_continuous_scale=px.colors.sequential.YlOrRd)
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis_range=[0, max(df['Percentage']) * 1.25])
    return fig

# Thay thế hàm cũ bằng hàm này
def create_combined_latency_chart(packet):
    """Tạo biểu đồ kết hợp cột (độ trễ từng chặng) và đường (độ trễ tích lũy)."""
    if not packet.get('hopRecords'): return go.Figure()
    
    df = pd.DataFrame(packet['hopRecords'])
    df['hopLabel'] = df['fromNodeId'] + ' → ' + df['toNodeId']
    # Tính độ trễ tích lũy
    df['cumulativeLatency'] = df['latencyMs'].cumsum()
    
    fig = go.Figure()

    # Thêm biểu đồ CỘT cho độ trễ từng chặng
    fig.add_trace(go.Bar(
        x=df['hopLabel'],
        y=df['latencyMs'],
        name='Độ trễ từng chặng',
        marker_color='royalblue',
        text=df['latencyMs'],
        texttemplate='%{text:.1f}',
        textposition='auto'
    ))

    # Thêm biểu đồ ĐƯỜNG cho độ trễ tích lũy
    fig.add_trace(go.Scatter(
        x=df['hopLabel'],
        y=df['cumulativeLatency'],
        name='Độ trễ tích lũy',
        mode='lines+markers',
        line=dict(color='firebrick', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))

    # === SỬA LỖI TẠI ĐÂY ===
    fig.update_layout(
        title_text='Phân tích Độ trễ Từng chặng và Tích lũy',
        xaxis_title='Các chặng trên đường đi',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        
        # Cấu trúc ĐÚNG cho trục y1 (bên trái)
        yaxis=dict(
            title=dict(
                text='Độ trễ từng chặng (ms)',
                font=dict(color='royalblue')  # 'titlefont' -> 'font' bên trong 'title'
            ),
            tickfont=dict(color='royalblue')
        ),
        
        # Cấu trúc ĐÚNG cho trục y2 (bên phải)
        yaxis2=dict(
            title=dict(
                text='Tổng độ trễ tích lũy (ms)',
                font=dict(color='firebrick') # 'titlefont' -> 'font' bên trong 'title'
            ),
            tickfont=dict(color='firebrick'),
            overlaying='y',
            side='right'
        )
    )
    
    return fig

# --- 4. GIAO DIỆN STREAMLIT CHÍNH ---

st.set_page_config(layout="wide", page_title="Packet Performance Analysis")
st.title("🔬 Bảng điều khiển Phân tích Hiệu năng Gói tin")

all_nodes, all_packets = load_app_data()

if all_nodes and all_packets:
    st.sidebar.header("Bảng điều khiển")
    packet_ids = [p.get('packetId') for p in all_packets]
    selected_packet_id = st.sidebar.selectbox("Chọn Gói tin để Phân tích", packet_ids)
    selected_packet = next((p for p in all_packets if p.get('packetId') == selected_packet_id), None)

    if selected_packet:
        st.header(f"Phân tích chi tiết Packet: `{selected_packet['packetId']}`")
        st.markdown("---")

        # --- Các chỉ số KPI ---
        total_latency = selected_packet.get('accumulatedDelayMs', 0)
        num_hops = len(selected_packet.get('hopRecords', []))
        service_type = selected_packet.get('serviceType', 'N/A')

        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng Độ trễ", f"{total_latency:.1f} ms")
        col2.metric("Số chặng (Hops)", num_hops)
        col3.metric("Loại Dịch vụ", service_type)
        
        st.markdown("---")

        # --- Hàng Biểu đồ ---
        st.header("Trực quan hóa Hiệu năng")
        
        # Biểu đồ tài nguyên
        st.subheader("Phân tích Sử dụng Tài nguyên")
        fig_resource = create_resource_usage_chart(selected_packet, all_nodes)
        st.plotly_chart(fig_resource, use_container_width=True)
        
        # Biểu đồ độ trễ kết hợp
        st.subheader("Phân tích Nút thắt Cổ chai và Độ trễ Tích lũy")
        fig_latency_combined = create_combined_latency_chart(selected_packet)
        st.plotly_chart(fig_latency_combined, use_container_width=True)

        # Expander để xem dữ liệu thô
        with st.expander("📄 Xem dữ liệu chi tiết"):
            st.subheader("Dữ liệu các Chặng (Hop Records)")
            st.dataframe(pd.DataFrame(selected_packet.get('hopRecords', [])), use_container_width=True)
            st.subheader("JSON đầy đủ của Gói tin")
            st.json(selected_packet)
else:
    st.info("Đang chờ dữ liệu node từ database hoặc database trống...")