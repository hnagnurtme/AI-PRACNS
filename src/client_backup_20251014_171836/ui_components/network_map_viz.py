# client_app/ui_components/network_map_viz.py
import plotly.graph_objects as go
from typing import Dict, List, Any
import streamlit as st

def draw_network_map(config: Dict, rl_path_history: List[str], theory_path: List[str]):
    """Vẽ bản đồ mạng SAGSINs và làm nổi bật đường đi RL."""
    
    nodes_data = config['nodes']
    links_data = config['links']

    # Lấy tọa độ
    node_coords = {n.id: (n.pos[0], n.pos[1]) for n in nodes_data}

    # --- 1. Xử lý Edges (Liên kết) ---
    edge_x = []
    edge_y = []
    
    for link in links_data:
        x0, y0 = node_coords[link.source]
        x1, y1 = node_coords[link.target]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # --- 2. Xử lý Nodes (Điểm) ---
    node_x = [coords[0] for coords in node_coords.values()]
    node_y = [coords[1] for coords in node_coords.values()]
    
    node_labels = [n.label for n in nodes_data]
    
    # --- 3. Highlight Đường đi RL Phổ biến nhất ---
    # Tạo các scatter cho đường đi RL (dùng đường nét đứt)
    rl_path_edges_x = []
    rl_path_edges_y = []
    
    for i in range(len(rl_path_history) - 1):
        n1 = rl_path_history[i]
        n2 = rl_path_history[i+1]
        
        if n1 in node_coords and n2 in node_coords:
            x0, y0 = node_coords[n1]
            x1, y1 = node_coords[n2]
            rl_path_edges_x.extend([x0, x1, None])
            rl_path_edges_y.extend([y0, y1, None])


    # --- 4. Tạo Plotly Figure ---
    fig = go.Figure(data=[
        # Lớp 1: Liên kết nền (màu xám)
        go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'),
            
        # Lớp 2: Đường đi RL (màu xanh lá)
        go.Scatter(
            x=rl_path_edges_x, y=rl_path_edges_y,
            line=dict(width=3, color='green', dash='dot'),
            hoverinfo='none',
            mode='lines',
            name='RL Path'),
            
        # Lớp 3: Nodes (Điểm)
        go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="top center",
            hoverinfo='text',
            textfont=dict(size=10),
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                color=['blue' if n.id in rl_path_history else 'lightblue' for n in nodes_data],
                size=20,
                line_width=2))
    ],
    layout=go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    
    st.plotly_chart(fig, use_container_width=True)