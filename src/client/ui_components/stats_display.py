# client_app/ui_components/stats_display.py
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any

def display_delay_comparison_chart(df: pd.DataFrame, theory_delay: float):
    """Hiển thị biểu đồ so sánh độ trễ RL vs Lý thuyết."""
    
    successful_df = df[df['status'] == 'SUCCESS']
    if successful_df.empty:
        st.warning("No successful packets received to plot the distribution.")
        return
        
    avg_rl_delay = successful_df['rl_delay_ms'].mean()
    
    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Theory Delay (Benchmark)", f"{theory_delay:.2f} ms")
    col_m2.metric("Avg. RL Delay", f"{avg_rl_delay:.2f} ms", delta=f"{avg_rl_delay - theory_delay:.2f} ms")
    col_m3.metric("Avg. RTT", f"{successful_df['rtt_ms'].mean():.2f} ms")
    
    # Histogram
    st.subheader("RL Delay Distribution")
    fig = px.histogram(
        successful_df, 
        x='rl_delay_ms', 
        nbins=20, 
        title="Frequency of Actual RL Delays Under Load"
    )
    fig.add_vline(x=theory_delay, line_dash="dash", line_color="red", annotation_text="Theory Benchmark")
    st.plotly_chart(fig, use_container_width=True)

def display_incoming_data_table(incoming_data: List[Dict[str, Any]]):
    """Hiển thị bảng các gói DATA nhận được từ Client ngoài."""
    
    st.subheader("Incoming DATA Traffic Log")
    
    if not incoming_data:
        st.info("No external DATA packets received yet.")
        return

    # Chuyển list of dicts thành DataFrame
    df_incoming = pd.DataFrame([
        {'ID': d['packetId'], 'Source': d['sourceUserId'], 'Size (B)': d['payloadSizeByte'], 'Service': d['serviceType']}
        for d in incoming_data
    ])
    st.dataframe(df_incoming, use_container_width=True)