# python/utils/math_utils.py

import math
import numpy as np
from typing import Dict, Any

# Import các hằng số cấu hình từ file constants.py cùng thư mục
# Nếu bạn chưa tách file, hãy đảm bảo các biến này có giá trị
from .constants import (
    BOLTZMANN_CONST, 
    NOISE_TEMP_K, 
    MAX_SNR_DB, 
    WEATHER_LOSS_MAP
)

# --- Hằng số WGS-84 (Dùng riêng cho tính toán tọa độ) ---
EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1 / 298.257223563
EARTH_ECCEN_SQUARED = 2 * EARTH_FLATTENING - EARTH_FLATTENING**2

def to_cartesian_ecef(pos: Dict[str, float]) -> np.ndarray:
    """
    Chuyển đổi Lat/Lon/Alt sang vector ECEF (x, y, z).
    Input: 
        pos: dict {'latitude': float (deg), 'longitude': float (deg), 'altitude': float (km)}
    Output: 
        np.array([x, y, z])
    """
    # Lấy dữ liệu an toàn, mặc định 0.0 nếu thiếu
    lat_deg = pos.get('latitude', 0.0)
    lon_deg = pos.get('longitude', 0.0)
    alt_km = pos.get('altitude', 0.0)

    # Chuyển sang radian
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # Công thức WGS-84 chuẩn
    n_lat = EARTH_RADIUS_KM / math.sqrt(1 - EARTH_ECCEN_SQUARED * sin_lat * sin_lat)

    x = (n_lat + alt_km) * cos_lat * cos_lon
    y = (n_lat + alt_km) * cos_lat * sin_lon
    z = ((1 - EARTH_ECCEN_SQUARED) * n_lat + alt_km) * sin_lat

    return np.array([x, y, z], dtype=np.float32)

def calculate_link_budget_snr(tx_node: Dict[str, Any], rx_node: Dict[str, Any], dist_km: float, weather: str) -> float:
    """
    Tính SNR (Signal-to-Noise Ratio) dựa trên Link Budget Equation.
    SNR (dB) = Pt + Gt + Gr - FSPL - L_weather - Noise
    """
    # 1. Validate input cơ bản
    if dist_km <= 0:
        return MAX_SNR_DB
    
    tx_comm = tx_node.get('communication', {})
    rx_comm = rx_node.get('communication', {})

    # Nếu thiếu thông tin truyền thông, trả về SNR cực thấp (mất kết nối)
    if not tx_comm or not rx_comm:
        return -100.0

    freq_ghz = tx_comm.get('frequencyGHz', 2.4)
    if freq_ghz <= 0: freq_ghz = 2.4

    # 2. Tính Free Space Path Loss (FSPL)
    # FSPL(dB) = 92.45 + 20log10(d_km) + 20log10(f_GHz)
    fspl = 92.45 + 20 * math.log10(dist_km) + 20 * math.log10(freq_ghz)
    
    # 3. Các tham số Gain & Power
    pt = tx_comm.get('transmitPowerDbW', 10.0)
    gt = tx_comm.get('antennaGainDb', 0.0)
    gr = rx_comm.get('antennaGainDb', 0.0)
    
    # 4. Suy hao thời tiết
    l_weather = WEATHER_LOSS_MAP.get(weather, 0.0)
    
    # 5. Tính Noise Power (Thermal Noise)
    # Noise (dBW) = 10log10(k * T * B)
    # k = 1.38e-23 J/K ~ -228.6 dBW/K/Hz
    bandwidth_mhz = tx_comm.get('bandwidthMHz', 20.0)
    bw_hz = bandwidth_mhz * 1e6
    
    if bw_hz <= 0: bw_hz = 1.0 # Tránh log(0)
    
    noise_dbw = BOLTZMANN_CONST + 10 * math.log10(NOISE_TEMP_K) + 10 * math.log10(bw_hz)
    
    # 6. Tổng hợp SNR
    snr = pt + gt + gr - fspl - l_weather - noise_dbw
    
    return float(snr)