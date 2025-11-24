import math
import numpy as np

def to_cartesian_ecef(position: dict) -> np.ndarray:
    lat = position.get('latitude', 0)
    lon = position.get('longitude', 0)
    alt = position.get('altitude', 0)
    
    R_earth = 6371.0
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    R = R_earth + alt
    
    x = R * math.cos(lat_rad) * math.cos(lon_rad)
    y = R * math.cos(lat_rad) * math.sin(lon_rad)
    z = R * math.sin(lat_rad)
    
    return np.array([x, y, z])

def calculate_link_budget_snr(node1: dict, node2: dict, distance_km: float, weather: str = 'CLEAR') -> float:
    # Giả lập tính toán SNR đơn giản
    base_snr = 30.0
    weather_penalty = 0.0
    if weather == 'RAIN':
        weather_penalty = 10.0
    elif weather == 'STORM':
        weather_penalty = 20.0
        
    distance_penalty = distance_km * 0.001
    
    snr = base_snr - weather_penalty - distance_penalty
    return max(snr, 0.0)