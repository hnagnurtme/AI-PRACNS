import math
from typing import Tuple, List, Optional
from datetime import datetime, timezone
import numpy as np

class Position:
    def __init__(self, latitude: float, longitude: float, altitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.velocity_lat = 0.0  # deg/sec
        self.velocity_lon = 0.0  # deg/sec  
        self.velocity_alt = 0.0  # km/sec
        
    def to_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude
        }
    
    def to_xyz(self, R_earth: float = 6371.0) -> Tuple[float, float, float]:
        """Chuyển đổi tọa độ địa lý sang tọa độ Descartes 3D"""
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)
        R = R_earth + self.altitude
        x = R * math.cos(lat_rad) * math.cos(lon_rad)
        y = R * math.cos(lat_rad) * math.sin(lon_rad)
        z = R * math.sin(lat_rad)
        return x, y, z
    
    def update_position(self, delta_time: float):
        """Cập nhật vị trí dựa trên vận tốc"""
        self.latitude += self.velocity_lat * delta_time
        self.longitude += self.velocity_lon * delta_time
        self.altitude += self.velocity_alt * delta_time
        
        # Giới hạn latitude
        self.latitude = max(-90, min(90, self.latitude))
        # Giới hạn longitude trong khoảng [-180, 180]
        self.longitude = (self.longitude + 180) % 360 - 180