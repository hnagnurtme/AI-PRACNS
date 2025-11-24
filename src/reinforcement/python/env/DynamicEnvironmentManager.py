from models.Node import Node
from typing import Tuple, List, Optional
from datetime import datetime, timezone
import numpy as np
import math

class DynamicEnvironmentManager:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.simulation_time = 0.0
        self.weather_model = WeatherModel()
        self.traffic_model = TrafficModel(len(nodes))
        
    def update_environment(self, delta_time: float):
        """Cập nhật toàn bộ môi trường"""
        self.simulation_time += delta_time
        
        # Tính time_of_day (0-1) từ simulation_time
        time_of_day = (self.simulation_time % 86400) / 86400  # Giả sử 1 ngày = 86400 giây
        
        # Cập nhật mô hình thời tiết và traffic
        network_traffic = self.traffic_model.update_traffic(time_of_day)
        weather_impact = self.weather_model.update_weather()
        
        # Cập nhật từng node
        for node in self.nodes:
            node.update_dynamic_parameters(delta_time, weather_impact, 
                                         network_traffic, time_of_day)

class WeatherModel:
    def __init__(self):
        self.weather_intensity = 0.0
        
    def update_weather(self) -> float:
        """Cập nhật cường độ thời tiết toàn cục"""
        # Mô hình thời tiết theo chu kỳ
        time_variation = 0.5 + 0.5 * math.sin(self.weather_intensity)
        random_variation = np.random.uniform(0.8, 1.2)
        
        self.weather_intensity = time_variation * random_variation
        return min(1.0, self.weather_intensity)

class TrafficModel:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.base_traffic = 1.0
        
    def update_traffic(self, time_of_day: float) -> float:
        """Cập nhật lưu lượng mạng toàn cục"""
        # Pattern theo giờ
        hour_pattern = 0.5 + 0.5 * math.sin(2 * math.pi * time_of_day - math.pi/2)
        
        # Nhiễu ngẫu nhiên
        random_noise = np.random.uniform(0.7, 1.3)
        
        # Sự kiện đặc biệt
        special_event = 3.0 if np.random.random() < 0.01 else 1.0
        
        self.base_traffic = hour_pattern * random_noise * special_event
        return self.base_traffic