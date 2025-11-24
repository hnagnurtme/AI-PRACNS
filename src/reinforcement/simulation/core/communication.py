from typing import List, Dict, Optional
from .node import Node
import math


class Communication:
    def __init__(self, frequencyGHz: float, bandwidthMHz: float, transmitPowerDbW: float,
                 antennaGainDb: float, beamWidthDeg: float, maxRangeKm: float,
                 minElevationDeg: float, ipAddress: str, port: int, protocol: str = "TCP"):
        self.frequencyGHz = frequencyGHz
        self.bandwidthMHz = bandwidthMHz
        self.transmitPowerDbW = transmitPowerDbW
        self.antennaGainDb = antennaGainDb
        self.beamWidthDeg = beamWidthDeg
        self.maxRangeKm = maxRangeKm
        self.minElevationDeg = minElevationDeg
        self.ipAddress = ipAddress
        self.port = port
        self.protocol = protocol
        
        # Thông số động
        self.current_congestion = 0.0  # 0-1
        self.packet_loss_rate = 0.0    # 0-1
        self.actual_bandwidth = bandwidthMHz  # MHz
        self.link_quality = 1.0        # 0-1
        
    def update_communication_quality(self, weather_impact: float, distance: float, 
                                   traffic_load: float, time_of_day: float):
        """Cập nhật chất lượng liên kết dựa trên các yếu tố môi trường"""
        # Ảnh hưởng của thời tiết
        weather_penalty = weather_impact * 0.3
        
        # Ảnh hưởng của khoảng cách (tỷ lệ nghịch)
        distance_penalty = min(1.0, distance / self.maxRangeKm) * 0.4
        
        # Ảnh hưởng của tải traffic (theo chu kỳ ngày/đêm)
        daily_variation = 0.3 + 0.4 * abs(math.sin(2 * math.pi * time_of_day - math.pi/2))
        congestion = traffic_load * daily_variation
        self.current_congestion = min(1.0, congestion)
        
        # Tính toán packet loss rate
        self.packet_loss_rate = (weather_penalty + distance_penalty + 
                                self.current_congestion * 0.3) / 3.0
        
        # Tính toán bandwidth thực tế
        quality_factor = 1.0 - self.packet_loss_rate
        self.actual_bandwidth = self.bandwidthMHz * quality_factor
        
        # Tính toán chất lượng liên kết tổng thể
        self.link_quality = max(0.1, 1.0 - (weather_penalty + distance_penalty + 
                                           self.current_congestion) / 3.0)
    
    def get_current_delay(self) -> float:
        """Tính độ trễ hiện tại dựa trên chất lượng liên kết"""
        base_delay = 10  # ms
        congestion_delay = base_delay * (1 + 4 * self.current_congestion)
        return congestion_delay * (1.0 / max(0.1, self.link_quality))
    
    def to_dict(self):
        return {
            "frequencyGHz": self.frequencyGHz,
            "bandwidthMHz": self.bandwidthMHz,
            "actualBandwidthMHz": self.actual_bandwidth,
            "transmitPowerDbW": self.transmitPowerDbW,
            "antennaGainDb": self.antennaGainDb,
            "beamWidthDeg": self.beamWidthDeg,
            "maxRangeKm": self.maxRangeKm,
            "minElevationDeg": self.minElevationDeg,
            "ipAddress": self.ipAddress,
            "port": self.port,
            "protocol": self.protocol,
            "currentCongestion": self.current_congestion,
            "packetLossRate": self.packet_loss_rate,
            "linkQuality": self.link_quality
        }