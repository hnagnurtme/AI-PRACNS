import math
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone

class Position:
    def __init__(self, latitude: float, longitude: float, altitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.velocity_lat = 0.0
        self.velocity_lon = 0.0  
        self.velocity_alt = 0.0
        
    def to_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude
        }
    
    def to_xyz(self, R_earth: float = 6371.0) -> Tuple[float, float, float]:
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)
        R = R_earth + self.altitude
        x = R * math.cos(lat_rad) * math.cos(lon_rad)
        y = R * math.cos(lat_rad) * math.sin(lon_rad)
        z = R * math.sin(lat_rad)
        return x, y, z
    
    def update_position(self, delta_time: float):
        self.latitude += self.velocity_lat * delta_time
        self.longitude += self.velocity_lon * delta_time
        self.altitude += self.velocity_alt * delta_time
        self.latitude = max(-90, min(90, self.latitude))
        self.longitude = (self.longitude + 180) % 360 - 180

class Orbit:
    def __init__(self, semiMajorAxisKm: float = 0, eccentricity: float = 0, 
                 inclinationDeg: float = 0, raanDeg: float = 0, 
                 argumentOfPerigeeDeg: float = 0, trueAnomalyDeg: float = 0):
        self.semiMajorAxisKm = semiMajorAxisKm
        self.eccentricity = eccentricity
        self.inclinationDeg = inclinationDeg
        self.raanDeg = raanDeg
        self.argumentOfPerigeeDeg = argumentOfPerigeeDeg
        self.trueAnomalyDeg = trueAnomalyDeg
        self.angular_velocity_deg_per_sec = self._calculate_angular_velocity()
        
    def _calculate_angular_velocity(self) -> float:
        if self.semiMajorAxisKm <= 0:
            return 0.0
        GM = 398600.4418
        orbital_period = 2 * math.pi * math.sqrt(self.semiMajorAxisKm**3 / GM)
        return 360.0 / orbital_period if orbital_period > 0 else 0.0
    
    def update_orbit(self, delta_time: float):
        self.trueAnomalyDeg += self.angular_velocity_deg_per_sec * delta_time
        self.trueAnomalyDeg %= 360
    
    def to_dict(self):
        return {
            "semiMajorAxisKm": self.semiMajorAxisKm,
            "eccentricity": self.eccentricity,
            "inclinationDeg": self.inclinationDeg,
            "raanDeg": self.raanDeg,
            "argumentOfPerigeeDeg": self.argumentOfPerigeeDeg,
            "trueAnomalyDeg": self.trueAnomalyDeg,
            "angularVelocityDegPerSec": self.angular_velocity_deg_per_sec
        }

class Velocity:
    def __init__(self, velocityX: float = 0, velocityY: float = 0, velocityZ: float = 0):
        self.velocityX = velocityX
        self.velocityY = velocityY
        self.velocityZ = velocityZ
    
    def to_dict(self):
        return {
            "velocityX": self.velocityX,
            "velocityY": self.velocityY,
            "velocityZ": self.velocityZ
        }

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
        
        self.current_congestion = 0.0
        self.packet_loss_rate = 0.0
        self.actual_bandwidth = bandwidthMHz
        self.link_quality = 1.0
        
    def update_communication_quality(self, weather_impact: float, distance: float, 
                                   traffic_load: float, time_of_day: float):
        weather_penalty = weather_impact * 0.3
        distance_penalty = min(1.0, distance / self.maxRangeKm) * 0.4
        daily_variation = 0.3 + 0.4 * abs(math.sin(2 * math.pi * time_of_day - math.pi/2))
        congestion = traffic_load * daily_variation
        self.current_congestion = min(1.0, congestion)
        self.packet_loss_rate = (weather_penalty + distance_penalty + 
                                self.current_congestion * 0.3) / 3.0
        quality_factor = 1.0 - self.packet_loss_rate
        self.actual_bandwidth = self.bandwidthMHz * quality_factor
        self.link_quality = max(0.1, 1.0 - (weather_penalty + distance_penalty + 
                                           self.current_congestion) / 3.0)
    
    def get_current_delay(self) -> float:
        base_delay = 10
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

class Node:
    R_EARTH = 6371.0
    NODE_TYPE_MAX_RANGE = {
        "GROUND_STATION": 2000.0,
        "LEO_SATELLITE": 3000.0,
        "MEO_SATELLITE": 10000.0,
        "GEO_SATELLITE": 35000.0
    }
    
    MAX_RANGE_MAP = {
        ("LEO_SATELLITE", "MEO_SATELLITE"): 12000,
        ("LEO_SATELLITE", "GEO_SATELLITE"): 40000,
        ("MEO_SATELLITE", "GEO_SATELLITE"): 30000,
    }
    
    def __init__(self, nodeId: str, nodeName: str, nodeType: str, 
                 position: Position, orbit: Orbit, velocity: Velocity,
                 communication: Communication, isOperational: bool = True,
                 batteryChargePercent: int = 100, nodeProcessingDelayMs: float = 1.0,
                 packetLossRate: float = 0.0, resourceUtilization: float = 0.1,
                 packetBufferCapacity: int = 1000, currentPacketCount: int = 0,
                 weather: str = "CLEAR", healthy: bool = True, neighbors: Optional[List[str]] = None):
        
        self.nodeId = nodeId
        self.nodeName = nodeName
        self.nodeType = nodeType
        self.position = position
        self.orbit = orbit
        self.velocity = velocity
        self.communication = communication
        self.isOperational = isOperational
        self.batteryChargePercent = batteryChargePercent
        self.nodeProcessingDelayMs = nodeProcessingDelayMs
        self.packetLossRate = packetLossRate
        self.resourceUtilization = resourceUtilization
        self.packetBufferCapacity = packetBufferCapacity
        self.currentPacketCount = currentPacketCount
        self.weather = weather
        self.lastUpdated = datetime.now(timezone.utc).isoformat()
        self.healthy = healthy
        self.neighbors = neighbors if neighbors is not None else []
        
        self.failure_probability = self._calculate_base_failure_probability()
        self.congestion_level = 0.0
        self.traffic_load = 0.0
        self.energy_consumption_rate = self._calculate_energy_consumption()
        self.mobility_pattern = self._initialize_mobility_pattern()
        
    def _calculate_base_failure_probability(self) -> float:
        base_probabilities = {
            "GROUND_STATION": 0.01,
            "LEO_SATELLITE": 0.05,
            "MEO_SATELLITE": 0.03,
            "GEO_SATELLITE": 0.02
        }
        return base_probabilities.get(self.nodeType, 0.1)
    
    def _calculate_energy_consumption(self) -> float:
        consumption_rates = {
            "GROUND_STATION": 0.001,
            "LEO_SATELLITE": 0.002,
            "MEO_SATELLITE": 0.0015,
            "GEO_SATELLITE": 0.001
        }
        return consumption_rates.get(self.nodeType, 0.001)
    
    def _initialize_mobility_pattern(self):
        if self.nodeType == "GROUND_STATION":
            return {"type": "static", "velocity_range": (0, 0)}
        elif "SATELLITE" in self.nodeType:
            return {"type": "orbital", "velocity_range": (1, 10)}
        else:
            return {"type": "random_walk", "velocity_range": (0.1, 1)}
    
    def update_dynamic_parameters(self, delta_time: float, weather_impact: float, 
                                network_traffic: float, time_of_day: float):
        self.lastUpdated = datetime.now(timezone.utc).isoformat()
        self._update_position_and_motion(delta_time)
        self._update_weather_effects()
        self._update_traffic_load(network_traffic, time_of_day)
        distance_to_center = self._calculate_distance_to_center()
        self.communication.update_communication_quality(
            weather_impact, distance_to_center, self.traffic_load, time_of_day
        )
        self._update_battery_and_energy(delta_time)
        self._update_operational_status()
        self._update_packet_buffer()
    
    def _update_position_and_motion(self, delta_time: float):
        if self.nodeType == "GROUND_STATION":
            if np.random.random() < 0.01:
                self.position.velocity_lat = np.random.uniform(-0.001, 0.001)
                self.position.velocity_lon = np.random.uniform(-0.001, 0.001)
        elif "SATELLITE" in self.nodeType:
            self.orbit.update_orbit(delta_time)
        else:
            self.position.velocity_lat = np.random.uniform(-0.1, 0.1)
            self.position.velocity_lon = np.random.uniform(-0.1, 0.1)
            self.position.velocity_alt = np.random.uniform(-0.01, 0.01)
        self.position.update_position(delta_time)
    
    def _update_weather_effects(self):
        weather_transitions = {
            "CLEAR": {"CLEAR": 0.7, "CLOUDY": 0.2, "RAIN": 0.08, "STORM": 0.02},
            "CLOUDY": {"CLEAR": 0.3, "CLOUDY": 0.5, "RAIN": 0.15, "STORM": 0.05},
            "RAIN": {"CLEAR": 0.1, "CLOUDY": 0.3, "RAIN": 0.5, "STORM": 0.1},
            "STORM": {"CLEAR": 0.05, "CLOUDY": 0.15, "RAIN": 0.3, "STORM": 0.5}
        }
        current_weather_probs = weather_transitions.get(self.weather, weather_transitions["CLEAR"])
        weather_options = list(current_weather_probs.keys())
        weather_probs = list(current_weather_probs.values())
        self.weather = np.random.choice(weather_options, p=weather_probs)
    
    def _update_traffic_load(self, network_traffic: float, time_of_day: float):
        hour_variation = 0.5 + 0.5 * math.sin(2 * math.pi * time_of_day - math.pi/2)
        random_variation = np.random.uniform(0.8, 1.2)
        burst_traffic = 3.0 if np.random.random() < 0.02 else 1.0
        self.traffic_load = network_traffic * hour_variation * random_variation * burst_traffic
        self.traffic_load = max(0.1, min(5.0, self.traffic_load))
        self.congestion_level = min(1.0, self.traffic_load / 3.0)
    
    def _update_battery_and_energy(self, delta_time: float):
        if self.nodeType != "GROUND_STATION":
            base_consumption = self.energy_consumption_rate * delta_time
            traffic_consumption = base_consumption * self.traffic_load
            weather_penalty = 0.0
            if self.weather == "RAIN":
                weather_penalty = base_consumption * 0.5
            elif self.weather == "STORM":
                weather_penalty = base_consumption * 1.0
            total_consumption = base_consumption + traffic_consumption + weather_penalty
            self.batteryChargePercent -= total_consumption * 100
            if self.batteryChargePercent <= 0:
                self.batteryChargePercent = 0
                self.isOperational = False
                self.healthy = False
    
    def _update_operational_status(self):
        if self.healthy and self.batteryChargePercent > 0:
            weather_risk = 0.0
            if self.weather == "RAIN":
                weather_risk = 0.1
            elif self.weather == "STORM":
                weather_risk = 0.3
            congestion_risk = self.congestion_level * 0.2
            battery_risk = (100 - self.batteryChargePercent) / 100 * 0.1
            total_failure_prob = (self.failure_probability + weather_risk + 
                                congestion_risk + battery_risk)
            if np.random.random() < total_failure_prob * 0.01:
                self.isOperational = False
                self.healthy = False
        else:
            if np.random.random() < 0.05 and self.batteryChargePercent > 20:
                self.isOperational = True
                self.healthy = True
    
    def _update_packet_buffer(self):
        packet_change = int(self.traffic_load * 10 - 5)
        self.currentPacketCount += packet_change
        self.currentPacketCount = max(0, min(self.packetBufferCapacity, self.currentPacketCount))
        if self.currentPacketCount > self.packetBufferCapacity * 0.8:
            overflow = (self.currentPacketCount - self.packetBufferCapacity * 0.8) 
            max_overflow = self.packetBufferCapacity * 0.2
            self.packetLossRate = min(0.5, overflow / max_overflow * 0.5)
        else:
            self.packetLossRate = 0.01
    
    def _calculate_distance_to_center(self) -> float:
        return self.R_EARTH + self.position.altitude
    
    def get_current_delay(self) -> float:
        processing_delay = self.nodeProcessingDelayMs
        communication_delay = self.communication.get_current_delay()
        congestion_delay = self.congestion_level * 50
        return processing_delay + communication_delay + congestion_delay
    
    def get_link_quality_to(self, other_node: 'Node') -> float:
        if not self.isOperational or not other_node.isOperational:
            return 0.0
        pos1 = self.position.to_xyz()
        pos2 = other_node.position.to_xyz()
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
        max_range = self.communication.maxRangeKm
        if distance > max_range:
            return 0.0
        distance_quality = 1.0 - (distance / max_range)
        weather_quality = 1.0 if self.weather == "CLEAR" else 0.7
        congestion_quality = 1.0 - self.congestion_level
        return distance_quality * weather_quality * congestion_quality * self.communication.link_quality

    def to_dict(self):
        return {
            "nodeId": self.nodeId,
            "nodeName": self.nodeName,
            "nodeType": self.nodeType,
            "position": self.position.to_dict(),
            "orbit": self.orbit.to_dict(),
            "velocity": self.velocity.to_dict(),
            "communication": self.communication.to_dict(),
            "isOperational": self.isOperational,
            "batteryChargePercent": self.batteryChargePercent,
            "nodeProcessingDelayMs": self.nodeProcessingDelayMs,
            "packetLossRate": self.packetLossRate,
            "resourceUtilization": self.resourceUtilization,
            "packetBufferCapacity": self.packetBufferCapacity,
            "currentPacketCount": self.currentPacketCount,
            "weather": self.weather,
            "lastUpdated": self.lastUpdated,
            "healthy": self.healthy,
            "neighbors": self.neighbors
        }

    @classmethod
    def from_dict(cls, data: dict):
        position_data = data.get('position', {})
        position = Position(
            latitude=position_data.get('latitude', 0),
            longitude=position_data.get('longitude', 0),
            altitude=position_data.get('altitude', 0)
        )
        
        orbit_data = data.get('orbit', {})
        orbit = Orbit(
            semiMajorAxisKm=orbit_data.get('semiMajorAxisKm', 0),
            eccentricity=orbit_data.get('eccentricity', 0),
            inclinationDeg=orbit_data.get('inclinationDeg', 0),
            raanDeg=orbit_data.get('raanDeg', 0),
            argumentOfPerigeeDeg=orbit_data.get('argumentOfPerigeeDeg', 0),
            trueAnomalyDeg=orbit_data.get('trueAnomalyDeg', 0)
        )
        
        velocity_data = data.get('velocity', {})
        velocity = Velocity(
            velocityX=velocity_data.get('velocityX', 0),
            velocityY=velocity_data.get('velocityY', 0),
            velocityZ=velocity_data.get('velocityZ', 0)
        )
        
        communication_data = data.get('communication', {})
        communication = Communication(
            frequencyGHz=communication_data.get('frequencyGHz', 0),
            bandwidthMHz=communication_data.get('bandwidthMHz', 0),
            transmitPowerDbW=communication_data.get('transmitPowerDbW', 0),
            antennaGainDb=communication_data.get('antennaGainDb', 0),
            beamWidthDeg=communication_data.get('beamWidthDeg', 0),
            maxRangeKm=communication_data.get('maxRangeKm', 0),
            minElevationDeg=communication_data.get('minElevationDeg', 0),
            ipAddress=communication_data.get('ipAddress', ''),
            port=communication_data.get('port', 0),
            protocol=communication_data.get('protocol', 'TCP')
        )
        
        return cls(
            nodeId=data.get('nodeId', 'default_node_id'),
            nodeName=data.get('nodeName', 'default_node_name'),
            nodeType=data.get('nodeType', 'UNKNOWN'),
            position=position,
            orbit=orbit,
            velocity=velocity,
            communication=communication,
            isOperational=data.get('isOperational', True),
            batteryChargePercent=data.get('batteryChargePercent', 100),
            nodeProcessingDelayMs=data.get('nodeProcessingDelayMs', 1.0),
            packetLossRate=data.get('packetLossRate', 0.0),
            resourceUtilization=data.get('resourceUtilization', 0.1),
            packetBufferCapacity=data.get('packetBufferCapacity', 1000),
            currentPacketCount=data.get('currentPacketCount', 0),
            weather=data.get('weather', 'CLEAR'),
            healthy=data.get('healthy', True),
            neighbors=data.get('neighbors', [])
        )