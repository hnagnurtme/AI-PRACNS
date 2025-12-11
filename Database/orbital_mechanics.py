"""
Orbital Mechanics Engine
Tính toán vị trí satellite theo thời gian thực dựa trên orbital parameters
"""
import math
from datetime import datetime
from typing import Dict, Optional
from config_loader import get_config

class OrbitalMechanics:
    """Engine để tính toán vị trí satellite dựa trên quỹ đạo"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize orbital mechanics engine
        
        Args:
            config_file: Optional path to config file
        """
        self.config = get_config(config_file)
        
        # Load Earth constants from config
        earth_constants = self.config.get_earth_constants()
        self.EARTH_RADIUS_KM = earth_constants.get('radius_km', 6371.0)
        self.EARTH_RADIUS_M = earth_constants.get('radius_m', 6371000.0)
        
        # Load orbital periods from config
        leo_config = self.config.get_orbital_config('leo')
        meo_config = self.config.get_orbital_config('meo')
        geo_config = self.config.get_orbital_config('geo')
        
        self.LEO_PERIOD = leo_config.get('period_seconds', 5400)  # ~90 minutes
        self.MEO_PERIOD = meo_config.get('period_seconds', 43200)  # ~12 hours
        self.GEO_PERIOD = geo_config.get('period_seconds', 86400)  # 24 hours
        
        # Load speeds
        self.LEO_SPEED_MS = leo_config.get('speed_m_per_s', 7800)
        self.MEO_SPEED_MS = meo_config.get('speed_m_per_s', 3900)
        self.GEO_SPEED_MS = geo_config.get('speed_m_per_s', 3070)
        
        # Load default semi-major axes
        self.LEO_DEFAULT_SEMI_MAJOR = leo_config.get('default_semi_major_axis_km', 6928.0)
        self.MEO_DEFAULT_SEMI_MAJOR = meo_config.get('default_semi_major_axis_km', 26562.0)
        self.GEO_DEFAULT_SEMI_MAJOR = geo_config.get('default_semi_major_axis_km', 42164.0)
    
    def calculate_position_at_time(self, node: Dict, timestamp: float) -> Dict:
        """
        Tính vị trí satellite tại thời điểm cụ thể
        
        Args:
            node: Node dictionary với orbit parameters
            timestamp: Unix timestamp (seconds)
        
        Returns:
            Position dictionary với latitude, longitude, altitude
        """
        node_type = node.get('nodeType', '')
        
        if node_type == 'LEO_SATELLITE':
            return self._calculate_leo_position(node, timestamp)
        elif node_type == 'MEO_SATELLITE':
            return self._calculate_meo_position(node, timestamp)
        elif node_type == 'GEO_SATELLITE':
            return self._calculate_geo_position(node, timestamp)
        else:
            # Ground station - static position
            return node.get('position', {
                'latitude': 0.0,
                'longitude': 0.0,
                'altitude': 0.0
            })
    
    def _calculate_leo_position(self, node: Dict, timestamp: float) -> Dict:
        """
        Tính vị trí LEO satellite (circular orbit, period ~90 minutes)
        
        LEO satellites:
        - Altitude: ~550-2000 km
        - Period: ~90 minutes
        - Speed: ~7.8 km/s
        """
        orbit = node.get('orbit', {})
        initial_position = node.get('position', {})
        
        # Get initial parameters
        initial_longitude = initial_position.get('longitude', orbit.get('raanDeg', 0))
        initial_latitude = initial_position.get('latitude', orbit.get('inclinationDeg', 0))
        semi_major_axis_km = orbit.get('semiMajorAxisKm', self.LEO_DEFAULT_SEMI_MAJOR)
        
        # Get epoch time (thời điểm ban đầu)
        epoch_time = node.get('epochTime')
        if epoch_time is None:
            # Use current time as epoch if not set
            epoch_time = timestamp
            # Store epoch for future use
            node['epochTime'] = epoch_time
        
        # Calculate elapsed time since epoch
        elapsed = (timestamp - epoch_time) % self.LEO_PERIOD
        
        # Angular velocity (radians per second)
        angular_velocity_rad = (2 * math.pi) / self.LEO_PERIOD
        
        # Current angle in radians
        initial_angle_rad = math.radians(initial_longitude)
        current_angle_rad = initial_angle_rad + (angular_velocity_rad * elapsed)
        
        # Convert to degrees
        current_longitude = math.degrees(current_angle_rad) % 360
        if current_longitude < 0:
            current_longitude += 360
        
        # Altitude (semi-major axis - Earth radius)
        altitude_m = (semi_major_axis_km - self.EARTH_RADIUS_KM) * 1000
        
        return {
            'latitude': initial_latitude,  # LEO thường equatorial (0°)
            'longitude': current_longitude,
            'altitude': altitude_m
        }
    
    def _calculate_meo_position(self, node: Dict, timestamp: float) -> Dict:
        """
        Tính vị trí MEO satellite (circular orbit, period ~12 hours)
        
        MEO satellites:
        - Altitude: ~20,000 km
        - Period: ~12 hours
        - Speed: ~3.9 km/s
        """
        orbit = node.get('orbit', {})
        initial_position = node.get('position', {})
        
        # Get initial parameters
        initial_longitude = initial_position.get('longitude', orbit.get('raanDeg', 0))
        initial_latitude = initial_position.get('latitude', orbit.get('inclinationDeg', 0))
        semi_major_axis_km = orbit.get('semiMajorAxisKm', self.MEO_DEFAULT_SEMI_MAJOR)
        
        # Get epoch time
        epoch_time = node.get('epochTime')
        if epoch_time is None:
            epoch_time = timestamp
            node['epochTime'] = epoch_time
        
        # Calculate elapsed time
        elapsed = (timestamp - epoch_time) % self.MEO_PERIOD
        
        # Angular velocity
        angular_velocity_rad = (2 * math.pi) / self.MEO_PERIOD
        
        # Current angle
        initial_angle_rad = math.radians(initial_longitude)
        current_angle_rad = initial_angle_rad + (angular_velocity_rad * elapsed)
        
        current_longitude = math.degrees(current_angle_rad) % 360
        if current_longitude < 0:
            current_longitude += 360
        
        # Altitude
        altitude_m = (semi_major_axis_km - self.EARTH_RADIUS_KM) * 1000
        
        return {
            'latitude': initial_latitude,  # MEO có thể có inclination
            'longitude': current_longitude,
            'altitude': altitude_m
        }
    
    def _calculate_geo_position(self, node: Dict, timestamp: float) -> Dict:
        """
        Tính vị trí GEO satellite (geostationary, period = 24 hours)
        
        GEO satellites:
        - Altitude: ~35,786 km
        - Period: 24 hours (geostationary)
        - Speed: ~3.07 km/s
        - Vị trí cố định so với mặt đất
        """
        orbit = node.get('orbit', {})
        initial_position = node.get('position', {})
        
        # GEO satellites are geostationary - position doesn't change relative to Earth
        # But we still update timestamp for consistency
        
        longitude = initial_position.get('longitude', orbit.get('raanDeg', 0))
        latitude = initial_position.get('latitude', 0.0)  # GEO thường ở equator
        semi_major_axis_km = orbit.get('semiMajorAxisKm', self.GEO_DEFAULT_SEMI_MAJOR)
        
        altitude_m = (semi_major_axis_km - self.EARTH_RADIUS_KM) * 1000
        
        return {
            'latitude': latitude,
            'longitude': longitude,  # GEO không đổi longitude
            'altitude': altitude_m
        }
    
    def calculate_velocity(self, node: Dict, timestamp: float) -> Dict:
        """
        Tính vận tốc satellite tại thời điểm cụ thể
        
        Returns:
            Velocity dictionary với velocityX, velocityY, velocityZ (m/s)
        """
        node_type = node.get('nodeType', '')
        
        if node_type == 'LEO_SATELLITE':
            speed_ms = self.LEO_SPEED_MS
        elif node_type == 'MEO_SATELLITE':
            speed_ms = self.MEO_SPEED_MS
        elif node_type == 'GEO_SATELLITE':
            speed_ms = self.GEO_SPEED_MS
        else:
            # Ground station - no velocity
            return {'velocityX': 0.0, 'velocityY': 0.0, 'velocityZ': 0.0}
        
        # Calculate velocity components (simplified - circular orbit)
        position = self.calculate_position_at_time(node, timestamp)
        longitude_rad = math.radians(position['longitude'])
        latitude_rad = math.radians(position['latitude'])
        
        # Velocity in circular orbit (tangential to orbit)
        velocity_x = -speed_ms * math.sin(longitude_rad) * math.cos(latitude_rad)
        velocity_y = speed_ms * math.cos(longitude_rad) * math.cos(latitude_rad)
        velocity_z = speed_ms * math.sin(latitude_rad)
        
        return {
            'velocityX': round(velocity_x, 3),
            'velocityY': round(velocity_y, 3),
            'velocityZ': round(velocity_z, 3)
        }

