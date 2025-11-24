import numpy as np

class DynamicModels:
    @staticmethod
    def orbital_motion(semi_major_axis, eccentricity, inclination, raan, arg_perigee, true_anomaly, delta_time):
        
        new_true_anomaly = true_anomaly + 0.1 * delta_time
        return new_true_anomaly % 360
    
    @staticmethod
    def weather_impact(current_weather, delta_time):
        # Giả lập sự thay đổi thời tiết
        transitions = {
            'CLEAR': {'CLEAR': 0.7, 'CLOUDY': 0.2, 'RAIN': 0.08, 'STORM': 0.02},
            'CLOUDY': {'CLEAR': 0.3, 'CLOUDY': 0.5, 'RAIN': 0.15, 'STORM': 0.05},
            'RAIN': {'CLEAR': 0.1, 'CLOUDY': 0.3, 'RAIN': 0.5, 'STORM': 0.1},
            'STORM': {'CLEAR': 0.05, 'CLOUDY': 0.15, 'RAIN': 0.3, 'STORM': 0.5}
        }
        return transitions.get(current_weather, transitions['CLEAR'])