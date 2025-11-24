
import numpy as np
from typing import Dict, Any,Optional

class TrafficModel:
    """
    Manages traffic load in the network.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_load = self.config.get('base_load', 1.0)
        self.peak_load = self.config.get('peak_load', 3.0)
        self.period = self.config.get('period', 24 * 60 * 60)  # 24 hours in seconds
        self.current_load = self.base_load

    def update(self, simulation_time: float) -> float:
        """
        Updates the traffic load based on the simulation time.
        This model uses a sinusoidal pattern to simulate daily traffic variations.
        """
        # Simulate a daily traffic pattern (e.g., high during the day, low at night)
        time_of_day = simulation_time % self.period
        # Scale time_of_day to be from 0 to 2*pi
        scaled_time = (time_of_day / self.period) * 2 * np.pi
        
        # Use a sine wave to model traffic load
        # The sine wave is shifted to have its peak during the "day"
        load_multiplier = (np.sin(scaled_time - np.pi / 2) + 1) / 2  # Ranges from 0 to 1
        
        self.current_load = self.base_load + (self.peak_load - self.base_load) * load_multiplier
        return self.current_load

    def get_node_load(self, node_id: str) -> float:
        """
        Gets the current traffic load for a specific node.
        For simplicity, this model returns a global traffic load.
        """
        return self.current_load

    def reset(self):
        """Resets the traffic model to its initial state."""
        self.current_load = self.base_load
