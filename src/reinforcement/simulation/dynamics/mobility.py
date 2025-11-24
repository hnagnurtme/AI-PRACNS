
import numpy as np
from typing import Dict, List, Tuple, Optional

class MobilityManager:
    """
    Manages the movement of nodes in the SAGIN and provides dynamic neighbor computation.
    This is the central place for computing neighbors based on real-time positions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.nodes = {}  # This will be populated from the environment
        self.MAX_NEIGHBORS = self.config.get('max_neighbors', 10)
        
        # Cache for neighbor computation to improve performance
        self._neighbor_cache = {}
        self._cache_timestamp = 0.0
        self._cache_validity_duration = 0.1  # seconds

    def set_nodes(self, nodes: Dict):
        """
        Sets the nodes in the network for the mobility manager to use.
        """
        self.nodes = nodes

    def update_nodes(self, delta_time: float):
        """Updates the position of all nodes."""
        for node_id, node in self.nodes.items():
            if hasattr(node, 'update_position'):
                node.update_position(delta_time)
        
        # Invalidate cache when nodes move
        self._neighbor_cache.clear()

    def get_current_neighbors(self, node_id: str, dynamic_state: Optional[Dict] = None) -> List[str]:
        """
        Gets the current neighbors of a node, considering their dynamic positions.
        This is the authoritative method for computing neighbors in dynamic scenarios.
        
        Args:
            node_id: ID of the node to get neighbors for
            dynamic_state: Optional dynamic state with weather_impact, traffic_load, etc.
        
        Returns:
            List of neighbor node IDs, sorted by link quality
        """
        if dynamic_state is None:
            dynamic_state = {}
            
        # Check cache
        cache_key = (node_id, self._cache_timestamp)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        
        node = self.nodes.get(node_id)
        if not node:
            return []

        # Handle both dict and object nodes
        is_operational = node.get('isOperational', True) if isinstance(node, dict) else node.isOperational
        if not is_operational:
            return []

        potential_neighbors = []
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue

            # Check if other node is operational
            other_operational = (other_node.get('isOperational', True) if isinstance(other_node, dict)
                               else other_node.isOperational)
            if not other_operational:
                continue

            # Get positions (handle both dict and object)
            node_pos = node.get('position') if isinstance(node, dict) else node.position
            other_pos = other_node.get('position') if isinstance(other_node, dict) else other_node.position

            distance = self._calculate_distance(node_pos, other_pos)

            # Get communication range (handle both dict and object)
            comm_range = (node.get('communication', {}).get('maxRangeKm', 1000.0)
                         if isinstance(node, dict) else node.communication.maxRangeKm)

            # Check if the other node is within communication range
            if distance <= comm_range:
                link_quality = self._calculate_dynamic_link_quality(
                    distance, dynamic_state
                )
                if link_quality > 0.1:  # Minimum link quality threshold
                    potential_neighbors.append((other_id, distance, link_quality))

        # Sort neighbors by link quality in descending order
        potential_neighbors.sort(key=lambda x: x[2], reverse=True)

        # Return the IDs of the best neighbors, up to MAX_NEIGHBORS
        neighbor_ids = [n[0] for n in potential_neighbors[:self.MAX_NEIGHBORS]]
        
        # Cache the result
        self._neighbor_cache[cache_key] = neighbor_ids
        
        return neighbor_ids
    
    def compute_all_neighbors(self, dynamic_state: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        Compute neighbors for all nodes in the network.
        
        Args:
            dynamic_state: Optional dynamic state with weather_impact, traffic_load, etc.
        
        Returns:
            Dictionary mapping node_id to list of neighbor node_ids
        """
        all_neighbors = {}
        for node_id in self.nodes.keys():
            all_neighbors[node_id] = self.get_current_neighbors(node_id, dynamic_state)
        return all_neighbors
    
    def update_node_neighbors(self, dynamic_state: Optional[Dict] = None):
        """
        Update the 'neighbors' attribute of all nodes with current dynamic neighbors.
        This ensures baseline algorithms using node.neighbors get updated information.
        
        Args:
            dynamic_state: Optional dynamic state with weather_impact, traffic_load, etc.
        """
        for node_id, node in self.nodes.items():
            current_neighbors = self.get_current_neighbors(node_id, dynamic_state)
            if isinstance(node, dict):
                node['neighbors'] = current_neighbors
            else:
                node.neighbors = current_neighbors

    def _calculate_distance(self, pos1, pos2) -> float:
        """
        Calculates the Euclidean distance between two positions.
        Assumes positions are given in a consistent Cartesian-like coordinate system.
        Handles both dict and object positions.
        """
        # Handle both dict and object positions
        lat1 = pos1.get('latitude', 0) if isinstance(pos1, dict) else pos1.latitude
        lon1 = pos1.get('longitude', 0) if isinstance(pos1, dict) else pos1.longitude
        alt1 = pos1.get('altitude', 0) if isinstance(pos1, dict) else pos1.altitude

        lat2 = pos2.get('latitude', 0) if isinstance(pos2, dict) else pos2.latitude
        lon2 = pos2.get('longitude', 0) if isinstance(pos2, dict) else pos2.longitude
        alt2 = pos2.get('altitude', 0) if isinstance(pos2, dict) else pos2.altitude

        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2 + (alt1 - alt2)**2)

    def _calculate_dynamic_link_quality(self, distance: float, dynamic_state: Dict) -> float:
        """
        Calculates the quality of a link, considering dynamic factors.
        """
        # Base quality is inversely proportional to distance
        base_quality = 1.0 / (1.0 + distance)
        
        # Factor in weather impact
        weather_impact = dynamic_state.get('weather_impact', 1.0)
        
        # A simple combined quality metric
        quality = base_quality * weather_impact
        
        return quality

    def reset(self):
        """
        Resets the state of the mobility manager.
        In this implementation, there is no internal state to reset besides what's in the node objects themselves.
        """
        pass
