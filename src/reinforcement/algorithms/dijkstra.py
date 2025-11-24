"""
Dijkstra's shortest path algorithm for SAGIN routing.
Adapted for satellite-aerial-ground integrated networks.
"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from data.mongodb.connection import MongoDBManager


class DijkstraRouter:
    """
    Implements Dijkstra's algorithm for finding optimal paths in SAGIN networks.
    Considers link quality, delay, and distance as edge weights.
    """

    def __init__(self, db_manager: MongoDBManager, weight_config: Optional[Dict] = None):
        """
        Initialize Dijkstra router.

        Args:
            db_manager: Database manager for accessing network state
            weight_config: Configuration for edge weight calculation
        """
        self.db_manager = db_manager
        self.weight_config = weight_config or {
            'delay_weight': 1.0,
            'distance_weight': 0.1,
            'quality_weight': 0.5,
            'battery_weight': 0.3
        }

    def find_shortest_path(self, source: str, destination: str,
                          nodes: Dict[str, Any]) -> Optional[List[str]]:
        """
        Find shortest path from source to destination using Dijkstra's algorithm.

        Args:
            source: Source node ID
            destination: Destination node ID
            nodes: Dictionary of all network nodes

        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        if source not in nodes or destination not in nodes:
            return None

        if source == destination:
            return [source]

        # Initialize distances and previous nodes
        distances = {node_id: float('inf') for node_id in nodes}
        distances[source] = 0
        previous = {node_id: None for node_id in nodes}
        visited = set()

        # Priority queue: (distance, node_id)
        pq = [(0, source)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            # Skip if already visited
            if current_node in visited:
                continue

            visited.add(current_node)

            # Found destination
            if current_node == destination:
                break

            # Skip if this path is worse than a known path
            if current_dist > distances[current_node]:
                continue

            # Get current node data
            current_node_data = nodes.get(current_node)
            if not current_node_data:
                continue

            # Check if node is operational
            if not current_node_data.get('isOperational', True):
                continue

            # Explore neighbors
            neighbors = self._get_neighbors(current_node, nodes)

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue

                neighbor_data = nodes.get(neighbor_id)
                if not neighbor_data or not neighbor_data.get('isOperational', True):
                    continue

                # Calculate edge weight
                edge_weight = self._calculate_edge_weight(
                    current_node_data, neighbor_data
                )

                new_distance = current_dist + edge_weight

                # Update if found shorter path
                if new_distance < distances[neighbor_id]:
                    distances[neighbor_id] = new_distance
                    previous[neighbor_id] = current_node
                    heapq.heappush(pq, (new_distance, neighbor_id))

        # Reconstruct path
        if distances[destination] == float('inf'):
            return None  # No path found

        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()
        return path if path[0] == source else None

    def _get_neighbors(self, node_id: str, nodes: Dict[str, Any]) -> List[str]:
        """
        Get valid neighbors for a node.

        Args:
            node_id: Node ID
            nodes: Dictionary of all network nodes

        Returns:
            List of neighbor node IDs
        """
        node_data = nodes.get(node_id)
        if not node_data:
            return []

        # Get neighbors from node data
        neighbors = node_data.get('neighbors', [])

        # If no pre-computed neighbors, compute based on distance/range
        if not neighbors:
            neighbors = self._compute_neighbors(node_id, nodes)

        return neighbors

    def _compute_neighbors(self, node_id: str, nodes: Dict[str, Any]) -> List[str]:
        """
        Compute neighbors based on communication range and line of sight.

        Args:
            node_id: Node ID
            nodes: Dictionary of all network nodes

        Returns:
            List of neighbor node IDs
        """
        node_data = nodes.get(node_id)
        if not node_data:
            return []

        node_pos = node_data.get('position', {})
        node_type = node_data.get('nodeType', 'UNKNOWN')

        # Get communication range based on node type
        comm_range = self._get_communication_range(node_type)

        neighbors = []
        for other_id, other_data in nodes.items():
            if other_id == node_id:
                continue

            if not other_data.get('isOperational', True):
                continue

            other_pos = other_data.get('position', {})
            distance = self._calculate_distance(node_pos, other_pos)

            if distance <= comm_range:
                neighbors.append(other_id)

        return neighbors

    def _calculate_edge_weight(self, from_node: Dict, to_node: Dict) -> float:
        """
        Calculate edge weight considering multiple factors.

        Args:
            from_node: Source node data
            to_node: Destination node data

        Returns:
            Edge weight (lower is better)
        """
        # Base delay
        delay = from_node.get('nodeProcessingDelayMs', 5.0)

        # Distance component
        from_pos = from_node.get('position', {})
        to_pos = to_node.get('position', {})
        distance = self._calculate_distance(from_pos, to_pos)

        # Link quality (inverse - higher quality = lower weight)
        link_quality = to_node.get('link_quality', 0.8)
        quality_factor = 1.0 / max(link_quality, 0.1)

        # Battery consideration (prefer nodes with higher battery)
        battery = to_node.get('batteryChargePercent', 100.0)
        battery_factor = 1.0 + (100.0 - battery) / 100.0

        # Congestion penalty
        congestion = to_node.get('resourceUtilization', 0.0)
        congestion_factor = 1.0 + congestion

        # Calculate weighted sum
        weight = (
            self.weight_config['delay_weight'] * delay +
            self.weight_config['distance_weight'] * distance +
            self.weight_config['quality_weight'] * quality_factor +
            self.weight_config['battery_weight'] * battery_factor
        ) * congestion_factor

        return weight

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """
        Calculate 3D Euclidean distance between two positions.

        Args:
            pos1: First position {lat, lon, alt}
            pos2: Second position {lat, lon, alt}

        Returns:
            Distance in kilometers
        """
        lat1, lon1, alt1 = pos1.get('lat', 0), pos1.get('lon', 0), pos1.get('alt', 0)
        lat2, lon2, alt2 = pos2.get('lat', 0), pos2.get('lon', 0), pos2.get('alt', 0)

        # Simple Euclidean distance (for more accuracy, use haversine for lat/lon)
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)

        # Haversine formula for surface distance
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        earth_radius = 6371.0  # km
        surface_distance = earth_radius * c

        # Add altitude difference
        altitude_diff = abs(alt2 - alt1)

        # 3D distance
        distance = np.sqrt(surface_distance**2 + altitude_diff**2)

        return distance

    def _get_communication_range(self, node_type: str) -> float:
        """
        Get communication range based on node type.

        Args:
            node_type: Type of node

        Returns:
            Communication range in kilometers
        """
        ranges = {
            'GEO_SATELLITE': 40000.0,
            'MEO_SATELLITE': 10000.0,
            'LEO_SATELLITE': 3000.0,
            'UAV': 100.0,
            'GROUND_STATION': 5000.0,
        }
        return ranges.get(node_type, 1000.0)

    def route_packet(self, packet_data: Dict[str, Any], nodes: Dict[str, Any],
                    max_hops: int = 15) -> Dict[str, Any]:
        """
        Route a packet using Dijkstra's algorithm.

        Args:
            packet_data: Packet information including source and destination
            nodes: Dictionary of all network nodes
            max_hops: Maximum number of hops allowed

        Returns:
            Routing result with path, metrics, and success status
        """
        source = packet_data.get('current_holding_node_id')
        destination = packet_data.get('station_dest')

        if not source or not destination:
            return {
                'success': False,
                'path': [],
                'total_delay': 0.0,
                'hops': 0,
                'reason': 'Invalid source or destination'
            }

        # Find shortest path
        path = self.find_shortest_path(source, destination, nodes)

        if path is None or len(path) > max_hops + 1:
            return {
                'success': False,
                'path': path or [],
                'total_delay': 0.0,
                'hops': len(path) - 1 if path else 0,
                'reason': 'No valid path found' if path is None else 'Path exceeds max hops'
            }

        # Calculate metrics for the path
        total_delay = 0.0
        min_quality = 1.0

        for i in range(len(path) - 1):
            node_data = nodes.get(path[i])
            if node_data:
                total_delay += node_data.get('nodeProcessingDelayMs', 5.0)
                min_quality = min(min_quality, node_data.get('link_quality', 1.0))

        return {
            'success': True,
            'path': path,
            'total_delay': total_delay,
            'hops': len(path) - 1,
            'min_quality': min_quality,
            'algorithm': 'Dijkstra'
        }
