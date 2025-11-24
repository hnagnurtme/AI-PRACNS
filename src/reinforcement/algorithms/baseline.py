"""
Baseline routing algorithms for comparison with RL-based routing.
Includes greedy, random, and other simple routing strategies.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Any
from data.mongodb.connection import MongoDBManager


class GreedyRouter:
    """
    Greedy routing algorithm that always selects the neighbor
    closest to the destination.
    """

    def __init__(self, db_manager: MongoDBManager):
        """
        Initialize greedy router.

        Args:
            db_manager: Database manager for accessing network state
        """
        self.db_manager = db_manager

    def route_packet(self, packet_data: Dict[str, Any], nodes: Dict[str, Any],
                    max_hops: int = 15) -> Dict[str, Any]:
        """
        Route a packet using greedy algorithm.

        Args:
            packet_data: Packet information
            nodes: Dictionary of all network nodes
            max_hops: Maximum number of hops

        Returns:
            Routing result
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

        path = [source]
        current = source
        visited = {source}
        total_delay = 0.0
        min_quality = 1.0

        dest_pos = nodes[destination].get('position', {})

        for hop in range(max_hops):
            if current == destination:
                return {
                    'success': True,
                    'path': path,
                    'total_delay': total_delay,
                    'hops': len(path) - 1,
                    'min_quality': min_quality,
                    'algorithm': 'Greedy'
                }

            current_node = nodes.get(current)
            if not current_node:
                break

            # Get neighbors
            neighbors = self._get_neighbors(current, nodes)

            # Filter out visited and non-operational neighbors
            valid_neighbors = [
                n for n in neighbors
                if n not in visited and nodes.get(n, {}).get('isOperational', True)
            ]

            if not valid_neighbors:
                break

            # Select neighbor closest to destination
            best_neighbor = None
            best_distance = float('inf')

            for neighbor_id in valid_neighbors:
                neighbor_pos = nodes[neighbor_id].get('position', {})
                distance = self._calculate_distance(neighbor_pos, dest_pos)

                if distance < best_distance:
                    best_distance = distance
                    best_neighbor = neighbor_id

            if best_neighbor is None:
                break

            # Update metrics
            total_delay += current_node.get('nodeProcessingDelayMs', 5.0)
            min_quality = min(min_quality, current_node.get('link_quality', 1.0))

            # Move to next hop
            current = best_neighbor
            path.append(current)
            visited.add(current)

        return {
            'success': False,
            'path': path,
            'total_delay': total_delay,
            'hops': len(path) - 1,
            'reason': 'Could not reach destination'
        }

    def _get_neighbors(self, node_id: str, nodes: Dict[str, Any]) -> List[str]:
        """Get neighbors for a node."""
        node_data = nodes.get(node_id)
        if not node_data:
            return []

        neighbors = node_data.get('neighbors', [])

        # If no neighbors, compute based on range
        if not neighbors:
            neighbors = self._compute_neighbors(node_id, nodes)

        return neighbors

    def _compute_neighbors(self, node_id: str, nodes: Dict[str, Any]) -> List[str]:
        """Compute neighbors based on communication range."""
        node_data = nodes.get(node_id)
        if not node_data:
            return []

        node_pos = node_data.get('position', {})
        node_type = node_data.get('nodeType', 'UNKNOWN')
        comm_range = self._get_communication_range(node_type)

        neighbors = []
        for other_id, other_data in nodes.items():
            if other_id == node_id:
                continue

            other_pos = other_data.get('position', {})
            distance = self._calculate_distance(node_pos, other_pos)

            if distance <= comm_range:
                neighbors.append(other_id)

        return neighbors

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate 3D Euclidean distance."""
        lat1, lon1, alt1 = pos1.get('lat', 0), pos1.get('lon', 0), pos1.get('alt', 0)
        lat2, lon2, alt2 = pos2.get('lat', 0), pos2.get('lon', 0), pos2.get('alt', 0)

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        earth_radius = 6371.0
        surface_distance = earth_radius * c

        altitude_diff = abs(alt2 - alt1)
        distance = np.sqrt(surface_distance**2 + altitude_diff**2)

        return distance

    def _get_communication_range(self, node_type: str) -> float:
        """Get communication range by node type."""
        ranges = {
            'GEO_SATELLITE': 40000.0,
            'MEO_SATELLITE': 10000.0,
            'LEO_SATELLITE': 3000.0,
            'UAV': 100.0,
            'GROUND_STATION': 5000.0,
        }
        return ranges.get(node_type, 1000.0)


class RandomRouter:
    """
    Random routing algorithm that randomly selects next hop
    from available neighbors.
    """

    def __init__(self, db_manager: MongoDBManager, seed: Optional[int] = None):
        """
        Initialize random router.

        Args:
            db_manager: Database manager
            seed: Random seed for reproducibility
        """
        self.db_manager = db_manager
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def route_packet(self, packet_data: Dict[str, Any], nodes: Dict[str, Any],
                    max_hops: int = 15) -> Dict[str, Any]:
        """
        Route a packet using random selection.

        Args:
            packet_data: Packet information
            nodes: Dictionary of all network nodes
            max_hops: Maximum number of hops

        Returns:
            Routing result
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

        path = [source]
        current = source
        visited = {source}
        total_delay = 0.0
        min_quality = 1.0

        for hop in range(max_hops):
            if current == destination:
                return {
                    'success': True,
                    'path': path,
                    'total_delay': total_delay,
                    'hops': len(path) - 1,
                    'min_quality': min_quality,
                    'algorithm': 'Random'
                }

            current_node = nodes.get(current)
            if not current_node:
                break

            # Get neighbors
            neighbors = current_node.get('neighbors', [])

            # Filter out visited and non-operational neighbors
            valid_neighbors = [
                n for n in neighbors
                if n not in visited and nodes.get(n, {}).get('isOperational', True)
            ]

            if not valid_neighbors:
                break

            # Randomly select next neighbor
            next_neighbor = random.choice(valid_neighbors)

            # Update metrics
            total_delay += current_node.get('nodeProcessingDelayMs', 5.0)
            min_quality = min(min_quality, current_node.get('link_quality', 1.0))

            # Move to next hop
            current = next_neighbor
            path.append(current)
            visited.add(current)

        return {
            'success': False,
            'path': path,
            'total_delay': total_delay,
            'hops': len(path) - 1,
            'reason': 'Could not reach destination'
        }


class QualityAwareRouter:
    """
    Quality-aware routing that prioritizes links with better quality.
    """

    def __init__(self, db_manager: MongoDBManager):
        """Initialize quality-aware router."""
        self.db_manager = db_manager

    def route_packet(self, packet_data: Dict[str, Any], nodes: Dict[str, Any],
                    max_hops: int = 15) -> Dict[str, Any]:
        """Route packet prioritizing link quality."""
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

        path = [source]
        current = source
        visited = {source}
        total_delay = 0.0
        min_quality = 1.0

        for hop in range(max_hops):
            if current == destination:
                return {
                    'success': True,
                    'path': path,
                    'total_delay': total_delay,
                    'hops': len(path) - 1,
                    'min_quality': min_quality,
                    'algorithm': 'QualityAware'
                }

            current_node = nodes.get(current)
            if not current_node:
                break

            neighbors = current_node.get('neighbors', [])
            valid_neighbors = [
                n for n in neighbors
                if n not in visited and nodes.get(n, {}).get('isOperational', True)
            ]

            if not valid_neighbors:
                break

            # Select neighbor with best link quality
            best_neighbor = max(
                valid_neighbors,
                key=lambda n: nodes[n].get('link_quality', 0.0)
            )

            # Update metrics
            total_delay += current_node.get('nodeProcessingDelayMs', 5.0)
            min_quality = min(min_quality, current_node.get('link_quality', 1.0))

            # Move to next hop
            current = best_neighbor
            path.append(current)
            visited.add(current)

        return {
            'success': False,
            'path': path,
            'total_delay': total_delay,
            'hops': len(path) - 1,
            'reason': 'Could not reach destination'
        }
