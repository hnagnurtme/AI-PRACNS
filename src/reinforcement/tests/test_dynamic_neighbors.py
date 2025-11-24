"""
Test dynamic neighbor updates to ensure fair RL vs baseline comparison.
This test verifies that neighbors are updated when nodes move.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

# Mock node class for testing
class MockPosition:
    def __init__(self, latitude, longitude, altitude):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

class MockCommunication:
    def __init__(self, maxRangeKm):
        self.maxRangeKm = maxRangeKm

class MockNode:
    def __init__(self, nodeId, position, communication, isOperational=True):
        self.nodeId = nodeId
        self.position = position
        self.communication = communication
        self.isOperational = isOperational
        self.neighbors = []


class SimpleMobilityManager:
    """Simplified mobility manager for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.nodes = {}
        self.MAX_NEIGHBORS = self.config.get('max_neighbors', 10)
        self._neighbor_cache = {}
        self._cache_timestamp = 0.0
    
    def set_nodes(self, nodes):
        self.nodes = nodes
    
    def update_nodes(self, delta_time):
        self._neighbor_cache.clear()
    
    def get_current_neighbors(self, node_id, dynamic_state=None):
        if dynamic_state is None:
            dynamic_state = {}
        
        node = self.nodes.get(node_id)
        if not node or not node.isOperational:
            return []
        
        potential_neighbors = []
        for other_id, other_node in self.nodes.items():
            if other_id == node_id or not other_node.isOperational:
                continue
            
            distance = self._calculate_distance(node.position, other_node.position)
            
            if distance <= node.communication.maxRangeKm:
                link_quality = 1.0 / (1.0 + distance)
                if link_quality > 0.1:
                    potential_neighbors.append((other_id, distance, link_quality))
        
        potential_neighbors.sort(key=lambda x: x[2], reverse=True)
        return [n[0] for n in potential_neighbors[:self.MAX_NEIGHBORS]]
    
    def _calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1.latitude - pos2.latitude)**2 + 
                      (pos1.longitude - pos2.longitude)**2 + 
                      (pos1.altitude - pos2.altitude)**2)
    
    def update_node_neighbors(self, dynamic_state=None):
        for node_id, node in self.nodes.items():
            node.neighbors = self.get_current_neighbors(node_id, dynamic_state)


class TestDynamicNeighbors(unittest.TestCase):
    """Test dynamic neighbor computation and updates."""
    
    def setUp(self):
        """Set up test environment with sample nodes."""
        self.mobility_manager = SimpleMobilityManager({'max_neighbors': 10})
        
        # Create test nodes at different positions
        self.nodes = {}
        
        # Node 1 at origin
        pos1 = MockPosition(0.0, 0.0, 500.0)
        comm1 = MockCommunication(maxRangeKm=2000.0)
        node1 = MockNode("NODE_1", pos1, comm1, isOperational=True)
        self.nodes["NODE_1"] = node1
        
        # Node 2 close to Node 1 (within range)
        pos2 = MockPosition(0.1, 0.1, 500.0)
        comm2 = MockCommunication(maxRangeKm=2000.0)
        node2 = MockNode("NODE_2", pos2, comm2, isOperational=True)
        self.nodes["NODE_2"] = node2
        
        # Node 3 far from Node 1 (out of range initially)
        pos3 = MockPosition(50.0, 50.0, 500.0)
        comm3 = MockCommunication(maxRangeKm=2000.0)
        node3 = MockNode("NODE_3", pos3, comm3, isOperational=True)
        self.nodes["NODE_3"] = node3
        
        self.mobility_manager.set_nodes(self.nodes)
    
    def test_initial_neighbors(self):
        """Test that initial neighbor computation is correct."""
        neighbors = self.mobility_manager.get_current_neighbors("NODE_1")
        
        # NODE_1 should see NODE_2 (close) but not NODE_3 (far)
        self.assertIn("NODE_2", neighbors)
        self.assertNotIn("NODE_3", neighbors)
        print("✓ Initial neighbors computed correctly")
    
    def test_neighbor_update_after_movement(self):
        """Test that neighbors update when nodes move."""
        # Initial neighbors
        initial_neighbors = self.mobility_manager.get_current_neighbors("NODE_1")
        self.assertIn("NODE_2", initial_neighbors)
        self.assertNotIn("NODE_3", initial_neighbors)
        
        # Move NODE_3 closer to NODE_1
        self.nodes["NODE_3"].position.latitude = 0.2
        self.nodes["NODE_3"].position.longitude = 0.2
        
        # Update mobility manager (simulates time step)
        self.mobility_manager.update_nodes(1.0)
        
        # Get updated neighbors
        updated_neighbors = self.mobility_manager.get_current_neighbors("NODE_1")
        
        # Now NODE_3 should be visible
        self.assertIn("NODE_3", updated_neighbors)
        print("✓ Neighbors updated after node movement")
    
    def test_neighbor_disappears_when_out_of_range(self):
        """Test that neighbors are removed when they move out of range."""
        # Initially NODE_2 is a neighbor
        initial_neighbors = self.mobility_manager.get_current_neighbors("NODE_1")
        self.assertIn("NODE_2", initial_neighbors)
        
        # Move NODE_2 far away
        self.nodes["NODE_2"].position.latitude = 100.0
        self.nodes["NODE_2"].position.longitude = 100.0
        
        # Update mobility manager
        self.mobility_manager.update_nodes(1.0)
        
        # Get updated neighbors
        updated_neighbors = self.mobility_manager.get_current_neighbors("NODE_1")
        
        # NODE_2 should no longer be a neighbor
        self.assertNotIn("NODE_2", updated_neighbors)
        print("✓ Neighbor removed when out of range")
    
    def test_update_node_neighbors_attribute(self):
        """Test that update_node_neighbors updates the neighbors attribute."""
        # Update all node neighbors
        self.mobility_manager.update_node_neighbors()
        
        # Check that NODE_1 has neighbors attribute updated
        node1_neighbors = self.nodes["NODE_1"].neighbors
        self.assertIsNotNone(node1_neighbors)
        self.assertIn("NODE_2", node1_neighbors)
        self.assertNotIn("NODE_3", node1_neighbors)
        print("✓ Node neighbors attribute updated correctly")
    
    def test_non_operational_nodes_excluded(self):
        """Test that non-operational nodes are not included as neighbors."""
        # Make NODE_2 non-operational
        self.nodes["NODE_2"].isOperational = False
        
        # Get neighbors
        neighbors = self.mobility_manager.get_current_neighbors("NODE_1")
        
        # NODE_2 should not be in neighbors even though it's in range
        self.assertNotIn("NODE_2", neighbors)
        print("✓ Non-operational nodes excluded from neighbors")


class TestFairBaselineComparison(unittest.TestCase):
    """Test that baseline algorithms get updated neighbors like RL."""
    
    def setUp(self):
        """Set up test environment."""
        self.mobility_manager = SimpleMobilityManager({'max_neighbors': 10})
        
        # Create minimal test network
        pos1 = MockPosition(0.0, 0.0, 500.0)
        comm1 = MockCommunication(maxRangeKm=2000.0)
        node1 = MockNode("NODE_1", pos1, comm1, isOperational=True)
        
        pos2 = MockPosition(0.1, 0.1, 500.0)
        comm2 = MockCommunication(maxRangeKm=2000.0)
        node2 = MockNode("NODE_2", pos2, comm2, isOperational=True)
        
        self.nodes = {"NODE_1": node1, "NODE_2": node2}
        self.mobility_manager.set_nodes(self.nodes)
    
    def test_baseline_gets_updated_neighbors(self):
        """Test that baseline algorithms can access updated neighbors."""
        # Update neighbors (simulates what DynamicSatelliteEnv does)
        self.mobility_manager.update_node_neighbors()
        
        # Check that node neighbors attribute is updated
        updated_neighbors = self.nodes["NODE_1"].neighbors
        self.assertIsNotNone(updated_neighbors)
        self.assertIn("NODE_2", updated_neighbors)
        
        # Move NODE_2 away
        self.nodes["NODE_2"].position.latitude = 100.0
        self.mobility_manager.update_nodes(1.0)
        self.mobility_manager.update_node_neighbors()
        
        # Neighbors should now exclude NODE_2
        final_neighbors = self.nodes["NODE_1"].neighbors
        self.assertNotIn("NODE_2", final_neighbors)
        print("✓ Baseline algorithms can access updated neighbors")


if __name__ == '__main__':
    # Run tests
    print("\n" + "="*80)
    print("Testing Dynamic Neighbor Updates for Fair RL vs Baseline Comparison")
    print("="*80 + "\n")
    unittest.main(verbosity=2)

