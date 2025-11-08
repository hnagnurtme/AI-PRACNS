"""
Unit tests for the metrics tracking system.
"""

import unittest
import time
from python.utils.metrics_tracker import HopMetrics, RouteMetrics, MetricsComparator


class TestHopMetrics(unittest.TestCase):
    """Test HopMetrics class."""
    
    def test_hop_metrics_creation(self):
        """Test creating a hop metric."""
        hop = HopMetrics(
            from_node_id="NODE-A",
            to_node_id="NODE-B",
            timestamp_ms=time.time() * 1000,
            latency_ms=10.5,
            node_cpu_utilization=0.5,
            node_memory_utilization=0.6,
            node_bandwidth_utilization=0.4
        )
        
        self.assertEqual(hop.from_node_id, "NODE-A")
        self.assertEqual(hop.to_node_id, "NODE-B")
        self.assertEqual(hop.latency_ms, 10.5)
        self.assertEqual(hop.node_cpu_utilization, 0.5)
    
    def test_hop_metrics_to_dict(self):
        """Test converting hop metrics to dictionary."""
        hop = HopMetrics(
            from_node_id="NODE-A",
            to_node_id="NODE-B",
            timestamp_ms=123456789.0,
            latency_ms=10.5
        )
        
        hop_dict = hop.to_dict()
        self.assertIsInstance(hop_dict, dict)
        self.assertEqual(hop_dict['fromNodeId'], "NODE-A")
        self.assertEqual(hop_dict['toNodeId'], "NODE-B")
        self.assertEqual(hop_dict['latencyMs'], 10.5)


class TestRouteMetrics(unittest.TestCase):
    """Test RouteMetrics class."""
    
    def test_route_metrics_creation(self):
        """Test creating route metrics."""
        route = RouteMetrics(
            packet_id="PKT-001",
            source_id="NODE-A",
            dest_id="NODE-Z",
            algorithm="RL-DQN",
            success=False
        )
        
        self.assertEqual(route.packet_id, "PKT-001")
        self.assertEqual(route.algorithm, "RL-DQN")
        self.assertFalse(route.success)
        self.assertEqual(route.total_hops, 0)
    
    def test_add_hop(self):
        """Test adding hop records."""
        route = RouteMetrics(
            packet_id="PKT-001",
            source_id="NODE-A",
            dest_id="NODE-Z",
            algorithm="RL-DQN",
            success=False
        )
        
        hop1 = HopMetrics(
            from_node_id="NODE-A",
            to_node_id="NODE-B",
            timestamp_ms=time.time() * 1000,
            latency_ms=10.0
        )
        
        hop2 = HopMetrics(
            from_node_id="NODE-B",
            to_node_id="NODE-C",
            timestamp_ms=time.time() * 1000,
            latency_ms=15.0
        )
        
        route.add_hop(hop1)
        route.add_hop(hop2)
        
        self.assertEqual(route.total_hops, 2)
        self.assertEqual(route.total_latency_ms, 25.0)
        self.assertEqual(len(route.hop_records), 2)
    
    def test_finalize(self):
        """Test finalizing a route."""
        route = RouteMetrics(
            packet_id="PKT-001",
            source_id="NODE-A",
            dest_id="NODE-Z",
            algorithm="RL-DQN",
            success=False
        )
        
        route.finalize(success=True)
        
        self.assertTrue(route.success)
        self.assertTrue(route.packet_delivered)
        self.assertGreater(route.end_time_ms, 0)
    
    def test_get_average_node_utilization(self):
        """Test calculating average node utilization."""
        route = RouteMetrics(
            packet_id="PKT-001",
            source_id="NODE-A",
            dest_id="NODE-Z",
            algorithm="RL-DQN",
            success=False
        )
        
        hop1 = HopMetrics(
            from_node_id="NODE-A",
            to_node_id="NODE-B",
            timestamp_ms=time.time() * 1000,
            latency_ms=10.0,
            node_cpu_utilization=0.5,
            node_memory_utilization=0.6,
            node_bandwidth_utilization=0.4
        )
        
        hop2 = HopMetrics(
            from_node_id="NODE-B",
            to_node_id="NODE-C",
            timestamp_ms=time.time() * 1000,
            latency_ms=15.0,
            node_cpu_utilization=0.3,
            node_memory_utilization=0.4,
            node_bandwidth_utilization=0.2
        )
        
        route.add_hop(hop1)
        route.add_hop(hop2)
        
        avg_util = route.get_average_node_utilization()
        # hop1: (0.5 + 0.6 + 0.4) / 3 = 0.5
        # hop2: (0.3 + 0.4 + 0.2) / 3 = 0.3
        # average: (0.5 + 0.3) / 2 = 0.4
        self.assertAlmostEqual(avg_util, 0.4, places=5)


class TestMetricsComparator(unittest.TestCase):
    """Test MetricsComparator class."""
    
    def test_comparator_creation(self):
        """Test creating a comparator."""
        comparator = MetricsComparator()
        self.assertEqual(len(comparator.rl_metrics), 0)
        self.assertEqual(len(comparator.dijkstra_metrics), 0)
    
    def test_add_metrics(self):
        """Test adding metrics to comparator."""
        comparator = MetricsComparator()
        
        rl_route = RouteMetrics(
            packet_id="PKT-001",
            source_id="NODE-A",
            dest_id="NODE-Z",
            algorithm="RL-DQN",
            success=True
        )
        rl_route.add_hop(HopMetrics(
            from_node_id="A",
            to_node_id="B",
            timestamp_ms=time.time() * 1000,
            latency_ms=10.0
        ))
        
        dijkstra_route = RouteMetrics(
            packet_id="PKT-001",
            source_id="NODE-A",
            dest_id="NODE-Z",
            algorithm="DIJKSTRA",
            success=True
        )
        dijkstra_route.add_hop(HopMetrics(
            from_node_id="A",
            to_node_id="B",
            timestamp_ms=time.time() * 1000,
            latency_ms=12.0
        ))
        
        comparator.add_rl_metric(rl_route)
        comparator.add_dijkstra_metric(dijkstra_route)
        
        self.assertEqual(len(comparator.rl_metrics), 1)
        self.assertEqual(len(comparator.dijkstra_metrics), 1)
    
    def test_calculate_improvement(self):
        """Test improvement calculation."""
        comparator = MetricsComparator()
        
        # Lower is better (latency)
        improvement = comparator._calculate_improvement(100, 80, higher_is_better=False)
        self.assertEqual(improvement, 20.0)  # 20% improvement
        
        # Higher is better (delivery rate)
        improvement = comparator._calculate_improvement(80, 90, higher_is_better=True)
        self.assertEqual(improvement, 12.5)  # 12.5% improvement
    
    def test_comparison_with_data(self):
        """Test comparison summary with actual data."""
        comparator = MetricsComparator()
        
        # Create RL metrics (better performance)
        for i in range(5):
            route = RouteMetrics(
                packet_id=f"PKT-{i}",
                source_id="A",
                dest_id="Z",
                algorithm="RL-DQN",
                success=True
            )
            route.add_hop(HopMetrics(
                from_node_id="A",
                to_node_id="B",
                timestamp_ms=time.time() * 1000,
                latency_ms=10.0 + i
            ))
            comparator.add_rl_metric(route)
        
        # Create Dijkstra metrics (worse performance)
        for i in range(5):
            route = RouteMetrics(
                packet_id=f"PKT-{i}",
                source_id="A",
                dest_id="Z",
                algorithm="DIJKSTRA",
                success=True
            )
            route.add_hop(HopMetrics(
                from_node_id="A",
                to_node_id="B",
                timestamp_ms=time.time() * 1000,
                latency_ms=15.0 + i
            ))
            comparator.add_dijkstra_metric(route)
        
        summary = comparator.get_comparison_summary()
        
        self.assertIn('rl', summary)
        self.assertIn('dijkstra', summary)
        self.assertIn('comparison', summary)
        
        # RL should show lower latency
        self.assertLess(
            summary['rl']['avg_latency_ms'],
            summary['dijkstra']['avg_latency_ms']
        )
        
        # Should show positive improvement
        self.assertGreater(
            summary['comparison']['latency_improvement_percent'],
            0
        )


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
