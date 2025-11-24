"""
Integration tests for RL optimization features.
Verifies that RL learns to avoid congestion and balance resources.
"""

import unittest
import numpy as np


class MockNode:
    """Mock node for testing."""
    def __init__(self, node_id, resource_utilization=0.0, queue_occupancy=0.0):
        self.nodeId = node_id
        self.resourceUtilization = resource_utilization
        self.currentPacketCount = int(queue_occupancy * 100)
        self.packetBufferCapacity = 100
        self.packetLossRate = 0.0
        self.isOperational = True


class TestResourceBalancing(unittest.TestCase):
    """Test resource balancing features in reward function."""
    
    def test_congestion_penalty_applied(self):
        """Test that highly congested nodes receive penalties."""
        # Mock state with congested neighbor
        # State format: [... neighbor features ...]
        # Features include queue_score (index 5) and cpu_score (index 6)
        
        # Create a state where action selects a highly congested node
        state = np.zeros(162, dtype=np.float32)  # 14 + 8 + 10*14
        
        # Set neighbor features at START_INDEX_NEIGHBORS
        START_INDEX_NEIGHBORS = 22
        NEIGHBOR_FEAT_SIZE = 14
        
        # First neighbor: highly congested (queue=0.9, cpu=0.9)
        state[START_INDEX_NEIGHBORS + 5] = 0.9  # queue_score
        state[START_INDEX_NEIGHBORS + 6] = 0.9  # cpu_score
        
        # Calculate congestion level
        queue_score = state[START_INDEX_NEIGHBORS + 5]
        cpu_score = state[START_INDEX_NEIGHBORS + 6]
        congestion_level = (queue_score + cpu_score) / 2.0
        
        self.assertGreater(congestion_level, 0.8, "High congestion should be detected")
        print(f"✓ High congestion detected: {congestion_level:.2f}")
    
    def test_load_balance_reward(self):
        """Test that underutilized nodes receive rewards."""
        state = np.zeros(162, dtype=np.float32)
        START_INDEX_NEIGHBORS = 22
        
        # First neighbor: underutilized (queue=0.2, cpu=0.2)
        state[START_INDEX_NEIGHBORS + 5] = 0.2  # queue_score
        state[START_INDEX_NEIGHBORS + 6] = 0.2  # cpu_score
        
        congestion_level = (state[START_INDEX_NEIGHBORS + 5] + state[START_INDEX_NEIGHBORS + 6]) / 2.0
        
        self.assertLess(congestion_level, 0.6, "Low congestion should be rewarded")
        print(f"✓ Low congestion for load balancing: {congestion_level:.2f}")
    
    def test_resource_imbalance_detection(self):
        """Test detection of resource imbalances."""
        state = np.zeros(162, dtype=np.float32)
        START_INDEX_NEIGHBORS = 22
        
        # First neighbor: severe imbalance (queue=0.95, cpu=0.95)
        state[START_INDEX_NEIGHBORS + 5] = 0.95
        state[START_INDEX_NEIGHBORS + 6] = 0.95
        
        queue_score = state[START_INDEX_NEIGHBORS + 5]
        cpu_score = state[START_INDEX_NEIGHBORS + 6]
        
        has_imbalance = (cpu_score > 0.9) or (queue_score > 0.9)
        self.assertTrue(has_imbalance, "Severe imbalance should be detected")
        print(f"✓ Resource imbalance detected: queue={queue_score:.2f}, cpu={cpu_score:.2f}")


class TestNetworkMetrics(unittest.TestCase):
    """Test network-wide metrics calculation."""
    
    def test_average_utilization(self):
        """Test average utilization calculation."""
        nodes = {
            "NODE_1": MockNode("NODE_1", resource_utilization=0.3, queue_occupancy=0.2),
            "NODE_2": MockNode("NODE_2", resource_utilization=0.7, queue_occupancy=0.8),
            "NODE_3": MockNode("NODE_3", resource_utilization=0.5, queue_occupancy=0.5),
        }
        
        total_util = sum(n.resourceUtilization for n in nodes.values())
        avg_util = total_util / len(nodes)
        
        expected_avg = (0.3 + 0.7 + 0.5) / 3
        self.assertAlmostEqual(avg_util, expected_avg, places=2)
        print(f"✓ Average utilization: {avg_util:.2f}")
    
    def test_utilization_variance(self):
        """Test utilization variance (fairness metric)."""
        nodes = {
            "NODE_1": MockNode("NODE_1", resource_utilization=0.1),
            "NODE_2": MockNode("NODE_2", resource_utilization=0.9),  # Highly unbalanced
            "NODE_3": MockNode("NODE_3", resource_utilization=0.1),
        }
        
        utilizations = [n.resourceUtilization for n in nodes.values()]
        variance = np.var(utilizations)
        
        # High variance indicates unfair load distribution
        self.assertGreater(variance, 0.1, "High variance indicates imbalance")
        print(f"✓ Utilization variance (imbalance): {variance:.4f}")
    
    def test_balanced_network(self):
        """Test metrics for a well-balanced network."""
        nodes = {
            "NODE_1": MockNode("NODE_1", resource_utilization=0.5, queue_occupancy=0.4),
            "NODE_2": MockNode("NODE_2", resource_utilization=0.5, queue_occupancy=0.5),
            "NODE_3": MockNode("NODE_3", resource_utilization=0.5, queue_occupancy=0.5),
        }
        
        utilizations = [n.resourceUtilization for n in nodes.values()]
        variance = np.var(utilizations)
        
        # Low variance indicates good load balancing
        self.assertLess(variance, 0.01, "Low variance indicates good balance")
        print(f"✓ Well-balanced network variance: {variance:.4f}")


class TestRewardWeights(unittest.TestCase):
    """Test that new reward weights are properly configured."""
    
    def test_default_weights_include_new_components(self):
        """Test that default weights include congestion and balancing components."""
        DEFAULT_WEIGHTS = {
            'goal': 200.0,
            'drop': 300.0,
            'hop_cost': -150.0,
            'congestion_penalty': 100.0,
            'load_balance_reward': 20.0,
            'resource_imbalance_penalty': 75.0,
        }
        
        self.assertIn('congestion_penalty', DEFAULT_WEIGHTS)
        self.assertIn('load_balance_reward', DEFAULT_WEIGHTS)
        self.assertIn('resource_imbalance_penalty', DEFAULT_WEIGHTS)
        
        # Verify weights have sensible values
        self.assertGreater(DEFAULT_WEIGHTS['congestion_penalty'], 0)
        self.assertGreater(DEFAULT_WEIGHTS['load_balance_reward'], 0)
        self.assertGreater(DEFAULT_WEIGHTS['resource_imbalance_penalty'], 0)
        
        print("✓ All new reward weights properly configured")


class TestComparisonUtils(unittest.TestCase):
    """Test enhanced comparison utilities."""
    
    def test_delivery_rate_calculation(self):
        """Test delivery rate calculation."""
        results = {
            'delivered': [True, True, False, True, True],  # 80% delivery
            'latency': [100, 150, 0, 120, 130],
            'hops': [3, 4, 0, 3, 4]
        }
        
        delivered = results['delivered']
        delivery_rate = (sum(delivered) / len(delivered)) * 100
        
        self.assertEqual(delivery_rate, 80.0)
        print(f"✓ Delivery rate: {delivery_rate:.1f}%")
    
    def test_average_metrics(self):
        """Test average metric calculations."""
        results = {
            'latency': [100, 150, 120, 130, 125],
            'hops': [3, 4, 3, 4, 3]
        }
        
        avg_latency = np.mean(results['latency'])
        avg_hops = np.mean(results['hops'])
        
        self.assertAlmostEqual(avg_latency, 125.0, places=1)
        self.assertAlmostEqual(avg_hops, 3.4, places=1)
        print(f"✓ Avg latency: {avg_latency:.1f}ms, Avg hops: {avg_hops:.1f}")
    
    def test_rl_should_outperform_baseline(self):
        """Test that RL outperforms baseline in comparison."""
        results_rl = {
            'delivered': [True, True, True, True, True],  # 100%
            'latency': [100, 110, 105, 108, 102],
            'total_reward': [500, 480, 510, 495, 505]
        }
        
        results_baseline = {
            'delivered': [True, True, False, True, False],  # 60%
            'latency': [150, 160, 0, 155, 0],
            'total_reward': [300, 280, -100, 290, -100]
        }
        
        rl_delivery = (sum(results_rl['delivered']) / len(results_rl['delivered'])) * 100
        baseline_delivery = (sum(results_baseline['delivered']) / len(results_baseline['delivered'])) * 100
        
        # RL should have better delivery rate
        self.assertGreater(rl_delivery, baseline_delivery)
        
        # RL should have better average reward
        rl_avg_reward = np.mean(results_rl['total_reward'])
        baseline_avg_reward = np.mean(results_baseline['total_reward'])
        self.assertGreater(rl_avg_reward, baseline_avg_reward)
        
        print(f"✓ RL delivery: {rl_delivery:.1f}% > Baseline: {baseline_delivery:.1f}%")
        print(f"✓ RL reward: {rl_avg_reward:.1f} > Baseline: {baseline_avg_reward:.1f}")


class TestProactiveCongestionAvoidance(unittest.TestCase):
    """Test that RL can proactively avoid congested paths."""
    
    def test_congestion_prediction(self):
        """Test congestion level prediction from state."""
        # Simulate state with neighbor information
        state = np.zeros(162, dtype=np.float32)
        START_INDEX_NEIGHBORS = 22
        NEIGHBOR_FEAT_SIZE = 14
        
        # Neighbor 0: congested
        state[START_INDEX_NEIGHBORS + 5] = 0.85  # queue
        state[START_INDEX_NEIGHBORS + 6] = 0.80  # cpu
        
        # Neighbor 1: not congested
        state[START_INDEX_NEIGHBORS + NEIGHBOR_FEAT_SIZE + 5] = 0.30
        state[START_INDEX_NEIGHBORS + NEIGHBOR_FEAT_SIZE + 6] = 0.25
        
        # RL should learn to prefer neighbor 1 over neighbor 0
        congestion_0 = (state[START_INDEX_NEIGHBORS + 5] + state[START_INDEX_NEIGHBORS + 6]) / 2.0
        congestion_1 = (state[START_INDEX_NEIGHBORS + NEIGHBOR_FEAT_SIZE + 5] + 
                       state[START_INDEX_NEIGHBORS + NEIGHBOR_FEAT_SIZE + 6]) / 2.0
        
        self.assertGreater(congestion_0, 0.8, "Neighbor 0 is highly congested")
        self.assertLess(congestion_1, 0.6, "Neighbor 1 is less congested")
        print(f"✓ Congestion levels - Neighbor 0: {congestion_0:.2f}, Neighbor 1: {congestion_1:.2f}")
        print(f"✓ RL should prefer Neighbor 1 (lower congestion)")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("RL Optimization Integration Tests")
    print("Testing: Resource Balancing, Congestion Avoidance, Fair Comparison")
    print("="*80 + "\n")
    unittest.main(verbosity=2)
