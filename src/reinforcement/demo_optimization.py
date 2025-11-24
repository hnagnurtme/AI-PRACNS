#!/usr/bin/env python3
"""
Demonstration script for RL optimization improvements.
Shows how dynamic neighbor updates and enhanced rewards work.
"""

import sys
import numpy as np
from typing import Dict, List


class SimpleNode:
    """Simplified node for demonstration."""
    def __init__(self, node_id: str, x: float, y: float, z: float, 
                 comm_range: float = 1000.0, utilization: float = 0.0):
        self.nodeId = node_id
        self.position = {'x': x, 'y': y, 'z': z}
        self.comm_range = comm_range
        self.resourceUtilization = utilization
        self.queue_occupancy = 0.0
        self.isOperational = True
        self.neighbors = []
    
    def distance_to(self, other: 'SimpleNode') -> float:
        """Calculate Euclidean distance to another node."""
        dx = self.position['x'] - other.position['x']
        dy = self.position['y'] - other.position['y']
        dz = self.position['z'] - other.position['z']
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def move(self, dx: float, dy: float, dz: float):
        """Move node by delta."""
        self.position['x'] += dx
        self.position['y'] += dy
        self.position['z'] += dz


class SimpleMobilityManager:
    """Simplified mobility manager for demonstration."""
    def __init__(self):
        self.nodes: Dict[str, SimpleNode] = {}
    
    def add_node(self, node: SimpleNode):
        """Add a node to the network."""
        self.nodes[node.nodeId] = node
    
    def compute_neighbors(self, node_id: str) -> List[str]:
        """Compute neighbors based on current positions."""
        node = self.nodes.get(node_id)
        if not node or not node.isOperational:
            return []
        
        neighbors = []
        for other_id, other_node in self.nodes.items():
            if other_id == node_id or not other_node.isOperational:
                continue
            
            distance = node.distance_to(other_node)
            if distance <= node.comm_range:
                neighbors.append(other_id)
        
        return neighbors
    
    def update_all_neighbors(self):
        """Update neighbors for all nodes (CRITICAL for fair comparison)."""
        for node_id in self.nodes:
            self.nodes[node_id].neighbors = self.compute_neighbors(node_id)
    
    def print_network_state(self):
        """Print current network state."""
        print("\n" + "="*80)
        print("Network State")
        print("="*80)
        for node_id, node in self.nodes.items():
            pos = node.position
            neighbors = node.neighbors
            util = node.resourceUtilization
            print(f"{node_id:8s} | Pos: ({pos['x']:6.1f}, {pos['y']:6.1f}, {pos['z']:6.1f}) | "
                  f"Util: {util:.2f} | Neighbors: {len(neighbors)} {neighbors}")


def demonstrate_dynamic_neighbors():
    """Demonstrate dynamic neighbor updates."""
    print("\n" + "="*80)
    print("DEMONSTRATION 1: Dynamic Neighbor Updates")
    print("="*80)
    
    manager = SimpleMobilityManager()
    
    # Create nodes
    node_a = SimpleNode("NODE_A", 0, 0, 0, comm_range=500)
    node_b = SimpleNode("NODE_B", 100, 100, 0, comm_range=500)
    node_c = SimpleNode("NODE_C", 1000, 1000, 0, comm_range=500)
    
    manager.add_node(node_a)
    manager.add_node(node_b)
    manager.add_node(node_c)
    
    print("\n--- Initial Configuration ---")
    manager.update_all_neighbors()
    manager.print_network_state()
    
    print("\n--- After NODE_C moves closer to NODE_A ---")
    node_c.move(-900, -900, 0)  # Move C close to A
    manager.update_all_neighbors()
    manager.print_network_state()
    
    print("\n✓ KEY INSIGHT: Neighbors updated dynamically when nodes move!")
    print("✓ This ensures fair comparison between RL and baseline algorithms.")


def demonstrate_congestion_awareness():
    """Demonstrate congestion-aware routing decisions."""
    print("\n" + "="*80)
    print("DEMONSTRATION 2: Congestion-Aware Routing")
    print("="*80)
    
    manager = SimpleMobilityManager()
    
    # Create nodes with different utilization levels
    current = SimpleNode("CURRENT", 0, 0, 0, comm_range=500, utilization=0.3)
    neighbor_1 = SimpleNode("NEIGHBOR_1", 100, 0, 0, comm_range=500, utilization=0.9)  # Congested!
    neighbor_2 = SimpleNode("NEIGHBOR_2", 0, 100, 0, comm_range=500, utilization=0.2)  # Underutilized
    
    manager.add_node(current)
    manager.add_node(neighbor_1)
    manager.add_node(neighbor_2)
    manager.update_all_neighbors()
    
    print("\n--- Network Configuration ---")
    manager.print_network_state()
    
    print("\n--- Routing Decision Analysis ---")
    print(f"NEIGHBOR_1: Utilization = {neighbor_1.resourceUtilization:.2f} (CONGESTED)")
    print(f"NEIGHBOR_2: Utilization = {neighbor_2.resourceUtilization:.2f} (AVAILABLE)")
    
    # Simulate reward calculation
    weights = {
        'congestion_penalty': 100.0,
        'load_balance_reward': 20.0,
    }
    
    # For NEIGHBOR_1 (congested)
    congestion_1 = neighbor_1.resourceUtilization
    if congestion_1 > 0.8:
        penalty_1 = -weights['congestion_penalty']
    else:
        penalty_1 = weights['load_balance_reward'] * (1.0 - congestion_1)
    
    # For NEIGHBOR_2 (underutilized)
    congestion_2 = neighbor_2.resourceUtilization
    penalty_2 = weights['load_balance_reward'] * (1.0 - congestion_2)
    
    print(f"\nReward adjustment for selecting NEIGHBOR_1: {penalty_1:+.1f}")
    print(f"Reward adjustment for selecting NEIGHBOR_2: {penalty_2:+.1f}")
    print(f"\n✓ RL learns to prefer NEIGHBOR_2 (reward difference: {penalty_2 - penalty_1:+.1f})")
    print("✓ This achieves proactive congestion avoidance and load balancing!")


def demonstrate_network_metrics():
    """Demonstrate network-wide metrics calculation."""
    print("\n" + "="*80)
    print("DEMONSTRATION 3: Network-Wide Metrics")
    print("="*80)
    
    # Scenario 1: Balanced network
    print("\n--- Scenario 1: Well-Balanced Network ---")
    balanced_utils = [0.5, 0.5, 0.5, 0.5, 0.5]
    avg_util = np.mean(balanced_utils)
    variance = np.var(balanced_utils)
    print(f"Node utilizations: {balanced_utils}")
    print(f"Average utilization: {avg_util:.2f}")
    print(f"Utilization variance: {variance:.4f} (LOW = GOOD)")
    
    # Scenario 2: Imbalanced network
    print("\n--- Scenario 2: Imbalanced Network ---")
    imbalanced_utils = [0.1, 0.9, 0.1, 0.9, 0.1]
    avg_util = np.mean(imbalanced_utils)
    variance = np.var(imbalanced_utils)
    print(f"Node utilizations: {imbalanced_utils}")
    print(f"Average utilization: {avg_util:.2f}")
    print(f"Utilization variance: {variance:.4f} (HIGH = BAD)")
    
    print(f"\n✓ Variance is {variance / 0.0000:.1f}x higher in imbalanced network!")
    print("✓ RL optimizes for low variance to achieve fair load distribution.")


def demonstrate_fair_comparison():
    """Demonstrate fair comparison between RL and baseline."""
    print("\n" + "="*80)
    print("DEMONSTRATION 4: Fair RL vs Baseline Comparison")
    print("="*80)
    
    print("\n--- BEFORE Fix ---")
    print("Baseline (Dijkstra): Uses static node.neighbors")
    print("RL: Uses dynamic MobilityManager.get_current_neighbors()")
    print("❌ UNFAIR: RL has advantage when nodes move")
    
    print("\n--- AFTER Fix ---")
    print("Both algorithms: Use node.neighbors updated by MobilityManager")
    print("MobilityManager.update_node_neighbors() called in step_dynamics()")
    print("✅ FAIR: Both see same current network topology")
    
    print("\n--- Example Scenario ---")
    manager = SimpleMobilityManager()
    
    node_a = SimpleNode("NODE_A", 0, 0, 0)
    node_b = SimpleNode("NODE_B", 100, 0, 0)
    
    manager.add_node(node_a)
    manager.add_node(node_b)
    
    print("\nStep 1: Initial state")
    manager.update_all_neighbors()
    print(f"  NODE_A neighbors: {node_a.neighbors}")
    
    print("\nStep 2: NODE_B moves away")
    node_b.move(2000, 0, 0)
    manager.update_all_neighbors()
    print(f"  NODE_A neighbors: {node_a.neighbors} (updated!)")
    
    print("\n✓ Both RL and Baseline see NODE_B is no longer a neighbor")
    print("✓ Fair comparison ensures RL wins based on learning, not data advantage")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("RL Optimization Demonstration")
    print("Showing: Dynamic Neighbors, Congestion Avoidance, Fair Comparison")
    print("="*80)
    
    demonstrate_dynamic_neighbors()
    demonstrate_congestion_awareness()
    demonstrate_network_metrics()
    demonstrate_fair_comparison()
    
    print("\n" + "="*80)
    print("Summary of Optimizations")
    print("="*80)
    print("✅ Dynamic neighbor updates ensure fair RL vs baseline comparison")
    print("✅ Enhanced reward function promotes congestion avoidance")
    print("✅ Load balancing incentives improve network fairness")
    print("✅ Network-wide metrics track resource utilization")
    print("✅ RL should consistently outperform baselines now!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
