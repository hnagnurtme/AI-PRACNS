# python/utils/metrics_tracker.py

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class HopMetrics:
    """Detailed metrics for a single hop in the routing path."""
    from_node_id: str
    to_node_id: str
    timestamp_ms: float
    latency_ms: float
    node_cpu_utilization: float = 0.0
    node_memory_utilization: float = 0.0
    node_bandwidth_utilization: float = 0.0
    node_packet_count: int = 0
    node_buffer_capacity: int = 0
    distance_km: float = 0.0
    fspl_db: float = 0.0
    packet_loss_rate: float = 0.0
    is_operational: bool = True
    drop_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'fromNodeId': self.from_node_id,
            'toNodeId': self.to_node_id,
            'timestampMs': self.timestamp_ms,
            'latencyMs': self.latency_ms,
            'nodeCpuUtilization': self.node_cpu_utilization,
            'nodeMemoryUtilization': self.node_memory_utilization,
            'nodeBandwidthUtilization': self.node_bandwidth_utilization,
            'nodePacketCount': self.node_packet_count,
            'nodeBufferCapacity': self.node_buffer_capacity,
            'distanceKm': self.distance_km,
            'fsplDb': self.fspl_db,
            'packetLossRate': self.packet_loss_rate,
            'isOperational': self.is_operational,
            'dropReason': self.drop_reason
        }


@dataclass
class RouteMetrics:
    """Complete metrics for an entire routing path."""
    packet_id: str
    source_id: str
    dest_id: str
    algorithm: str  # "RL-DQN" or "DIJKSTRA"
    success: bool
    hop_records: List[HopMetrics] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_hops: int = 0
    packet_delivered: bool = False
    drop_reason: Optional[str] = None
    start_time_ms: float = field(default_factory=lambda: time.time() * 1000)
    end_time_ms: float = 0.0
    
    def add_hop(self, hop: HopMetrics):
        """Add a hop record and update total metrics."""
        self.hop_records.append(hop)
        self.total_latency_ms += hop.latency_ms
        self.total_hops += 1
    
    def finalize(self, success: bool, drop_reason: Optional[str] = None):
        """Mark the route as complete."""
        self.success = success
        self.packet_delivered = success
        self.drop_reason = drop_reason
        self.end_time_ms = time.time() * 1000
    
    def get_average_node_utilization(self) -> float:
        """Calculate average node resource utilization across all hops."""
        if not self.hop_records:
            return 0.0
        total_util = sum(
            (hop.node_cpu_utilization + hop.node_memory_utilization + hop.node_bandwidth_utilization) / 3.0
            for hop in self.hop_records
        )
        return total_util / len(self.hop_records)
    
    def get_max_node_utilization(self) -> float:
        """Get maximum node utilization encountered in the path."""
        if not self.hop_records:
            return 0.0
        return max(
            (hop.node_cpu_utilization + hop.node_memory_utilization + hop.node_bandwidth_utilization) / 3.0
            for hop in self.hop_records
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'packetId': self.packet_id,
            'sourceId': self.source_id,
            'destId': self.dest_id,
            'algorithm': self.algorithm,
            'success': self.success,
            'packetDelivered': self.packet_delivered,
            'totalLatencyMs': self.total_latency_ms,
            'totalHops': self.total_hops,
            'dropReason': self.drop_reason,
            'startTimeMs': self.start_time_ms,
            'endTimeMs': self.end_time_ms,
            'durationMs': self.end_time_ms - self.start_time_ms,
            'averageNodeUtilization': self.get_average_node_utilization(),
            'maxNodeUtilization': self.get_max_node_utilization(),
            'hopRecords': [hop.to_dict() for hop in self.hop_records]
        }


class MetricsComparator:
    """Compare routing metrics between RL and Dijkstra algorithms."""
    
    def __init__(self):
        self.rl_metrics: List[RouteMetrics] = []
        self.dijkstra_metrics: List[RouteMetrics] = []
    
    def add_rl_metric(self, metric: RouteMetrics):
        """Add an RL routing metric."""
        self.rl_metrics.append(metric)
    
    def add_dijkstra_metric(self, metric: RouteMetrics):
        """Add a Dijkstra routing metric."""
        self.dijkstra_metrics.append(metric)
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive comparison summary."""
        if not self.rl_metrics or not self.dijkstra_metrics:
            return {
                'error': 'Insufficient data for comparison',
                'rl_count': len(self.rl_metrics),
                'dijkstra_count': len(self.dijkstra_metrics)
            }
        
        rl_stats = self._calculate_statistics(self.rl_metrics, "RL-DQN")
        dijkstra_stats = self._calculate_statistics(self.dijkstra_metrics, "DIJKSTRA")
        
        return {
            'rl': rl_stats,
            'dijkstra': dijkstra_stats,
            'comparison': {
                'latency_improvement_percent': self._calculate_improvement(
                    dijkstra_stats['avg_latency_ms'], rl_stats['avg_latency_ms']
                ),
                'hop_count_improvement_percent': self._calculate_improvement(
                    dijkstra_stats['avg_hops'], rl_stats['avg_hops']
                ),
                'resource_utilization_improvement_percent': self._calculate_improvement(
                    dijkstra_stats['avg_node_utilization'], rl_stats['avg_node_utilization']
                ),
                'delivery_rate_improvement_percent': self._calculate_improvement(
                    rl_stats['delivery_rate_percent'], dijkstra_stats['delivery_rate_percent'],
                    higher_is_better=True
                )
            }
        }
    
    def _calculate_statistics(self, metrics: List[RouteMetrics], algorithm: str) -> Dict[str, Any]:
        """Calculate statistics for a set of metrics."""
        successful = [m for m in metrics if m.success]
        
        latencies = [m.total_latency_ms for m in successful]
        hops = [m.total_hops for m in successful]
        utilizations = [m.get_average_node_utilization() for m in successful]
        
        return {
            'algorithm': algorithm,
            'total_packets': len(metrics),
            'successful_packets': len(successful),
            'delivery_rate_percent': (len(successful) / len(metrics) * 100) if metrics else 0,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'median_latency_ms': np.median(latencies) if latencies else 0,
            'std_latency_ms': np.std(latencies) if latencies else 0,
            'avg_hops': np.mean(hops) if hops else 0,
            'median_hops': np.median(hops) if hops else 0,
            'avg_node_utilization': np.mean(utilizations) if utilizations else 0,
            'max_node_utilization': max(utilizations) if utilizations else 0
        }
    
    def _calculate_improvement(self, baseline: float, optimized: float, 
                               higher_is_better: bool = False) -> float:
        """Calculate percentage improvement."""
        if baseline == 0:
            return 0.0
        if higher_is_better:
            return ((optimized - baseline) / baseline) * 100
        else:
            return ((baseline - optimized) / baseline) * 100
    
    def save_to_file(self, filepath: str):
        """Save comparison results to JSON file."""
        summary = self.get_comparison_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def print_summary(self):
        """Print a formatted comparison summary."""
        summary = self.get_comparison_summary()
        
        if 'error' in summary:
            print(f"Error: {summary['error']}")
            return
        
        print("\n" + "="*80)
        print("ROUTING ALGORITHM COMPARISON SUMMARY")
        print("="*80)
        
        print("\n📊 RL-DQN Performance:")
        self._print_stats(summary['rl'])
        
        print("\n📊 Dijkstra Performance:")
        self._print_stats(summary['dijkstra'])
        
        print("\n📈 Improvement Analysis:")
        comp = summary['comparison']
        print(f"  Latency Reduction:        {comp['latency_improvement_percent']:+.2f}%")
        print(f"  Hop Count Reduction:      {comp['hop_count_improvement_percent']:+.2f}%")
        print(f"  Resource Optimization:    {comp['resource_utilization_improvement_percent']:+.2f}%")
        print(f"  Delivery Rate Change:     {comp['delivery_rate_improvement_percent']:+.2f}%")
        print("="*80 + "\n")
    
    def _print_stats(self, stats: Dict[str, Any]):
        """Print statistics in a formatted way."""
        print(f"  Algorithm:              {stats['algorithm']}")
        print(f"  Total Packets:          {stats['total_packets']}")
        print(f"  Successful Deliveries:  {stats['successful_packets']}")
        print(f"  Delivery Rate:          {stats['delivery_rate_percent']:.2f}%")
        print(f"  Average Latency:        {stats['avg_latency_ms']:.2f} ms")
        print(f"  Median Latency:         {stats['median_latency_ms']:.2f} ms")
        print(f"  Average Hops:           {stats['avg_hops']:.2f}")
        print(f"  Avg Node Utilization:   {stats['avg_node_utilization']:.4f}")
