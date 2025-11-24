"""
Utilities for comparing RL agent performance against baseline algorithms.
Provides comprehensive metrics and fairness analysis.
"""

from typing import Dict, List, Any
import numpy as np


class ComparisonUtils:
    """
    Enhanced comparison utilities for RL vs Baseline algorithms.
    Ensures fair comparison with dynamic neighbor updates.
    """
    
    @staticmethod
    def compare_algorithms(results_rl: dict, results_dijkstra: dict, 
                          results_baseline: dict) -> dict:
        """
        Compare algorithm results with detailed metrics.
        
        Args:
            results_rl: RL algorithm results
            results_dijkstra: Dijkstra algorithm results  
            results_baseline: Baseline (greedy/random) algorithm results
        
        Returns:
            Comprehensive comparison dictionary
        """
        comparison = {
            'algorithms': {
                'RL': results_rl,
                'Dijkstra': results_dijkstra,
                'Baseline': results_baseline
            },
            'summary': ComparisonUtils._generate_summary(
                results_rl, results_dijkstra, results_baseline
            ),
            'winner': ComparisonUtils._determine_winner(
                results_rl, results_dijkstra, results_baseline
            )
        }
        return comparison
    
    @staticmethod
    def _generate_summary(results_rl: dict, results_dijkstra: dict, 
                         results_baseline: dict) -> dict:
        """Generate comparative summary statistics."""
        
        def safe_mean(results: dict, key: str, default=0.0) -> float:
            values = results.get(key, [])
            return np.mean(values) if values else default
        
        def safe_success_rate(results: dict) -> float:
            delivered = results.get('delivered', [])
            return (sum(delivered) / len(delivered) * 100) if delivered else 0.0
        
        summary = {
            'delivery_rate': {
                'RL': safe_success_rate(results_rl),
                'Dijkstra': safe_success_rate(results_dijkstra),
                'Baseline': safe_success_rate(results_baseline),
            },
            'avg_latency': {
                'RL': safe_mean(results_rl, 'latency'),
                'Dijkstra': safe_mean(results_dijkstra, 'latency'),
                'Baseline': safe_mean(results_baseline, 'latency'),
            },
            'avg_hops': {
                'RL': safe_mean(results_rl, 'hops'),
                'Dijkstra': safe_mean(results_dijkstra, 'hops'),
                'Baseline': safe_mean(results_baseline, 'hops'),
            },
            'avg_reward': {
                'RL': safe_mean(results_rl, 'total_reward'),
                'Dijkstra': safe_mean(results_dijkstra, 'total_reward', 0.0),
                'Baseline': safe_mean(results_baseline, 'total_reward', 0.0),
            }
        }
        
        # Add network-wide metrics if available
        if 'network_metrics' in results_rl:
            summary['network_efficiency'] = {
                'RL': results_rl.get('network_metrics', {}),
                'Dijkstra': results_dijkstra.get('network_metrics', {}),
                'Baseline': results_baseline.get('network_metrics', {}),
            }
        
        return summary
    
    @staticmethod
    def _determine_winner(results_rl: dict, results_dijkstra: dict,
                         results_baseline: dict) -> dict:
        """
        Determine which algorithm performs best across multiple metrics.
        RL should consistently win if properly optimized.
        """
        
        def safe_success_rate(results: dict) -> float:
            delivered = results.get('delivered', [])
            return (sum(delivered) / len(delivered) * 100) if delivered else 0.0
        
        def safe_mean(results: dict, key: str) -> float:
            values = results.get(key, [])
            return np.mean(values) if values else float('inf')
        
        scores = {
            'RL': 0,
            'Dijkstra': 0,
            'Baseline': 0
        }
        
        # Compare delivery rate (higher is better)
        delivery_rates = {
            'RL': safe_success_rate(results_rl),
            'Dijkstra': safe_success_rate(results_dijkstra),
            'Baseline': safe_success_rate(results_baseline),
        }
        winner = max(delivery_rates, key=delivery_rates.get)
        scores[winner] += 3  # Delivery rate is most important
        
        # Compare latency (lower is better)
        latencies = {
            'RL': safe_mean(results_rl, 'latency'),
            'Dijkstra': safe_mean(results_dijkstra, 'latency'),
            'Baseline': safe_mean(results_baseline, 'latency'),
        }
        winner = min(latencies, key=latencies.get)
        scores[winner] += 2
        
        # Compare hops (lower is better)
        hops = {
            'RL': safe_mean(results_rl, 'hops'),
            'Dijkstra': safe_mean(results_dijkstra, 'hops'),
            'Baseline': safe_mean(results_baseline, 'hops'),
        }
        winner = min(hops, key=hops.get)
        scores[winner] += 1
        
        overall_winner = max(scores, key=scores.get)
        
        return {
            'overall': overall_winner,
            'scores': scores,
            'metrics': {
                'delivery_rate': delivery_rates,
                'latency': latencies,
                'hops': hops
            },
            'rl_advantage': scores['RL'] > max(scores['Dijkstra'], scores['Baseline']),
            'warning': 'RL should outperform baselines!' if overall_winner != 'RL' else None
        }
    
    @staticmethod
    def calculate_improvement_percentage(rl_value: float, baseline_value: float, 
                                        lower_is_better: bool = False) -> float:
        """
        Calculate percentage improvement of RL over baseline.
        
        Args:
            rl_value: RL metric value
            baseline_value: Baseline metric value
            lower_is_better: True if lower values are better (e.g., latency)
        
        Returns:
            Percentage improvement (positive means RL is better)
        """
        if baseline_value == 0:
            return 0.0
        
        if lower_is_better:
            # For metrics like latency where lower is better
            improvement = ((baseline_value - rl_value) / baseline_value) * 100
        else:
            # For metrics like delivery rate where higher is better
            improvement = ((rl_value - baseline_value) / baseline_value) * 100
        
        return improvement
    
    @staticmethod
    def generate_report(comparison: dict) -> str:
        """Generate human-readable comparison report."""
        summary = comparison.get('summary', {})
        winner = comparison.get('winner', {})
        
        report = []
        report.append("=" * 80)
        report.append("RL vs Baseline Algorithm Comparison Report")
        report.append("=" * 80)
        report.append("")
        
        # Delivery Rate
        report.append("Packet Delivery Rate:")
        for algo, rate in summary.get('delivery_rate', {}).items():
            report.append(f"  {algo:12s}: {rate:6.2f}%")
        report.append("")
        
        # Average Latency
        report.append("Average Latency (ms):")
        for algo, latency in summary.get('avg_latency', {}).items():
            report.append(f"  {algo:12s}: {latency:8.2f}")
        report.append("")
        
        # Average Hops
        report.append("Average Hops:")
        for algo, hops in summary.get('avg_hops', {}).items():
            report.append(f"  {algo:12s}: {hops:6.2f}")
        report.append("")
        
        # Overall Winner
        report.append("=" * 80)
        report.append(f"Overall Winner: {winner.get('overall', 'N/A')}")
        report.append(f"Scores: {winner.get('scores', {})}")
        
        if winner.get('warning'):
            report.append("")
            report.append(f"⚠️  WARNING: {winner['warning']}")
        
        report.append("=" * 80)
        
        return "\n".join(report)