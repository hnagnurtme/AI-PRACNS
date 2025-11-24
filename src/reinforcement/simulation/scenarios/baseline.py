"""
Baseline scenario for SAGIN network simulation.

This scenario provides a stable, ideal network condition for baseline testing:
- No weather effects
- Minimal traffic load
- No node failures
- Static node positions
"""

import numpy as np
from typing import Dict, Any


class BaselineScenario:
    """
    Baseline scenario with ideal network conditions.
    Used for establishing performance baselines.
    """

    def __init__(self):
        self.name = "baseline"
        self.description = "Ideal network conditions for baseline testing"

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for baseline scenario.

        Returns:
            Dictionary with scenario configuration
        """
        return {
            'dynamics': {
                'enabled': False,
                'time_step': 1.0,
                'mobility': {
                    'enabled': False,
                    'max_neighbors': 10
                },
                'weather': {
                    'enabled': False,
                    'change_probability': 0.0
                },
                'traffic': {
                    'enabled': False,
                    'base_load': 1.0,
                    'peak_load': 1.0
                },
                'failures': {
                    'enabled': False,
                    'node_failure_prob': 0.0,
                    'link_failure_prob': 0.0
                }
            },
            'reward_weights': {
                'delivery': 1000.0,
                'drop': 500.0,
                'hop_cost': 10.0,
                'delay_penalty': 1.0,
                'battery_weight': 0.1,
                'congestion_weight': 0.2,
                'link_quality_weight': 0.5,
                'weather_penalty': 0.0,
                'traffic_bonus': 0.0
            },
            'training': {
                'max_hops': 10,
                'num_episodes': 1000,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 5000,
                'batch_size': 64,
                'target_update': 10
            },
            'evaluation': {
                'num_episodes': 50,
                'deterministic': True
            }
        }

    def get_test_packets(self, num_packets: int = 10) -> list:
        """
        Generate test packets for baseline scenario.

        Args:
            num_packets: Number of test packets to generate

        Returns:
            List of packet configurations
        """
        packets = []

        # Predefined source-destination pairs for consistent testing
        pairs = [
            ('gs1', 'gs2'),
            ('gs1', 'leo1'),
            ('leo1', 'gs2'),
            ('leo1', 'leo2'),
            ('gs2', 'leo1')
        ]

        for i in range(num_packets):
            src, dest = pairs[i % len(pairs)]
            packets.append({
                'currentHoldingNodeId': src,
                'stationDest': dest,
                'ttl': 50,
                'serviceQoS': {
                    'maxLatencyMs': 1000.0
                },
                'accumulatedDelayMs': 0.0,
                'packetSize': 1024
            })

        return packets

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate results meet baseline expectations.

        Args:
            results: Results dictionary from simulation

        Returns:
            True if results meet baseline expectations
        """
        # For baseline scenario, expect high delivery rate
        delivery_rate = results.get('delivery_rate', 0.0)
        avg_latency = results.get('avg_latency', float('inf'))
        avg_hops = results.get('avg_hops', float('inf'))

        return (
            delivery_rate >= 0.8 and  # At least 80% delivery
            avg_latency <= 1500.0 and  # Max 1500ms average latency
            avg_hops <= 8.0  # Max 8 hops on average
        )


def create_baseline_scenario() -> BaselineScenario:
    """Factory function to create baseline scenario"""
    return BaselineScenario()
