"""
Dynamic changes scenario for SAGIN network simulation.

This scenario tests the network under realistic dynamic conditions:
- Weather effects enabled
- Variable traffic load
- Occasional node/link failures
- Node mobility
"""

import numpy as np
from typing import Dict, Any


class DynamicChangesScenario:
    """
    Scenario with dynamic network conditions.
    Tests RL agent's ability to adapt to changing environments.
    """

    def __init__(self):
        self.name = "dynamic_changes"
        self.description = "Realistic dynamic network conditions"

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for dynamic changes scenario.

        Returns:
            Dictionary with scenario configuration
        """
        return {
            'dynamics': {
                'enabled': True,
                'time_step': 1.0,
                'mobility': {
                    'enabled': True,
                    'max_neighbors': 10,
                    'update_frequency': 5.0  # Update positions every 5 seconds
                },
                'weather': {
                    'enabled': True,
                    'change_probability': 0.05,  # 5% chance per time step
                    'regional_weather': True,
                    'regions': ['region_A', 'region_B', 'region_C', 'region_D']
                },
                'traffic': {
                    'enabled': True,
                    'base_load': 1.0,
                    'peak_load': 2.5,
                    'period': 86400  # 24 hours in seconds
                },
                'failures': {
                    'enabled': True,
                    'node_failure_prob': 0.001,  # 0.1% per time step
                    'link_failure_prob': 0.002,  # 0.2% per time step
                    'recovery_time': 300.0  # 5 minutes to recover
                }
            },
            'reward_weights': {
                'delivery': 1000.0,
                'drop': 500.0,
                'hop_cost': 10.0,
                'delay_penalty': 1.5,
                'battery_weight': 0.2,
                'congestion_weight': 0.3,
                'link_quality_weight': 0.6,
                'weather_penalty': 50.0,
                'traffic_bonus': 20.0
            },
            'training': {
                'max_hops': 12,
                'num_episodes': 2000,
                'learning_rate': 0.0005,
                'gamma': 0.98,
                'epsilon_start': 1.0,
                'epsilon_end': 0.02,
                'epsilon_decay': 8000,
                'batch_size': 128,
                'target_update': 20
            },
            'evaluation': {
                'num_episodes': 100,
                'deterministic': False  # Stochastic evaluation
            }
        }

    def get_test_packets(self, num_packets: int = 20) -> list:
        """
        Generate test packets for dynamic scenario.

        Args:
            num_packets: Number of test packets to generate

        Returns:
            List of packet configurations with varying QoS requirements
        """
        packets = []

        # Mix of different packet types and QoS requirements
        node_pairs = [
            ('gs1', 'gs2'),
            ('gs1', 'leo1'),
            ('gs2', 'leo2'),
            ('leo1', 'gs3'),
            ('leo2', 'leo3'),
            ('gs3', 'gs1'),
            ('leo3', 'gs2')
        ]

        qos_levels = [
            {'maxLatencyMs': 500.0, 'priority': 'high'},
            {'maxLatencyMs': 1000.0, 'priority': 'medium'},
            {'maxLatencyMs': 2000.0, 'priority': 'low'}
        ]

        packet_sizes = [256, 512, 1024, 1500]

        for i in range(num_packets):
            src, dest = node_pairs[i % len(node_pairs)]
            qos = qos_levels[i % len(qos_levels)]
            size = packet_sizes[i % len(packet_sizes)]

            packets.append({
                'currentHoldingNodeId': src,
                'stationDest': dest,
                'ttl': np.random.randint(30, 60),
                'serviceQoS': qos.copy(),
                'accumulatedDelayMs': 0.0,
                'packetSize': size
            })

        return packets

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate results meet dynamic scenario expectations.

        Args:
            results: Results dictionary from simulation

        Returns:
            True if results are acceptable under dynamic conditions
        """
        delivery_rate = results.get('delivery_rate', 0.0)
        avg_latency = results.get('avg_latency', float('inf'))
        avg_hops = results.get('avg_hops', float('inf'))

        # More lenient thresholds due to dynamic conditions
        return (
            delivery_rate >= 0.65 and  # At least 65% delivery
            avg_latency <= 2500.0 and  # Max 2500ms average latency
            avg_hops <= 10.0  # Max 10 hops on average
        )

    def get_weather_events(self) -> list:
        """
        Get scheduled weather events for scenario.

        Returns:
            List of weather event configurations
        """
        return [
            {'time': 1000, 'region': 'region_A', 'condition': 'heavy_rain'},
            {'time': 2000, 'region': 'region_B', 'condition': 'storm'},
            {'time': 3000, 'region': 'region_C', 'condition': 'clear'},
            {'time': 4000, 'region': 'region_D', 'condition': 'light_rain'}
        ]

    def get_failure_events(self) -> list:
        """
        Get scheduled failure events for scenario.

        Returns:
            List of failure event configurations
        """
        return [
            {'time': 1500, 'node_id': 'leo2', 'duration': 300},
            {'time': 2500, 'node_id': 'gs2', 'duration': 180},
            {'time': 3500, 'link': ('leo1', 'leo3'), 'duration': 240}
        ]


def create_dynamic_scenario() -> DynamicChangesScenario:
    """Factory function to create dynamic changes scenario"""
    return DynamicChangesScenario()
