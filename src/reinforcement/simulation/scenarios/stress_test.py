"""
Stress test scenario for SAGIN network simulation.

This scenario tests the network under extreme conditions:
- Severe weather conditions
- High traffic load
- Frequent node/link failures
- Rapid node mobility
"""

import numpy as np
from typing import Dict, Any


class StressTestScenario:
    """
    Stress test scenario with extreme network conditions.
    Tests RL agent's robustness and worst-case performance.
    """

    def __init__(self):
        self.name = "stress_test"
        self.description = "Extreme network conditions for stress testing"

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for stress test scenario.

        Returns:
            Dictionary with scenario configuration
        """
        return {
            'dynamics': {
                'enabled': True,
                'time_step': 0.5,  # Faster time steps
                'mobility': {
                    'enabled': True,
                    'max_neighbors': 8,
                    'update_frequency': 2.0,  # Update positions every 2 seconds
                    'velocity_multiplier': 2.0  # Faster movement
                },
                'weather': {
                    'enabled': True,
                    'change_probability': 0.15,  # 15% chance per time step
                    'regional_weather': True,
                    'regions': ['region_A', 'region_B', 'region_C', 'region_D', 'region_E'],
                    'extreme_weather_prob': 0.3  # 30% of weather events are extreme
                },
                'traffic': {
                    'enabled': True,
                    'base_load': 2.0,
                    'peak_load': 5.0,  # Very high peak load
                    'period': 43200,  # 12 hours (faster cycles)
                    'spike_probability': 0.1  # 10% chance of traffic spikes
                },
                'failures': {
                    'enabled': True,
                    'node_failure_prob': 0.005,  # 0.5% per time step (5x baseline)
                    'link_failure_prob': 0.01,  # 1% per time step (5x baseline)
                    'recovery_time': 600.0,  # 10 minutes to recover
                    'cascading_failures': True  # Failures can cascade
                }
            },
            'reward_weights': {
                'delivery': 1500.0,  # Higher reward for successful delivery under stress
                'drop': 750.0,  # Higher penalty for drops
                'hop_cost': 15.0,
                'delay_penalty': 2.0,
                'battery_weight': 0.3,
                'congestion_weight': 0.5,
                'link_quality_weight': 0.8,
                'weather_penalty': 100.0,
                'traffic_bonus': 50.0,
                'resilience_bonus': 100.0  # Bonus for successful routing despite failures
            },
            'training': {
                'max_hops': 15,  # Allow more hops due to difficult conditions
                'num_episodes': 3000,
                'learning_rate': 0.0003,
                'gamma': 0.97,
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,  # Higher minimum exploration
                'epsilon_decay': 10000,
                'batch_size': 256,  # Larger batch for stability
                'target_update': 30,
                'prioritized_replay': True  # Use prioritized experience replay
            },
            'evaluation': {
                'num_episodes': 150,
                'deterministic': False,  # Stochastic evaluation
                'stress_level': 'extreme'
            }
        }

    def get_test_packets(self, num_packets: int = 50) -> list:
        """
        Generate test packets for stress test scenario.

        Args:
            num_packets: Number of test packets to generate

        Returns:
            List of packet configurations with challenging requirements
        """
        packets = []

        # Many diverse node pairs
        node_pairs = [
            ('gs1', 'gs4'), ('gs2', 'gs5'), ('gs3', 'gs1'),
            ('leo1', 'gs1'), ('leo2', 'gs3'), ('leo3', 'gs5'),
            ('gs1', 'leo1'), ('gs2', 'leo2'), ('gs4', 'leo3'),
            ('leo1', 'leo2'), ('leo2', 'leo3'), ('leo3', 'leo1')
        ]

        # Mix of strict and relaxed QoS
        qos_levels = [
            {'maxLatencyMs': 300.0, 'priority': 'critical'},   # Very strict
            {'maxLatencyMs': 600.0, 'priority': 'high'},
            {'maxLatencyMs': 1200.0, 'priority': 'medium'},
            {'maxLatencyMs': 3000.0, 'priority': 'low'}
        ]

        packet_sizes = [128, 256, 512, 1024, 1500, 2048]

        for i in range(num_packets):
            src, dest = node_pairs[np.random.randint(0, len(node_pairs))]
            qos = qos_levels[np.random.randint(0, len(qos_levels))].copy()
            size = packet_sizes[np.random.randint(0, len(packet_sizes))]

            # Some packets start with accumulated delay (mid-transmission)
            initial_delay = np.random.uniform(0, 200) if np.random.random() < 0.3 else 0

            packets.append({
                'currentHoldingNodeId': src,
                'stationDest': dest,
                'ttl': np.random.randint(20, 50),  # Lower TTL for stress
                'serviceQoS': qos,
                'accumulatedDelayMs': initial_delay,
                'packetSize': size
            })

        return packets

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate results meet stress test expectations.

        Args:
            results: Results dictionary from simulation

        Returns:
            True if results are acceptable under stress conditions
        """
        delivery_rate = results.get('delivery_rate', 0.0)
        avg_latency = results.get('avg_latency', float('inf'))
        avg_hops = results.get('avg_hops', float('inf'))

        # Very lenient thresholds due to extreme conditions
        return (
            delivery_rate >= 0.45 and  # At least 45% delivery (challenging!)
            avg_latency <= 4000.0 and  # Max 4000ms average latency
            avg_hops <= 14.0  # Max 14 hops on average
        )

    def get_extreme_weather_events(self) -> list:
        """
        Get extreme weather events for stress testing.

        Returns:
            List of extreme weather event configurations
        """
        return [
            {'time': 500, 'region': 'region_A', 'condition': 'storm', 'intensity': 'severe'},
            {'time': 800, 'region': 'region_B', 'condition': 'thunderstorm', 'intensity': 'severe'},
            {'time': 1200, 'region': 'region_C', 'condition': 'heavy_rain', 'intensity': 'extreme'},
            {'time': 1600, 'region': 'region_D', 'condition': 'storm', 'intensity': 'severe'},
            {'time': 2000, 'region': 'region_E', 'condition': 'thunderstorm', 'intensity': 'extreme'},
            {'time': 2500, 'region': 'region_A', 'condition': 'clear', 'intensity': 'normal'},
            {'time': 3000, 'region': 'region_B', 'condition': 'storm', 'intensity': 'severe'}
        ]

    def get_failure_cascades(self) -> list:
        """
        Get cascading failure events for stress testing.

        Returns:
            List of cascading failure configurations
        """
        return [
            {
                'time': 1000,
                'initial_node': 'leo1',
                'cascade_probability': 0.4,
                'max_affected': 3,
                'duration': 400
            },
            {
                'time': 2000,
                'initial_node': 'gs2',
                'cascade_probability': 0.5,
                'max_affected': 4,
                'duration': 500
            },
            {
                'time': 3000,
                'initial_node': 'leo3',
                'cascade_probability': 0.3,
                'max_affected': 2,
                'duration': 300
            }
        ]

    def get_traffic_spikes(self) -> list:
        """
        Get traffic spike events for stress testing.

        Returns:
            List of traffic spike configurations
        """
        return [
            {'time': 600, 'magnitude': 8.0, 'duration': 200, 'affected_nodes': ['gs1', 'gs2', 'leo1']},
            {'time': 1400, 'magnitude': 10.0, 'duration': 300, 'affected_nodes': ['gs3', 'leo2', 'leo3']},
            {'time': 2200, 'magnitude': 12.0, 'duration': 250, 'affected_nodes': ['gs1', 'gs4', 'leo1', 'leo2']},
            {'time': 2800, 'magnitude': 15.0, 'duration': 400, 'affected_nodes': ['gs2', 'gs3', 'gs5']}
        ]


def create_stress_test_scenario() -> StressTestScenario:
    """Factory function to create stress test scenario"""
    return StressTestScenario()
