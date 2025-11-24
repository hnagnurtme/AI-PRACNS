from typing import List, Dict, Any,Optional
from .dynamic_env import DynamicSatelliteEnv


class EnvironmentManager:
    def __init__(self, config: dict):
        self.config = config
        self.environments: List[DynamicSatelliteEnv] = []

    def create_environment(self, state_builder, nodes: Dict, weights=None):
        env = DynamicSatelliteEnv(
            state_builder=state_builder,
            nodes=nodes,
            weights=weights or {},
            dynamic_config=self.config.get('dynamics', {})
        )
        self.environments.append(env)
        return env

    def reset_all(self, initial_packet_data: Optional[Dict[str, Any]] = None):
        """
        Reset all environments.

        Args:
            initial_packet_data: Optional initial packet data. If not provided,
                                uses default packet data.
        """
        if initial_packet_data is None:
            # Default packet data
            initial_packet_data = {
                "currentHoldingNodeId": "gs1",
                "stationDest": "gs2",
                "ttl": 50,
                "serviceQoS": {"maxLatencyMs": 1000.0},
                "accumulatedDelayMs": 0.0,
                "packetSize": 1024
            }

        for env in self.environments:
            from simulation.core.packet import Packet  # Import the Packet class if not already imported
            initial_packet = Packet(**initial_packet_data)
            env.reset(initial_packet)