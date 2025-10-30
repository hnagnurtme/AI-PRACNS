# test_satellite_env.py

import logging
from unittest.mock import Mock
import numpy as np

from python.env.satellite_simulator import SatelliteEnv
from python.utils.state_builder import StateBuilder

# --- Cấu hình Logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mock Data ---
MOCK_CURRENT_NODE = {
    "nodeId": "SAT-LEO-1",
    "nodeType": "LEO_SATELLITE",
    "position": {"latitude": 10.0, "longitude": 100.0, "altitude": 500.0},
    "resourceUtilization": 0.5,
    "currentPacketCount": 50,
    "packetBufferCapacity": 100,
    "packetLossRate": 0.01,
    "nodeProcessingDelayMs": 5.0,
    "operational": True,
    "lastUpdated": None,
    "neighbors": ["N-HANOI", "SAT-LEO-2"],
    "communication": {"frequencyGHz": 8.2, "bandwidthMHz": 500.0}
}

MOCK_DEST_NODE = {
    "nodeId": "GS-HANOI",
    "position": {"latitude": 21.0, "longitude": 105.0, "altitude": 0.0}
}

MOCK_NEIGHBORS = {
    "N-HANOI": {
        "nodeId": "N-HANOI",
        "nodeType": "GROUND_STATION",
        "position": {"latitude": 21.0, "longitude": 105.0, "altitude": 0.0},
        "resourceUtilization": 0.1,
        "packetLossRate": 0.001,
        "nodeProcessingDelayMs": 1.0,
        "operational": True,
        "communication": {"frequencyGHz": 2.2, "bandwidthMHz": 100.0}
    },
    "SAT-LEO-2": {
        "nodeId": "SAT-LEO-2",
        "nodeType": "LEO_SATELLITE",
        "position": {"latitude": 15.0, "longitude": 102.0, "altitude": 550.0},
        "resourceUtilization": 0.8,
        "packetLossRate": 0.05,
        "nodeProcessingDelayMs": 5.0,
        "operational": True,
        "communication": {"frequencyGHz": 8.2, "bandwidthMHz": 500.0}
    }
}

MOCK_PACKET = {
    "packetId": "TEST_PKT_01",
    "currentHoldingNodeId": "SAT-LEO-1",
    "stationDest": "GS-HANOI",
    "accumulatedDelayMs": 10.0,
    "ttl": 15,
    "serviceQoS": {
        "serviceType": "VIDEO_STREAM",
        "maxLatencyMs": 150.0,
        "minBandwidthMbps": 5.0,
        "maxLossRate": 0.02
    }
}

# --- Mock StateBuilder ---
mock_db = Mock()

# get_node side_effect
mock_db.get_node.side_effect = lambda node_id, projection=None: {
    MOCK_CURRENT_NODE["nodeId"]: MOCK_CURRENT_NODE,
    MOCK_DEST_NODE["nodeId"]: MOCK_DEST_NODE,
    "N-HANOI": MOCK_NEIGHBORS["N-HANOI"],
    "SAT-LEO-2": MOCK_NEIGHBORS["SAT-LEO-2"]
}.get(node_id)

# get_neighbor_status_batch
mock_db.get_neighbor_status_batch.side_effect = lambda neighbor_ids, projection=None: {
    nid: MOCK_NEIGHBORS[nid] for nid in neighbor_ids if nid in MOCK_NEIGHBORS
}

builder = StateBuilder(mock_db)
env = SatelliteEnv(builder)

# --- Test Reset ---
state = env.reset(MOCK_PACKET)
logger.info(f"Reset state vector: {state}")

# --- Simulate Hop to Neighbor N-HANOI ---
next_packet = MOCK_PACKET.copy()
next_packet["currentHoldingNodeId"] = "N-HANOI"
next_packet["accumulatedDelayMs"] += 5.0
next_packet["ttl"] -= 1

next_state, reward, done = env.step(action_index=0, neighbor_id="N-HANOI", new_packet_data=next_packet)
logger.info(f"Next state vector: {next_state}")
logger.info(f"Reward: {reward}")
logger.info(f"Done: {done}")

# --- Simulate Hop to Destination ---
final_packet = next_packet.copy()
final_packet["currentHoldingNodeId"] = "GS-HANOI"
final_packet["accumulatedDelayMs"] += 5.0
final_packet["ttl"] -= 1

final_state, final_reward, final_done = env.step(action_index=0, neighbor_id="GS-HANOI", new_packet_data=final_packet)
logger.info(f"Final state vector: {final_state}")
logger.info(f"Final step reward (goal): {final_reward}, done: {final_done}")
