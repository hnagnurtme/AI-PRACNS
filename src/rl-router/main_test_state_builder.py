import numpy as np
import logging
from unittest.mock import Mock
from datetime import datetime
import time
from python.utils.state_builder import StateBuilder, MAX_NEIGHBORS, NUM_SERVICE_TYPES, NUM_NODE_TYPES, NEIGHBOR_SLOT_SIZE
from python.utils.db_connector import MongoConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CURRENT_TIMESTAMP_MS = time.time()*1000

# --- MOCK DATA ---
MOCK_CURRENT_NODE = {
    "nodeId":"SAT-LEO-1",
    "nodeType":"LEO_SATELLITE",
    "position":{"latitude":10.0,"longitude":100.0,"altitude":500.0},
    "resourceUtilization":0.5,
    "currentPacketCount":50,
    "packetBufferCapacity":100,
    "packetLossRate":0.01,
    "nodeProcessingDelayMs":5.0,
    "operational":True,
    "lastUpdated":datetime.fromtimestamp((CURRENT_TIMESTAMP_MS-1000)/1000),
    "neighbors":["N-HANOI","SAT-LEO-2"],
    "communication":{"frequencyGHz":8.2,"bandwidthMHz":500.0}
}

MOCK_DEST_NODE = {
    "nodeId":"GS-HANOI",
    "position":{"latitude":21.0,"longitude":105.0,"altitude":0.0}
}

MOCK_NEIGHBORS_DATA = {
    "N-HANOI":{
        "nodeId":"N-HANOI",
        "nodeType":"GROUND_STATION",
        "position":{"latitude":21.0,"longitude":105.0,"altitude":0.0},
        "resourceUtilization":0.1,
        "packetLossRate":0.001,
        "nodeProcessingDelayMs":1.0,
        "operational":True,
        "communication":{"frequencyGHz":2.2,"bandwidthMHz":100.0}
    },
    "SAT-LEO-2":{
        "nodeId":"SAT-LEO-2",
        "nodeType":"LEO_SATELLITE",
        "position":{"latitude":15.0,"longitude":102.0,"altitude":550.0},
        "resourceUtilization":0.8,
        "packetLossRate":0.05,
        "nodeProcessingDelayMs":5.0,
        "operational":True,
        "communication":{"frequencyGHz":8.2,"bandwidthMHz":500.0}
    }
}

MOCK_PACKET_DATA = {
    "packetId":"TEST_PKT_01",
    "currentHoldingNodeId":"SAT-LEO-1",
    "stationDest":"GS-HANOI",
    "accumulatedDelayMs":10.0,
    "ttl":15,
    "serviceQoS":{"serviceType":"VIDEO_STREAM","maxLatencyMs":150.0,"minBandwidthMbps":5.0,"maxLossRate":0.02}
}

# --- MOCK DB CONNECTOR ---
def create_mock_db_connector() -> MongoConnector:
    mock_db = Mock(spec=MongoConnector)
    mock_db.get_node.side_effect = lambda node_id, projection=None: \
        MOCK_CURRENT_NODE if node_id==MOCK_CURRENT_NODE["nodeId"] else \
        MOCK_DEST_NODE if node_id==MOCK_DEST_NODE["nodeId"] else None
    mock_db.get_neighbor_status_batch.side_effect = lambda ids, projection=None: {nid: MOCK_NEIGHBORS_DATA[nid] for nid in ids if nid in MOCK_NEIGHBORS_DATA}
    return mock_db

# --- TEST FUNCTION ---
def test_state_vector_builder():
    logger.info("=== BẮT ĐẦU KIỂM THỬ STATE BUILDER ===")
    builder = StateBuilder(create_mock_db_connector())
    S = builder.get_state_vector(MOCK_PACKET_DATA)

    expected_size = (NUM_SERVICE_TYPES+5) + (NUM_NODE_TYPES+4) + 6 + (MAX_NEIGHBORS*NEIGHBOR_SLOT_SIZE)
    logger.info(f"Kích thước vector: {S.size}, mong đợi: {expected_size}")
    assert S.shape[0]==expected_size, "❌ Kích thước vector sai!"

    # Kiểm tra phạm vi [0,1]
    assert np.all(S>=0.0) and np.all(S<=1.0), "❌ Giá trị vector vượt giới hạn [0,1]"

    logger.info("✅ STATE VECTOR BUILDER SANITY CHECK PASSED!")

if __name__=="__main__":
    test_state_vector_builder()
