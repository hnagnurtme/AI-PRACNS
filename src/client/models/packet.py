import json
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field, asdict
import base64
from enum import Enum

@dataclass
class Position:
    latitude: float
    longitude: float
    altitude: float

@dataclass
class HopRecord:
    fromNodeId: str
    toNodeId: str
    latencyMs: float
    timestampMs: int
    fromNodePosition: Position
    toNodePosition: Position
    distanceKm: float
    fromNodeBufferState: Optional[Dict[str, int]] = None
    routingDecisionInfo: Optional[Dict[str, Any]] = None

class ServiceType(str, Enum):
    VIDEO_STREAM = "VIDEO_STREAM"
    AUDIO_CALL = "AUDIO_CALL"
    IMAGE_TRANSFER = "IMAGE_TRANSFER"
    TEXT_MESSAGE = "TEXT_MESSAGE"
    FILE_TRANSFER = "FILE_TRANSFER"

@dataclass
class ServiceQoS:
    serviceType: ServiceType
    defaultPriority: int
    maxLatencyMs: float
    maxJitterMs: float
    minBandwidthMbps: float
    maxLossRate: float

@dataclass
class Packet:
    packetId: str
    sourceUserId: str
    destinationUserId: str
    stationSource: str
    stationDest: str
    type: str
    acknowledgedPacketId: Optional[str]
    timeSentFromSourceMs: int
    payloadDataBase64: str
    payloadSizeByte: int
    serviceQoS: ServiceQoS
    TTL: int
    currentHoldingNodeId: str
    nextHopNodeId: str
    pathHistory: List[str]
    hopRecords: List[HopRecord] = field(default_factory=list)
    accumulatedDelayMs: float = 0.0
    priorityLevel: int = 1
    isUseRL: bool = False
    maxAcceptableLatencyMs: float = 150.0
    maxAcceptableLossRate: float = 0.01
    dropped: bool = False
    dropReason: Optional[str] = None
    analysisData: Optional[Dict[str, Any]] = None
    
    def get_decoded_payload(self) -> str:
        try:
            return base64.b64decode(self.payloadDataBase64).decode('utf-8')
        except Exception:
            return "[Error decoding Base64]"

    def to_json(self) -> str:
        """Serializes the Packet object to a JSON string."""
        # asdict correctly handles nested dataclasses and enums (as strings)
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, json_data: Union[str, bytes]) -> 'Packet':
        """Deserializes a Packet object from a JSON string or bytes."""
        data: Dict[str, Any] = json.loads(json_data)
        
        # <<< CHANGE: Remove the redundant top-level serviceType if it exists
        if 'serviceType' in data:
            del data['serviceType']
            
        # <<< CHANGE: Make deserialization more robust by explicitly handling the Enum
        # 1. Convert the serviceType string inside the QoS object back to an Enum member
        if 'serviceQoS' in data and 'serviceType' in data['serviceQoS']:
            data['serviceQoS']['serviceType'] = ServiceType(data['serviceQoS']['serviceType'])
        
        # 2. Reconstruct the ServiceQoS object
        data['serviceQoS'] = ServiceQoS(**data['serviceQoS'])
        
        # 3. Reconstruct the list of HopRecord objects
        reconstructed_hops = []
        for hr_data in data.get('hopRecords', []):
            hr_data['fromNodePosition'] = Position(**hr_data['fromNodePosition'])
            hr_data['toNodePosition'] = Position(**hr_data['toNodePosition'])
            reconstructed_hops.append(HopRecord(**hr_data))
        data['hopRecords'] = reconstructed_hops
        
        return cls(**data)

def get_qos_profile(service_type: ServiceType) -> ServiceQoS:
    """Returns a pre-configured ServiceQoS object based on the service type."""
    PROFILES = {
        ServiceType.VIDEO_STREAM: {"defaultPriority": 1, "maxLatencyMs": 150.0, "maxJitterMs": 30.0, "minBandwidthMbps": 5.0, "maxLossRate": 0.01},
        ServiceType.AUDIO_CALL: {"defaultPriority": 2, "maxLatencyMs": 80.0, "maxJitterMs": 10.0, "minBandwidthMbps": 0.5, "maxLossRate": 0.005},
        ServiceType.IMAGE_TRANSFER: {"defaultPriority": 3, "maxLatencyMs": 500.0, "maxJitterMs": 100.0, "minBandwidthMbps": 1.0, "maxLossRate": 0.02},
        ServiceType.FILE_TRANSFER: {"defaultPriority": 4, "maxLatencyMs": 2000.0, "maxJitterMs": 500.0, "minBandwidthMbps": 2.0, "maxLossRate": 0.05},
        ServiceType.TEXT_MESSAGE: {"defaultPriority": 5, "maxLatencyMs": 1000.0, "maxJitterMs": 200.0, "minBandwidthMbps": 0.1, "maxLossRate": 0.01}
    }
    profile_params = PROFILES.get(service_type)
    if not profile_params:
        raise ValueError(f"No QoS profile found for service type: {service_type}")
    return ServiceQoS(serviceType=service_type, **profile_params)

# --- Example usage block remains the same, but the sample JSON needs a small adjustment ---
if __name__ == '__main__':
    # <<< CHANGE: The sample JSON no longer needs the redundant top-level "serviceType"
    sample_json_updated = """
    {
        "packetId": "PKT-001",
        "sourceUserId": "USER_A",
        "destinationUserId": "USER_B",
        "stationSource": "GS-01",
        "stationDest": "GS-A5",
        "type": "DATA",
        "acknowledgedPacketId": null,
        "timeSentFromSourceMs": 1739512300000,
        "payloadDataBase64": "SGVsbG8gU0FHU0lOIENsdWIh",
        "payloadSizeByte": 512,
        "serviceQoS": {
            "serviceType": "VIDEO_STREAM",
            "defaultPriority": 1,
            "maxLatencyMs": 150.0,
            "maxJitterMs": 30.0,
            "minBandwidthMbps": 5.0,
            "maxLossRate": 0.01
        },
        "TTL": 6,
        "currentHoldingNodeId": "LEO-001",
        "nextHopNodeId": "MEO-002",
        "pathHistory": ["GS-01", "LEO-001"],
        "hopRecords": [
            {
                "fromNodeId": "GS-01",
                "toNodeId": "LEO-001",
                "latencyMs": 4.8,
                "timestampMs": 1739512304800,
                "fromNodePosition": {"latitude": 21.0285, "longitude": 105.8542, "altitude": 0.0},
                "toNodePosition": {"latitude": 21.5, "longitude": 106.2, "altitude": 550.0},
                "distanceKm": 552.4,
                "fromNodeBufferState": {"used": 128, "total": 4096},
                "routingDecisionInfo": {"algorithm": "shortest_path"}
            }
        ],
        "accumulatedDelayMs": 4.8,
        "priorityLevel": 1,
        "isUseRL": false,
        "maxAcceptableLatencyMs": 150.0,
        "maxAcceptableLossRate": 0.01,
        "dropped": false,
        "dropReason": null,
        "analysisData": null
    }
    """
    
    try:
        packet_obj = Packet.from_json(sample_json_updated)
        
        print("--- Deserialization Successful ---")
        print(f"Packet ID: {packet_obj.packetId}")
        # Now we access the service type from the single source of truth
        print(f"Service Type: {packet_obj.serviceQoS.serviceType.name}")
        print(f"Decoded Payload: {packet_obj.get_decoded_payload()}")

        json_output = packet_obj.to_json()
        print("\n--- Serialization Successful ---")
        print(json_output)

    except Exception as e:
        print(f"\nAn error occurred: {e}")