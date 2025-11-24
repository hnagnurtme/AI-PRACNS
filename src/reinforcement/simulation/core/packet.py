from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

class RoutingAlgorithm(str, Enum):
    DIJKSTRA = "Dijkstra"
    REINFORCEMENT_LEARNING = "ReinforcementLearning"

@dataclass
class Position:
    latitude: float
    longitude: float
    altitude: float

@dataclass
class BufferState:
    queue_size: int
    bandwidth_utilization: float  # 0.0 - 1.0

@dataclass
class RoutingDecisionInfo:
    algorithm: RoutingAlgorithm
    metric: Optional[str] = None
    reward: Optional[float] = None

@dataclass
class HopRecord:
    from_node_id: str
    to_node_id: str
    latency_ms: float
    timestamp_ms: float
    distance_km: float
    packet_loss_rate: float
    from_node_position: Optional[Position] = None
    to_node_position: Optional[Position] = None
    from_node_buffer_state: Optional[BufferState] = None
    routing_decision_info: Optional[RoutingDecisionInfo] = None
    scenario_type: Optional[str] = None
    node_load_percent: Optional[float] = None
    drop_reason_details: Optional[str] = None

@dataclass
class QoS:
    service_type: str
    default_priority: int
    max_latency_ms: float
    max_jitter_ms: float
    min_bandwidth_mbps: float
    max_loss_rate: float

@dataclass
class AnalysisData:
    avg_latency: float
    avg_distance_km: float
    route_success_rate: float
    total_distance_km: float
    total_latency_ms: float

@dataclass
class Packet:
    packet_id: str
    source_user_id: str
    destination_user_id: str
    station_source: str
    station_dest: str
    type: str
    time_sent_from_source_ms: float
    payload_data_base64: str
    payload_size_byte: int
    service_qos: QoS
    current_holding_node_id: str
    next_hop_node_id: str
    priority_level: int
    max_acceptable_latency_ms: float
    max_acceptable_loss_rate: float
    analysis_data: AnalysisData
    use_rl: bool
    ttl: int
    
    # Optional fields với giá trị mặc định
    acknowledged_packet_id: Optional[str] = None
    path_history: List[str] = field(default_factory=list)
    hop_records: List[HopRecord] = field(default_factory=list)
    accumulated_delay_ms: float = 0.0
    dropped: bool = False
    drop_reason: Optional[str] = None