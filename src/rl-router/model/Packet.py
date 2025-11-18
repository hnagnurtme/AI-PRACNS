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

    def __post_init__(self):
        if self.path_history is None:
            self.path_history = []
        if self.hop_records is None:
            self.hop_records = []

    def add_hop_record(self, hop_record: HopRecord):
        self.hop_records.append(hop_record)
        self.path_history.append(hop_record.to_node_id)
        self.accumulated_delay_ms += hop_record.latency_ms

    def to_dict(self):
        """Convert Packet to dictionary for MongoDB storage"""
        return {
            "packetId": self.packet_id,
            "sourceUserId": self.source_user_id,
            "destinationUserId": self.destination_user_id,
            "stationSource": self.station_source,
            "stationDest": self.station_dest,
            "type": self.type,
            "timeSentFromSourceMs": self.time_sent_from_source_ms,
            "payloadDataBase64": self.payload_data_base64,
            "payloadSizeByte": self.payload_size_byte,
            "serviceQoS": {
                "serviceType": self.service_qos.service_type,
                "defaultPriority": self.service_qos.default_priority,
                "maxLatencyMs": self.service_qos.max_latency_ms,
                "maxJitterMs": self.service_qos.max_jitter_ms,
                "minBandwidthMbps": self.service_qos.min_bandwidth_mbps,
                "maxLossRate": self.service_qos.max_loss_rate
            } if self.service_qos else None,
            "currentHoldingNodeId": self.current_holding_node_id,
            "nextHopNodeId": self.next_hop_node_id,
            "priorityLevel": self.priority_level,
            "maxAcceptableLatencyMs": self.max_acceptable_latency_ms,
            "maxAcceptableLossRate": self.max_acceptable_loss_rate,
            "analysisData": {
                "avgLatency": self.analysis_data.avg_latency,
                "avgDistanceKm": self.analysis_data.avg_distance_km,
                "routeSuccessRate": self.analysis_data.route_success_rate,
                "totalDistanceKm": self.analysis_data.total_distance_km,
                "totalLatencyMs": self.analysis_data.total_latency_ms
            } if self.analysis_data else None,
            "useRL": self.use_rl,
            "ttl": self.ttl,
            "acknowledgedPacketId": self.acknowledged_packet_id,
            "pathHistory": self.path_history,
            "hopRecords": [self._hop_record_to_dict(hr) for hr in self.hop_records],
            "accumulatedDelayMs": self.accumulated_delay_ms,
            "dropped": self.dropped,
            "dropReason": self.drop_reason
        }

    def _hop_record_to_dict(self, hop_record: HopRecord):
        """Convert HopRecord to dictionary"""
        return {
            "fromNodeId": hop_record.from_node_id,
            "toNodeId": hop_record.to_node_id,
            "latencyMs": hop_record.latency_ms,
            "timestampMs": hop_record.timestamp_ms,
            "distanceKm": hop_record.distance_km,
            "packetLossRate": hop_record.packet_loss_rate,
            "fromNodePosition": {
                "latitude": hop_record.from_node_position.latitude,
                "longitude": hop_record.from_node_position.longitude,
                "altitude": hop_record.from_node_position.altitude
            } if hop_record.from_node_position else None,
            "toNodePosition": {
                "latitude": hop_record.to_node_position.latitude,
                "longitude": hop_record.to_node_position.longitude,
                "altitude": hop_record.to_node_position.altitude
            } if hop_record.to_node_position else None,
            "fromNodeBufferState": {
                "queueSize": hop_record.from_node_buffer_state.queue_size,
                "bandwidthUtilization": hop_record.from_node_buffer_state.bandwidth_utilization
            } if hop_record.from_node_buffer_state else None,
            "routingDecisionInfo": {
                "algorithm": hop_record.routing_decision_info.algorithm.value if isinstance(hop_record.routing_decision_info.algorithm, RoutingAlgorithm) else hop_record.routing_decision_info.algorithm,
                "metric": hop_record.routing_decision_info.metric,
                "reward": hop_record.routing_decision_info.reward
            } if hop_record.routing_decision_info else None,
            "scenarioType": hop_record.scenario_type,
            "nodeLoadPercent": hop_record.node_load_percent,
            "dropReasonDetails": hop_record.drop_reason_details
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Packet':
        """Create Packet from dictionary"""
        # Handle nested QoS
        service_qos = None
        if data.get('serviceQoS'):
            qos_data = data['serviceQoS']
            service_qos = QoS(
                service_type=qos_data.get('serviceType', ''),
                default_priority=qos_data.get('defaultPriority', 0),
                max_latency_ms=qos_data.get('maxLatencyMs', 0.0),
                max_jitter_ms=qos_data.get('maxJitterMs', 0.0),
                min_bandwidth_mbps=qos_data.get('minBandwidthMbps', 0.0),
                max_loss_rate=qos_data.get('maxLossRate', 0.0)
            )

        # Handle nested AnalysisData
        analysis_data = None
        if data.get('analysisData'):
            ad_data = data['analysisData']
            analysis_data = AnalysisData(
                avg_latency=ad_data.get('avgLatency', 0.0),
                avg_distance_km=ad_data.get('avgDistanceKm', 0.0),
                route_success_rate=ad_data.get('routeSuccessRate', 0.0),
                total_distance_km=ad_data.get('totalDistanceKm', 0.0),
                total_latency_ms=ad_data.get('totalLatencyMs', 0.0)
            )

        # Handle hop records
        hop_records = []
        for hr_data in data.get('hopRecords', []):
            hop_records.append(cls._hop_record_from_dict(hr_data))

        return cls(
            packet_id=data.get('packetId', ''),
            source_user_id=data.get('sourceUserId', ''),
            destination_user_id=data.get('destinationUserId', ''),
            station_source=data.get('stationSource', ''),
            station_dest=data.get('stationDest', ''),
            type=data.get('type', ''),
            time_sent_from_source_ms=data.get('timeSentFromSourceMs', 0.0),
            payload_data_base64=data.get('payloadDataBase64', ''),
            payload_size_byte=data.get('payloadSizeByte', 0),
            service_qos=service_qos,
            current_holding_node_id=data.get('currentHoldingNodeId', ''),
            next_hop_node_id=data.get('nextHopNodeId', ''),
            priority_level=data.get('priorityLevel', 0),
            max_acceptable_latency_ms=data.get('maxAcceptableLatencyMs', 0.0),
            max_acceptable_loss_rate=data.get('maxAcceptableLossRate', 0.0),
            analysis_data=analysis_data,
            use_rl=data.get('useRL', False),
            ttl=data.get('ttl', 64),
            acknowledged_packet_id=data.get('acknowledgedPacketId'),
            path_history=data.get('pathHistory', []),
            hop_records=hop_records,
            accumulated_delay_ms=data.get('accumulatedDelayMs', 0.0),
            dropped=data.get('dropped', False),
            drop_reason=data.get('dropReason')
        )

    @staticmethod
    def _hop_record_from_dict(data: dict) -> HopRecord:
        """Convert dictionary to HopRecord"""
        from_pos = None
        if data.get('fromNodePosition'):
            pos = data['fromNodePosition']
            from_pos = Position(
                latitude=pos.get('latitude', 0.0),
                longitude=pos.get('longitude', 0.0),
                altitude=pos.get('altitude', 0.0)
            )

        to_pos = None
        if data.get('toNodePosition'):
            pos = data['toNodePosition']
            to_pos = Position(
                latitude=pos.get('latitude', 0.0),
                longitude=pos.get('longitude', 0.0),
                altitude=pos.get('altitude', 0.0)
            )

        buffer_state = None
        if data.get('fromNodeBufferState'):
            bs = data['fromNodeBufferState']
            buffer_state = BufferState(
                queue_size=bs.get('queueSize', 0),
                bandwidth_utilization=bs.get('bandwidthUtilization', 0.0)
            )

        routing_info = None
        if data.get('routingDecisionInfo'):
            ri = data['routingDecisionInfo']
            routing_info = RoutingDecisionInfo(
                algorithm=RoutingAlgorithm(ri.get('algorithm', 'Dijkstra')),
                metric=ri.get('metric'),
                reward=ri.get('reward')
            )

        return HopRecord(
            from_node_id=data.get('fromNodeId', ''),
            to_node_id=data.get('toNodeId', ''),
            latency_ms=data.get('latencyMs', 0.0),
            timestamp_ms=data.get('timestampMs', 0.0),
            distance_km=data.get('distanceKm', 0.0),
            packet_loss_rate=data.get('packetLossRate', 0.0),
            from_node_position=from_pos,
            to_node_position=to_pos,
            from_node_buffer_state=buffer_state,
            routing_decision_info=routing_info,
            scenario_type=data.get('scenarioType'),
            node_load_percent=data.get('nodeLoadPercent'),
            drop_reason_details=data.get('dropReasonDetails')
        )