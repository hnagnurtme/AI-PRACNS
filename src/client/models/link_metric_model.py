# models/link_metric_model.py

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LinkMetric:
    """
    Represents the quality and metrics of a communication link
    between two nodes in the network.
    """
    sourceNodeId: str
    destinationNodeId: str

    distanceKm: float
    maxBandwidthMbps: float
    currentAvailableBandwidthMbps: float
    latencyMs: float
    packetLossRate: float
    linkAttenuationDb: float  # Signal loss over the link

    # A composite score (0-100) representing overall link quality.
    linkScore: float

    isLinkActive: bool = True
    
    # Use Unix timestamp for efficient querying and storage.
    lastUpdated: int = field(default_factory=lambda: int(datetime.utcnow().timestamp()))

    def to_dict(self):
        """Converts the dataclass instance to a dictionary."""
        return {
            "sourceNodeId": self.sourceNodeId,
            "destinationNodeId": self.destinationNodeId,
            "distanceKm": self.distanceKm,
            "maxBandwidthMbps": self.maxBandwidthMbps,
            "currentAvailableBandwidthMbps": self.currentAvailableBandwidthMbps,
            "latencyMs": self.latencyMs,
            "packetLossRate": self.packetLossRate,
            "linkAttenuationDb": self.linkAttenuationDb,
            "linkScore": self.linkScore,
            "isLinkActive": self.isLinkActive,
            # Store lastUpdated as a readable ISO string and a timestamp
            "lastUpdated_iso": datetime.fromtimestamp(self.lastUpdated).isoformat() + "Z",
            "lastUpdated_ts": self.lastUpdated
        }