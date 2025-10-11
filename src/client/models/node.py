# client_app/models/node.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Node:
    id: str
    type: str
    label: str
    pos: Tuple[int, int]
    ip: str = "0.0.0.0"
    port: int = 0
    processing_ms: float = 1.0

@dataclass
class LinkMetric:
    source: str
    target: str
    distance_km: float
    base_latency_ms: float
    is_active: bool = True