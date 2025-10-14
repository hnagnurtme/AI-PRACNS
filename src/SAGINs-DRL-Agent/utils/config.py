# utils/config.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingConfig:
    # Data config
    mongo_host: str = "localhost"
    mongo_port: int = 27017
    db_name: str = 'sagins-network'
    cache_duration_minutes: int = 10
    db_username: str = 'user'
    db_password: str = 'password123'
    db_auth_source: str = 'admin'
    connection_string: str = f"mongodb://{db_username}:{db_password}@{mongo_host}:{mongo_port}/?authSource={db_auth_source}"
    
    # Training hyperparameters
    total_episodes: int = 20000
    warmup_steps: int = 2000  # Fix: Sử dụng warmup_steps thay vì warm_up_steps
    batch_size: int = 64
    target_update_freq: int = 1000
    save_interval: int = 500
    
    # DQN parameters
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # QoS requirements 
    default_qos: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_qos is None:
            self.default_qos = {
                "serviceType": "VIDEO_STREAMING",
                "maxLatencyMs": 50.0,
                "minBandwidthMbps": 500.0,
                "maxLossRate": 0.02
            }