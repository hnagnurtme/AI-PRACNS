"""
Configuration file for SAGIN RL Backend
Loads configuration from YAML file with environment variable support
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_yaml_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    if config_path is None:
        # Try to find config file
        env = os.getenv('ENVIRONMENT', 'dev').lower()
        base_dir = Path(__file__).parent
        if env == 'production' or env == 'prod':
            config_path = base_dir / 'config.pro.yaml'
        else:
            config_path = base_dir / 'config.dev.yaml'
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config or {}
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
    
    return {}

# Load YAML config
_yaml_config = load_yaml_config()

class Config:
    """Application configuration"""
    # MongoDB
    MONGODB_URI = os.getenv('MONGODB_URI', _yaml_config.get('mongodb', {}).get('uri', 'mongodb://admin:password@localhost:27017/aiprancs?authSource=admin'))
    DB_NAME = os.getenv('DB_NAME', _yaml_config.get('mongodb', {}).get('database', 'aiprancs'))
    
    # Flask
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # RL Training - DQN với ReplayBuffer
    RL_MODEL_PATH = os.getenv('RL_MODEL_PATH', _yaml_config.get('rl_agent', {}).get('model_path', './models/rl_agent'))
    RL_ALGORITHM = os.getenv('RL_ALGORITHM', _yaml_config.get('rl_agent', {}).get('algorithm', 'DQN'))
    RL_ARCHITECTURE = os.getenv('RL_ARCHITECTURE', _yaml_config.get('rl_agent', {}).get('architecture', 'dueling'))
    RL_LEARNING_RATE = float(os.getenv('RL_LEARNING_RATE', _yaml_config.get('rl_agent', {}).get('dqn', {}).get('learning_rate', 0.0001)))
    RL_BATCH_SIZE = int(os.getenv('RL_BATCH_SIZE', _yaml_config.get('rl_agent', {}).get('dqn', {}).get('batch_size', 32)))
    RL_BUFFER_SIZE = int(os.getenv('RL_BUFFER_SIZE', _yaml_config.get('rl_agent', {}).get('dqn', {}).get('buffer_size', 100000)))
    RL_LEARNING_STARTS = int(os.getenv('RL_LEARNING_STARTS', _yaml_config.get('rl_agent', {}).get('dqn', {}).get('learning_starts', 1000)))
    RL_TARGET_UPDATE_INTERVAL = int(os.getenv('RL_TARGET_UPDATE_INTERVAL', _yaml_config.get('rl_agent', {}).get('dqn', {}).get('target_update_interval', 1000)))
    
    # Bootstrap Learning
    RL_BOOTSTRAP_HEADS = int(os.getenv('RL_BOOTSTRAP_HEADS', _yaml_config.get('rl_agent', {}).get('dqn', {}).get('bootstrap', {}).get('n_bootstrap_heads', 10)))
    RL_BOOTSTRAP_PROBABILITY = float(os.getenv('RL_BOOTSTRAP_PROBABILITY', _yaml_config.get('rl_agent', {}).get('dqn', {}).get('bootstrap', {}).get('bootstrap_probability', 0.5)))
    
    # SAGIN Network Parameters
    MAX_SATELLITE_NODES = int(os.getenv('MAX_SATELLITE_NODES', _yaml_config.get('network', {}).get('max_satellite_nodes', 10)))
    MAX_AERIAL_NODES = int(os.getenv('MAX_AERIAL_NODES', _yaml_config.get('network', {}).get('max_aerial_nodes', 20)))
    MAX_GROUND_NODES = int(os.getenv('MAX_GROUND_NODES', _yaml_config.get('network', {}).get('max_ground_nodes', 50)))
    MAX_BANDWIDTH = float(os.getenv('MAX_BANDWIDTH', _yaml_config.get('network', {}).get('max_bandwidth_mhz', 1000.0)))
    MAX_POWER = float(os.getenv('MAX_POWER', _yaml_config.get('network', {}).get('max_power_watts', 100.0)))
    
    # Training Parameters
    MAX_EPISODES = int(os.getenv('MAX_EPISODES', _yaml_config.get('training', {}).get('max_episodes', 1000)))
    MAX_STEPS_PER_EPISODE = int(os.getenv('MAX_STEPS_PER_EPISODE', _yaml_config.get('training', {}).get('max_steps_per_episode', 100)))
    EVAL_FREQUENCY = int(os.getenv('EVAL_FREQUENCY', _yaml_config.get('training', {}).get('eval_frequency', 100)))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', _yaml_config.get('logging', {}).get('level', 'INFO'))
    TENSORBOARD_LOG_DIR = os.getenv('TENSORBOARD_LOG_DIR', _yaml_config.get('rl_agent', {}).get('tensorboard_log_dir', './logs/tensorboard'))
    
    # Get full YAML config for advanced usage
    @staticmethod
    def get_yaml_config() -> dict:
        """Get full YAML configuration"""
        return _yaml_config

