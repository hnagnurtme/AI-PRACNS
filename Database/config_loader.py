"""
Configuration Loader
Đọc và load configuration từ YAML files
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Load configuration from YAML files"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize config loader
        
        Args:
            config_file: Path to config file. If None, auto-detect based on environment
        """
        self.config_file = config_file or self._get_config_file()
        self.config = self._load_config()
    
    def _get_config_file(self) -> str:
        """Get config file based on environment"""
        env = os.getenv('ENVIRONMENT', 'dev').lower()
        base_dir = Path(__file__).parent
        
        if env == 'production' or env == 'prod':
            config_file = base_dir / 'config.pro.yaml'
        else:
            config_file = base_dir / 'config.dev.yaml'
        
        return str(config_file)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Expand environment variables
            config = self._expand_env_vars(config)
            
            return config or {}
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_file} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {str(e)}, using defaults")
            return self._get_default_config()
    
    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand environment variables in config"""
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Extract env var name
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'scheduler': {
                'update_interval_seconds': 10,
                'enable_logging': True,
                'log_file': 'node_update.log',
                'log_level': 'INFO'
            },
            'orbital_mechanics': {
                'earth': {
                    'radius_km': 6371.0,
                    'radius_m': 6371000.0
                },
                'leo': {
                    'period_seconds': 5400,
                    'speed_m_per_s': 7800,
                    'default_semi_major_axis_km': 6928.0
                },
                'meo': {
                    'period_seconds': 43200,
                    'speed_m_per_s': 3900,
                    'default_semi_major_axis_km': 26562.0
                },
                'geo': {
                    'period_seconds': 86400,
                    'speed_m_per_s': 3070,
                    'default_semi_major_axis_km': 42164.0
                }
            },
            'node_update': {
                'update_node_types': ['LEO_SATELLITE', 'MEO_SATELLITE', 'GEO_SATELLITE'],
                'only_operational': True
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value by key path (e.g., 'scheduler.update_interval_seconds')
        
        Args:
            key_path: Dot-separated key path
            default: Default value if key not found
        
        Returns:
            Config value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_orbital_config(self, satellite_type: str) -> Dict[str, Any]:
        """
        Get orbital configuration for specific satellite type
        
        Args:
            satellite_type: 'leo', 'meo', or 'geo'
        
        Returns:
            Orbital configuration dict
        """
        return self.get(f'orbital_mechanics.{satellite_type.lower()}', {})
    
    def get_earth_constants(self) -> Dict[str, float]:
        """Get Earth constants"""
        return self.get('orbital_mechanics.earth', {
            'radius_km': 6371.0,
            'radius_m': 6371000.0
        })

# Global config instance
_config_loader = None

def get_config(config_file: str = None) -> ConfigLoader:
    """Get global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_file)
    return _config_loader

