"""Configuration hierarchy management for Azure OpenAI Proxy."""
import os
import logging
from typing import Dict, Any, List, Optional, Callable
import copy
from dataclasses import dataclass

from app.config import config_loader
from app.instance.api_instance import APIInstance

logger = logging.getLogger(__name__)

@dataclass
class ConfigSource:
    """Configuration source with priority information."""
    name: str
    priority: int
    description: str
    get_config: Callable[[], Dict[str, Any]]
    timestamp: Optional[float] = None

class ConfigurationHierarchy:
    """
    Manages the hierarchy of configuration sources.
    
    Implements the priority order:
    1. Database (highest priority)
    2. State File
    3. YAML Configuration
    4. Default Values (lowest priority)
    """
    
    def __init__(self):
        """Initialize the configuration hierarchy."""
        self.sources: Dict[str, ConfigSource] = {}
        self.register_default_sources()
        
    def register_default_sources(self):
        """Register the default configuration sources."""
        # Add database source (highest priority)
        self.register_source(
            name="database",
            priority=100,
            description="SQLite database persistent storage",
            get_config=self._get_db_config
        )
        
        # Add state file source
        self.register_source(
            name="state_file",
            priority=80,
            description="Temporary state file",
            get_config=self._get_state_file_config
        )
        
        # Add YAML configuration source
        self.register_source(
            name="yaml",
            priority=60,
            description="YAML configuration files",
            get_config=self._get_yaml_config
        )
        
        # Add default values source (lowest priority)
        self.register_source(
            name="defaults",
            priority=0,
            description="Default values",
            get_config=self._get_default_config
        )
        
    def register_source(self, name: str, priority: int, description: str, 
                       get_config: Callable[[], Dict[str, Any]]):
        """
        Register a configuration source.
        
        Args:
            name: Source name
            priority: Source priority (higher = more important)
            description: Source description
            get_config: Function to get configuration from this source
        """
        self.sources[name] = ConfigSource(
            name=name,
            priority=priority,
            description=description,
            get_config=get_config
        )
        
    def _get_db_config(self) -> Dict[str, Any]:
        """Get configuration from SQLite database."""
        # This is just a stub - the actual implementation will be done
        # when the InstanceManager loads state from the SQLiteStateManager
        return {}
        
    def _get_state_file_config(self) -> Dict[str, Any]:
        """Get configuration from state file."""
        # This is just a stub - the actual implementation will be done
        # when the InstanceManager loads state from the FileStateManager
        return {}
        
    def _get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration from YAML files."""
        try:
            config = config_loader.load_config()
            if not config or not hasattr(config, 'instances'):
                return {}
                
            # Convert InstanceConfig objects to dictionaries
            instances = {}
            for instance_config in config.instances:
                instances[instance_config.name] = {
                    'name': instance_config.name,
                    'provider_type': instance_config.provider_type,
                    'api_key': instance_config.api_key,
                    'api_base': instance_config.api_base,
                    'api_version': instance_config.api_version,
                    'proxy_url': instance_config.proxy_url,
                    'priority': instance_config.priority,
                    'weight': instance_config.weight,
                    'max_tpm': instance_config.max_tpm,
                    'max_input_tokens': instance_config.max_input_tokens,
                    'supported_models': instance_config.supported_models,
                    'model_deployments': instance_config.model_deployments
                }
                
            return {'instances': instances}
            
        except Exception as e:
            logger.error(f"Error loading YAML configuration: {str(e)}")
            return {}
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'instances': {},
            'routing': {
                'strategy': 'failover'
            },
            'monitoring': {
                'stats_window_minutes': 5
            }
        }
        
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get merged configuration from all sources according to priority.
        
        Returns:
            Merged configuration dictionary
        """
        # Start with empty configuration
        merged_config = {}
        
        # Sort sources by priority (highest first)
        sorted_sources = sorted(
            self.sources.values(),
            key=lambda s: s.priority,
            reverse=True
        )
        
        # Apply each configuration source
        for source in sorted_sources:
            try:
                config = source.get_config()
                self._deep_merge(merged_config, config)
                logger.debug(f"Applied configuration from {source.name} (priority: {source.priority})")
            except Exception as e:
                logger.error(f"Error applying configuration from {source.name}: {str(e)}")
                
        return merged_config
        
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge source into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._deep_merge(target[key], value)
            elif key not in target or value is not None:
                # Only override if target doesn't have the key or source value is not None
                target[key] = copy.deepcopy(value)
                
    def get_configuration_sources(self) -> List[Dict[str, Any]]:
        """
        Get information about configuration sources.
        
        Returns:
            List of configuration source dictionaries
        """
        sources = []
        for source in sorted(
            self.sources.values(),
            key=lambda s: s.priority,
            reverse=True
        ):
            sources.append({
                'name': source.name,
                'priority': source.priority,
                'description': source.description,
                'timestamp': source.timestamp
            })
            
        return sources
    
    def get_instance_source(self, instance_name: str) -> Dict[str, Any]:
        """
        Get information about which source each instance setting comes from.
        
        Args:
            instance_name: Name of the instance
            
        Returns:
            Dictionary mapping each setting to its source
        """
        result = {}
        instance_data = {}
        
        # Check each source for this instance
        for source in sorted(
            self.sources.values(),
            key=lambda s: s.priority,
            reverse=True
        ):
            try:
                config = source.get_config()
                instances = config.get('instances', {})
                if instance_name in instances:
                    instance = instances[instance_name]
                    for key, value in instance.items():
                        if key not in result and value is not None:
                            result[key] = source.name
                            instance_data[key] = value
            except Exception as e:
                logger.error(f"Error checking source {source.name} for instance {instance_name}: {str(e)}")
                
        return {
            'sources': result,
            'instance': instance_data
        }

# Create a singleton instance of the configuration hierarchy
config_hierarchy = ConfigurationHierarchy() 