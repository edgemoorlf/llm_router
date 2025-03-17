"""
Configuration storage implementation.

This module provides the ConfigStore class for managing instance configurations
that are defined by operators and change infrequently.
"""

import json
import os
import logging
import threading
from typing import Dict, Optional, List
from app.models.instance import InstanceConfig

logger = logging.getLogger(__name__)

class ConfigStore:
    """
    Stores and manages instance configurations.
    
    This class handles the persistence and retrieval of instance configurations,
    which represent settings defined by operators.
    """
    
    def __init__(self, config_file: str = "instance_configs.json"):
        """
        Initialize the configuration store.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.configs: Dict[str, InstanceConfig] = {}
        self.file_lock = threading.RLock()  # Lock for thread-safe access
        
        # Try to load from JSON file first
        self._load_configs()
        
        # If no configs loaded, try to load from YAML
        if not self.configs:
            self._load_from_yaml()
    
    def reload(self):
        """
        Reload configurations from the file.
        """
        with self.file_lock:
            self._load_configs()
    
    def _load_configs(self):
        """Load configurations from storage."""
        with self.file_lock:
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, "r") as f:
                        data = json.load(f)
                        for name, config_data in data.items():
                            self.configs[name] = InstanceConfig(**config_data)
                    logger.info(f"Loaded {len(self.configs)} instance configurations from {self.config_file}")
                except Exception as e:
                    logger.error(f"Error loading configurations from {self.config_file}: {e}")
            else:
                logger.warning(f"Configuration file {self.config_file} not found. Starting with empty configuration.")
    
    def _save_configs(self):
        """Persist configurations to storage."""
        with self.file_lock:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_file) or '.', exist_ok=True)
                
                # Convert to dict of dicts for JSON serialization
                data = {name: config.dict() for name, config in self.configs.items()}
                
                # Write to a temporary file first
                temp_file = f"{self.config_file}.tmp"
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2)
                
                # Rename to the actual file
                os.replace(temp_file, self.config_file)
                
                logger.info(f"Saved {len(self.configs)} instance configurations to {self.config_file}")
            except Exception as e:
                logger.error(f"Error saving configurations to {self.config_file}: {e}")
    
    def get_config(self, name: str) -> Optional[InstanceConfig]:
        """
        Get a specific instance configuration.
        
        Args:
            name: Name of the instance
            
        Returns:
            The instance configuration or None if not found
        """
        with self.file_lock:
            return self.configs.get(name)
    
    def get_all_configs(self) -> Dict[str, InstanceConfig]:
        """
        Get all instance configurations.
        
        Returns:
            Dictionary of instance name to configuration
        """
        with self.file_lock:
            # Return a copy to avoid external modifications
            return dict(self.configs)
    
    def add_config(self, config: InstanceConfig) -> bool:
        """
        Add or update an instance configuration.
        
        Args:
            config: The instance configuration to add
            
        Returns:
            True if successful, False otherwise
        """
        with self.file_lock:
            try:
                self.configs[config.name] = config
                self._save_configs()
                return True
            except Exception as e:
                logger.error(f"Error adding configuration for {config.name}: {e}")
                return False
    
    def delete_config(self, name: str) -> bool:
        """
        Delete an instance configuration.
        
        Args:
            name: Name of the instance to delete
            
        Returns:
            True if the instance was deleted, False if it wasn't found or an error occurred
        """
        with self.file_lock:
            if name in self.configs:
                try:
                    del self.configs[name]
                    self._save_configs()
                    return True
                except Exception as e:
                    logger.error(f"Error deleting configuration for {name}: {e}")
                    return False
            return False
    
    def _load_from_yaml(self):
        """Load initial configurations from YAML if JSON file is empty or missing."""
        try:
            from app.config.config_hierarchy import config_hierarchy
            yaml_config = config_hierarchy.get_configuration()
            
            if yaml_config and 'instances' in yaml_config:
                for name, instance_data in yaml_config['instances'].items():
                    try:
                        # Ensure name is in the data
                        instance_data['name'] = name
                        config = InstanceConfig(**instance_data)
                        self.configs[name] = config
                    except Exception as e:
                        logger.error(f"Error loading instance {name} from YAML: {e}")
                
                # If we loaded configs from YAML, save them to JSON for persistence
                if self.configs:
                    self._save_configs()
                    logger.info(f"Initialized {len(self.configs)} instances from YAML configuration")
        except Exception as e:
            logger.error(f"Error loading configurations from YAML: {e}") 