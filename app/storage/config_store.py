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
            logger.debug(f"Explicitly reloading configs from file {self.config_file}")
            # Clear current configs to ensure a fresh load
            self.configs = {}
            # Try to load from JSON file first
            self._load_configs()
            
            # If no configs loaded, try to load from YAML
            if not self.configs:
                logger.info("No configs loaded from JSON, attempting to load from YAML")
                self._load_from_yaml()
            
            logger.debug(f"Reload complete. Loaded {len(self.configs)} instance configurations")
    
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
                # Ensure absolute path to file
                if not os.path.isabs(self.config_file):
                    self.config_file = os.path.abspath(self.config_file)
                    
                # Create directory if it doesn't exist
                file_dir = os.path.dirname(self.config_file)
                if file_dir and not os.path.exists(file_dir):
                    logger.info(f"Creating directory for config file: {file_dir}")
                    os.makedirs(file_dir, exist_ok=True)
                
                # Log the file path
                logger.debug(f"Saving configurations to file: {self.config_file}")
                
                # Convert to dict of dicts for JSON serialization
                data = {name: config.dict() for name, config in self.configs.items()}
                
                # Write to a temporary file first
                temp_file = f"{self.config_file}.tmp"
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data is written to disk
                
                # Verify the temporary file exists and has content
                if not os.path.exists(temp_file):
                    logger.error(f"Failed to create temporary file: {temp_file}")
                    return
                    
                temp_file_size = os.path.getsize(temp_file)
                logger.debug(f"Temporary file created: {temp_file}, size: {temp_file_size} bytes")
                
                # Rename to the actual file
                os.replace(temp_file, self.config_file)
                
                logger.info(f"Saved {len(self.configs)} instance configurations to {self.config_file}")
            except Exception as e:
                logger.error(f"Error saving configurations to {self.config_file}: {e}", exc_info=True)
    
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
        # Always reload configs from file to ensure fresh data across workers
        self.reload()
        return self.configs
    
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
                # Check if we need to convert a relative path to absolute
                if not os.path.isabs(self.config_file):
                    abs_path = os.path.abspath(self.config_file)
                    logger.debug(f"Converting relative path '{self.config_file}' to absolute: '{abs_path}'")
                    self.config_file = abs_path
                
                logger.debug(f"Adding config for instance '{config.name}' to file: {self.config_file}")
                
                # Add to in-memory dictionary
                self.configs[config.name] = config
                
                # Explicitly save to disk
                self._save_configs()
                
                # Verify the file was saved
                if os.path.exists(self.config_file):
                    file_size = os.path.getsize(self.config_file)
                    logger.debug(f"Config file saved successfully. Size: {file_size} bytes")
                else:
                    logger.error(f"Failed to save config file. File does not exist: {self.config_file}")
                    
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
    
    def remove_config(self, name: str) -> bool:
        """
        Remove a configuration. Alias for delete_config.
        
        Args:
            name: Name of the instance to remove
            
        Returns:
            True if the configuration was removed, False otherwise
        """
        return self.delete_config(name)
    
    def _load_from_yaml(self):
        """Load initial configurations from YAML if JSON file is empty or missing."""
        try:
            from app.config import config_loader
            
            # Load configuration directly from YAML
            config = config_loader.load_config()
            
            if hasattr(config, 'instances'):
                for instance_config in config.instances:
                    try:
                        # Convert instance config to dictionary
                        instance_data = instance_config.dict()
                        config = InstanceConfig(**instance_data)
                        self.configs[instance_config.name] = config
                    except Exception as e:
                        logger.error(f"Error loading instance {instance_config.name} from YAML: {e}")
                
                # If we loaded configs from YAML, save them to JSON for persistence
                if self.configs:
                    self._save_configs()
                    logger.info(f"Initialized {len(self.configs)} instances from YAML configuration")
        except Exception as e:
            logger.error(f"Error loading configurations from YAML: {e}") 
