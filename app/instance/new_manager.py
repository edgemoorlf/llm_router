"""
New instance manager implementation with separated config and state.

This module provides a new implementation of the InstanceManager class
that uses the separated configuration and state architecture.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union

from app.models.instance import InstanceConfig, InstanceState
from app.storage.config_store import ConfigStore
from app.storage.redis_state_store import RedisStateStore
from app.storage.legacy import LegacyInstanceAccessor

logger = logging.getLogger(__name__)

class NewInstanceManager:
    """
    Manages OpenAI-compatible API instances with separated config and state.
    
    This class handles the loading, tracking, and selection of API instances
    for forwarding requests to OpenAI-compatible APIs. It maintains a clear
    separation between configuration (defined by operators) and state (derived
    from runtime operation).
    """
    
    def __init__(self, 
                config_file: str = "instance_configs.json",
                redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize the instance manager.
        
        Args:
            config_file: Path to the configuration file
            redis_url: Redis connection URL for state storage
        """
        self.config_store = ConfigStore(config_file)
        self.state_store = RedisStateStore(redis_url=redis_url)
        self.legacy = LegacyInstanceAccessor(self.config_store, self.state_store)
        self.router = None  # Will be initialized later
        
        # Reset transient states on startup
        self.state_store.reset_states_on_startup()
        
    def reload_config(self):
        """Reload configurations from storage."""
        # Reinitialize the config store to reload from disk
        self.config_store = ConfigStore(self.config_store.config_file)
        
        # Make sure all configured instances have a state
        for name in self.config_store.get_all_configs():
            if not self.state_store.get_state(name):
                self.state_store.update_state(name, **create_instance_state(name).dict())
                
    def get_instance(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get an instance by name.
        
        Args:
            name: Name of the instance
            
        Returns:
            Combined instance data or None if not found
        """
        return self.legacy.get_instance(name)
        
    def get_all_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all instances.
        
        Returns:
            Dictionary of instance name to combined properties
        """
        return self.legacy.get_all_instances()
        
    def get_instance_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all instances.
        
        Returns:
            Dictionary with instance statistics
        """
        states = self.state_store.get_all_states()
        
        # Filter to only include instances that have configuration
        configs = self.config_store.get_all_configs()
        instance_stats = [
            state.dict() for name, state in states.items()
            if name in configs
        ]
        
        return {
            "status": "success",
            "timestamp": int(time.time()),
            "instances": instance_stats,
            "count": len(instance_stats)
        }
        
    def update_instance_state(self, name: str, **kwargs) -> bool:
        """
        Update state for a specific instance.
        
        Args:
            name: Name of the instance
            **kwargs: State properties to update
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the instance exists in config
        if not self.config_store.get_config(name):
            logger.warning(f"Cannot update state for non-existent instance: {name}")
            return False
            
        try:
            self.state_store.update_state(name, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error updating state for {name}: {e}")
            return False
            
    def clear_instance_error(self, name: str) -> bool:
        """
        Clear error state for an instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            True if successful, False otherwise
        """
        return self.state_store.clear_error_state(name)
        
    def record_request(self, name: str, success: bool, tokens: int = 0,
                      latency_ms: Optional[float] = None,
                      error: Optional[str] = None,
                      status_code: Optional[int] = None) -> bool:
        """
        Record a request to update instance metrics.
        
        Args:
            name: Name of the instance
            success: Whether the request was successful
            tokens: Number of tokens processed
            latency_ms: Request latency in milliseconds
            error: Error message if request failed
            status_code: HTTP status code if request failed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state_store.record_request(
                name=name,
                success=success,
                tokens=tokens,
                latency_ms=latency_ms,
                error=error,
                status_code=status_code
            )
            return True
        except Exception as e:
            logger.error(f"Error recording request for {name}: {e}")
            return False

    def get_all_configs(self) -> Dict[str, InstanceConfig]:
        """
        Get all instance configurations.
        
        Returns:
            Dictionary of instance name to configuration
        """
        return self.config_store.get_all_configs()

    def get_instance_config(self, name: str) -> Optional[InstanceConfig]:
        """
        Get configuration for a specific instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            The instance configuration or None if not found
        """
        return self.config_store.get_config(name)

    def get_instance_state(self, name: str) -> Optional[InstanceState]:
        """
        Get current state for a specific instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            The instance state or None if not found
        """
        return self.state_store.get_state(name)

    def get_all_states(self) -> Dict[str, InstanceState]:
        """
        Get all instance states.
        
        Returns:
            Dictionary of instance name to state
        """
        return self.state_store.get_all_states() 