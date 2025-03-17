"""
Factory for creating instance managers and routers.

This module provides a factory class for creating instance managers and routers,
allowing for a smooth transition between the old and new implementations.
"""

import logging
import os
from typing import Optional, Tuple, Dict, Any

# Import both old and new implementations
from app.instance.manager import InstanceManager as OldInstanceManager
from app.instance.new_manager import NewInstanceManager
from app.instance.new_router import InstanceRouter

logger = logging.getLogger(__name__)

class InstanceFactory:
    """
    Factory for creating instance managers and routers.
    
    This class provides methods for creating instance managers and routers,
    allowing for a smooth transition between the old and new implementations.
    """
    
    # Singleton instance
    _instance = None
    
    # Current manager and router instances
    _manager = None
    _router = None
    
    # Configuration
    _use_new_implementation = False
    _config_file = "instances.json"
    _config_file_new = "instance_configs.json"
    _state_file_new = "instance_states.json"
    
    @classmethod
    def initialize(cls, 
                  use_new_implementation: bool = False,
                  config_file: Optional[str] = None,
                  config_file_new: Optional[str] = None,
                  state_file_new: Optional[str] = None) -> None:
        """
        Initialize the factory with configuration.
        
        Args:
            use_new_implementation: Whether to use the new implementation
            config_file: Path to the old configuration file
            config_file_new: Path to the new configuration file
            state_file_new: Path to the new state file
        """
        cls._use_new_implementation = use_new_implementation
        
        if config_file:
            cls._config_file = config_file
        
        if config_file_new:
            cls._config_file_new = config_file_new
            
        if state_file_new:
            cls._state_file_new = state_file_new
            
        # Reset instances to force recreation
        cls._manager = None
        cls._router = None
        
        logger.info(f"Initialized InstanceFactory with use_new_implementation={use_new_implementation}")
    
    @classmethod
    def get_instance(cls) -> 'InstanceFactory':
        """
        Get the singleton instance of the factory.
        
        Returns:
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = InstanceFactory()
        return cls._instance
    
    @classmethod
    def get_manager(cls) -> Any:
        """
        Get the current instance manager.
        
        Returns:
            The current instance manager (old or new implementation)
        """
        if cls._manager is None:
            if cls._use_new_implementation:
                cls._manager = NewInstanceManager(
                    config_file=cls._config_file_new,
                    state_file=cls._state_file_new
                )
                # Initialize router
                cls._router = InstanceRouter(cls._manager)
                # Set router reference in manager
                cls._manager.router = cls._router
            else:
                # Create old manager without config_file parameter
                cls._manager = OldInstanceManager()
                
        return cls._manager
    
    @classmethod
    def get_router(cls) -> Any:
        """
        Get the current instance router.
        
        Returns:
            The current instance router
        """
        # Ensure manager is created first
        if cls._manager is None:
            cls.get_manager()
            
        return cls._router
    
    @classmethod
    def reload_config(cls) -> None:
        """
        Reload configuration for the current manager.
        """
        manager = cls.get_manager()
        manager.reload_config()
        
    @classmethod
    def is_using_new_implementation(cls) -> bool:
        """
        Check if the factory is using the new implementation.
        
        Returns:
            True if using the new implementation, False otherwise
        """
        return cls._use_new_implementation
        
    @classmethod
    def migrate_to_new_implementation(cls) -> Dict[str, Any]:
        """
        Migrate from old to new implementation.
        
        This method will:
        1. Create a new manager with the new implementation
        2. Copy all instances from the old manager to the new one
        3. Switch to using the new implementation
        
        Returns:
            Dictionary with migration results
        """
        # Get the old manager - create without config_file parameter
        old_manager = OldInstanceManager()
        
        # Create a new manager
        new_manager = NewInstanceManager(
            config_file=cls._config_file_new,
            state_file=cls._state_file_new
        )
        
        # Get all instances from the old manager
        old_instances = old_manager.get_all_instances()
        
        # Migrate each instance
        migrated = 0
        failed = 0
        
        for name, instance in old_instances.items():
            try:
                # Extract configuration properties
                from app.models.instance import InstanceConfig
                
                # Create a new config
                config_dict = {
                    "name": name,
                    "provider_type": instance.get("provider_type", "azure"),
                    "api_key": instance.get("api_key", ""),
                    "api_base": instance.get("api_base", ""),
                    "api_version": instance.get("api_version", ""),
                    "enabled": instance.get("enabled", True),
                    "supported_models": instance.get("supported_models", []),
                    "max_tokens": instance.get("max_tokens", 4000),
                    "max_parallel_requests": instance.get("max_parallel_requests", 10),
                    "timeout_seconds": instance.get("timeout_seconds", 30),
                    "deployment_name": instance.get("deployment_name", ""),
                    "resource_group": instance.get("resource_group", ""),
                    "subscription_id": instance.get("subscription_id", ""),
                    "tags": instance.get("tags", {})
                }
                
                config = InstanceConfig(**config_dict)
                
                # Add to new manager
                success = new_manager.add_instance(config)
                
                if success:
                    # Update state properties
                    new_manager.update_instance_state(
                        name,
                        is_healthy=instance.get("is_healthy", True),
                        last_error=instance.get("last_error", ""),
                        last_error_time=instance.get("last_error_time", 0),
                        total_requests=instance.get("total_requests", 0),
                        successful_requests=instance.get("successful_requests", 0),
                        failed_requests=instance.get("failed_requests", 0),
                        total_tokens=instance.get("total_tokens", 0),
                        active_requests=instance.get("active_requests", 0)
                    )
                    migrated += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error migrating instance {name}: {e}")
                failed += 1
                
        # Switch to using the new implementation
        cls._use_new_implementation = True
        cls._manager = new_manager
        cls._router = InstanceRouter(new_manager)
        new_manager.router = cls._router
        
        return {
            "status": "success",
            "migrated": migrated,
            "failed": failed,
            "total": len(old_instances)
        } 