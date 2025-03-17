"""
Legacy accessor for backward compatibility.

This module provides a compatibility layer for the transition period
between the old mixed config/state architecture and the new separated architecture.
"""

import logging
from typing import Dict, Optional, Any
from app.models.instance import InstanceConfig, InstanceState
from app.storage.config_store import ConfigStore
from app.storage.state_store import StateStore

logger = logging.getLogger(__name__)

class LegacyInstanceAccessor:
    """
    Provides backward compatible access to instance data.
    
    This class serves as a bridge between the old API that didn't separate
    configuration from state and the new architecture that does.
    """
    
    def __init__(self, config_store: ConfigStore, state_store: StateStore):
        """
        Initialize the legacy accessor.
        
        Args:
            config_store: The configuration store
            state_store: The state store
        """
        self.config_store = config_store
        self.state_store = state_store
    
    def get_instance(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get combined instance data like in the old system.
        
        Args:
            name: Instance name
            
        Returns:
            A dictionary with combined config and state properties, 
            or None if instance doesn't exist
        """
        config = self.config_store.get_config(name)
        state = self.state_store.get_state(name)
        
        if not config:
            return None
            
        # Combine into a single dict for backward compatibility
        combined = config.dict()
        if state:
            # Add state properties, but don't overwrite config
            for k, v in state.dict().items():
                if k != 'name':  # Skip name to avoid duplication
                    combined[k] = v
                    
        return combined
        
    def get_all_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all instances in the old format.
        
        Returns:
            Dictionary of instance name to combined properties
        """
        result = {}
        
        # Get all configs and states
        configs = self.config_store.get_all_configs()
        states = self.state_store.get_all_states()
        
        # Start with all configured instances
        for name, config in configs.items():
            combined = config.dict()
            
            # Add state if available
            state = states.get(name)
            if state:
                for k, v in state.dict().items():
                    if k != 'name':  # Skip name to avoid duplication
                        combined[k] = v
                        
            result[name] = combined
            
        return result
        
    def update_instance(self, name: str, **kwargs) -> bool:
        """
        Update instance properties as in the old system.
        
        Args:
            name: Instance name
            **kwargs: Properties to update
            
        Returns:
            True if the instance was updated successfully
        """
        # Get existing data
        config = self.config_store.get_config(name)
        
        # Determine which properties belong to config vs state
        config_props = set([f.name for f in InstanceConfig.__fields__.values()])
        state_props = set([f.name for f in InstanceState.__fields__.values()])
        
        # Split the kwargs into config and state updates
        config_updates = {}
        state_updates = {}
        
        for key, value in kwargs.items():
            if key in config_props:
                config_updates[key] = value
            elif key in state_props:
                state_updates[key] = value
            else:
                logger.warning(f"Property '{key}' not recognized as config or state, ignoring")
        
        # Update config if needed
        if config_updates:
            if not config:
                # Create new config if it doesn't exist
                if 'name' not in config_updates:
                    config_updates['name'] = name
                try:
                    new_config = InstanceConfig(**config_updates)
                    self.config_store.add_config(new_config)
                except Exception as e:
                    logger.error(f"Error creating config for {name}: {e}")
                    return False
            else:
                # Update existing config
                updated_config_dict = config.dict()
                updated_config_dict.update(config_updates)
                try:
                    updated_config = InstanceConfig(**updated_config_dict)
                    self.config_store.add_config(updated_config)
                except Exception as e:
                    logger.error(f"Error updating config for {name}: {e}")
                    return False
        
        # Update state if needed
        if state_updates:
            try:
                self.state_store.update_state(name, **state_updates)
            except Exception as e:
                logger.error(f"Error updating state for {name}: {e}")
                return False
        
        return True 