"""
Instance router implementation.

This module provides the InstanceRouter class for selecting appropriate instances
to handle API requests based on various routing strategies.
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from app.models.instance import InstanceConfig, InstanceState, create_instance_state

logger = logging.getLogger(__name__)

class InstanceRouter:
    """
    Routes requests to appropriate instances based on various strategies.
    
    This class implements different routing strategies for selecting instances
    to handle API requests, taking into account instance configuration, current
    state, and request requirements.
    """
    
    def __init__(self, instance_manager):
        """
        Initialize the instance router.
        
        Args:
            instance_manager: The instance manager to use for routing
        """
        self.instance_manager = instance_manager
        
    def select_instance(self, required_tokens: Optional[int] = None,
                       model_name: Optional[str] = None,
                       strategy: str = "random") -> Optional[str]:
        """
        Select an instance to handle a request.
        
        Args:
            required_tokens: Number of tokens required for the request
            model_name: Model name required for the request
            strategy: Routing strategy to use (random, least_busy, round_robin)
            
        Returns:
            Name of the selected instance or None if no suitable instance found
        """
        # Get all configs and states
        configs = self.instance_manager.get_all_configs()
        states = self.instance_manager.get_all_states()
        
        candidates = []
        for name, config in configs.items():
            # Skip if instance is disabled
            if not config.enabled:
                continue
                
            # Get state if available, otherwise create a default state
            state = states.get(name)
            if not state:
                state = create_instance_state(name)
                
            # Skip if instance is unhealthy
            if not state.is_healthy:
                continue
                
            # Skip if model is specified and not supported
            if model_name and model_name not in config.supported_models:
                continue
                
            # Skip if required tokens exceed capacity
            if required_tokens and required_tokens > config.max_tpm:
                continue
                
            # This instance is a candidate
            candidates.append((name, config, state))
            
        if not candidates:
            logger.warning(f"No suitable instances found for request (model={model_name}, tokens={required_tokens})")
            return None
            
        # Apply routing strategy
        if strategy == "random":
            selected = random.choice(candidates)
            return selected[0]
        elif strategy == "least_busy":
            # Sort by number of active requests
            candidates.sort(key=lambda x: x[2].current_tpm)
            return candidates[0][0]
        elif strategy == "round_robin":
            # Just take the first one for now (we'll implement proper round robin later)
            return candidates[0][0]
        else:
            logger.warning(f"Unknown routing strategy: {strategy}, using random")
            selected = random.choice(candidates)
            return selected[0]
            
    def get_instance_for_model(self, model_name: str) -> Optional[str]:
        """
        Find an instance that supports the specified model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Name of a suitable instance or None if not found
        """
        return self.select_instance(model_name=model_name)
        
    def get_instance_capacity(self, name: str) -> Tuple[bool, Optional[int]]:
        """
        Check if an instance has capacity for more requests.
        
        Args:
            name: Name of the instance
            
        Returns:
            Tuple of (has_capacity, available_slots)
            If instance doesn't exist, returns (False, None)
        """
        config = self.instance_manager.get_instance_config(name)
        state = self.instance_manager.get_instance_state(name)
        
        if not config or not state:
            return (False, None)
            
        # Check if instance is enabled and healthy
        if not config.enabled or not state.is_healthy:
            return (False, 0)
            
        # Calculate available capacity
        max_concurrent = config.max_parallel_requests
        current = state.active_requests
        
        available = max(0, max_concurrent - current)
        has_capacity = available > 0
        
        return (has_capacity, available)
        
    def get_supported_models(self) -> Dict[str, List[str]]:
        """
        Get a mapping of models to instances that support them.
        
        Returns:
            Dictionary mapping model names to lists of instance names
        """
        configs = self.instance_manager.get_all_configs()
        states = self.instance_manager.get_all_states()
        
        model_map = {}
        
        for name, config in configs.items():
            # Skip if instance doesn't exist in states
            if name not in states:
                continue
                
            state = states[name]
            
            # Skip if instance is disabled or unhealthy
            if not config.enabled or not state.is_healthy:
                continue
                
            # Add instance to each supported model
            for model in config.supported_models:
                if model not in model_map:
                    model_map[model] = []
                model_map[model].append(name)
                
        return model_map 