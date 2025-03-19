"""Service for selecting appropriate instances based on model requirements and health status."""
import logging
import random
import time
from typing import Dict, List, Any, Optional, Set

from app.instance.instance_context import instance_manager

logger = logging.getLogger(__name__)

class InstanceSelector:
    """Service for selecting Azure and other provider instances based on model requirements."""
    
    def __init__(self):
        """Initialize the instance selector service."""
        logger.info("Initialized Instance Selector service")
        
    def _get_instances(self, provider_type: Optional[str] = None) -> list:
        """
        Get all available instances, optionally filtered by provider type.
        
        Args:
            provider_type: Optional filter for provider type (e.g., "azure")
            
        Returns:
            List of instance configurations combined with their states
        """
        configs = instance_manager.get_all_configs()
        states = instance_manager.get_all_states()
        
        instances = []
        for name, config in configs.items():
            # Filter by provider type if specified
            if provider_type and config.provider_type != provider_type:
                continue
                
            # Create instance dict from config
            instance = config.dict()
            
            # Add state information if available
            state = states.get(name)
            if state:
                # Add state fields to instance dict
                state_dict = state.dict()
                state_dict.pop('name', None)  # Remove duplicate name field
                instance.update(state_dict)
            else:
                logger.warning(f"No state found for instance {name}, state might be None")
                # Set a default status to avoid None status issues
                instance["status"] = "healthy"
                
            instances.append(instance)
            
        return instances
    
    def get_instances_for_model(self, model_name: str, provider_type: Optional[str] = None) -> list:
        """
        Get instances that support a specific model, optionally filtered by provider type.
        
        Args:
            model_name: The model name to check support for
            provider_type: Optional filter for provider type
            
        Returns:
            List of instances that support the model
        """
        instances = self._get_instances(provider_type)
        
        return [
            instance for instance in instances
            if not instance.get("supported_models") or  # If no supported_models list, assume all supported
               model_name.lower() in [m.lower() for m in instance.get("supported_models", [])]
        ]
        
    def select_instances_for_model(
        self, 
        model_name: str, 
        required_tokens: int = 0, 
        provider_type: Optional[str] = None,
        exclude_instance_names: Optional[Set[str]] = None
    ) -> list:
        """
        Select instances suitable for a specific model, ordered by suitability.
        
        Args:
            model_name: The model name to use
            required_tokens: Required tokens for the request (for capacity checking)
            provider_type: Optional provider type filter
            exclude_instance_names: Optional set of instance names to exclude (e.g., after failures)
            
        Returns:
            List of instances ordered by suitability (primary instance first, then fallbacks)
        """
        excluded = set() if exclude_instance_names is None else exclude_instance_names
        logger.debug(f"Selecting instances for model '{model_name}', provider_type={provider_type}, excluded={excluded}")
            
        # Get instances that support this model
        instances = self.get_instances_for_model(model_name, provider_type)
        logger.debug(f"Found {len(instances)} instances supporting model '{model_name}'")
        
        for idx, instance in enumerate(instances):
            logger.debug(f"Instance {idx+1}/{len(instances)}: {instance.get('name')} - Status: {instance.get('status')}")
        
        # Filter out excluded instances
        instances = [
            instance for instance in instances 
            if instance.get("name") not in excluded
        ]
        
        if not instances:
            logger.debug(f"No instances available for model '{model_name}' after exclusions")
            return []
            
        # Select the best instance based on capacity and health
        primary_instance = self._select_primary_instance(instances, required_tokens, model_name)
        
        # If no instance has capacity (including TPM and rate limits), return empty list
        if not primary_instance:
            logger.debug(f"No instances available with capacity for {required_tokens} tokens for model '{model_name}'")
            return []
                
        return [primary_instance]
        
    def _select_primary_instance(self, instances: list, tokens: int, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Select primary instance based on routing strategy with strict model compatibility checking.
        
        Args:
            instances: List of available instances that support the model
            tokens: Required tokens for the request
            model_name: The model name to use
            
        Returns:
            Selected instance or None if no suitable instance is found
        """
        # Ensure we have at least one instance
        if not instances:
            logger.debug(f"No instances provided to select from for model '{model_name}'")
            return None
            
        # Find the best instance from the given list of pre-filtered instances
        # Filter instances with sufficient capacity
        eligible_instances = []
        
        for instance in instances:
            # Check if instance has capacity for the requested tokens
            if self._check_instance_capacity(instance, tokens):
                eligible_instances.append(instance)
                
        if not eligible_instances:
            logger.debug(f"No eligible instances with capacity for {tokens} tokens for model '{model_name}'")
            return None
            
        # Group instances by priority (lower is better)
        priority_groups = {}
        for instance in eligible_instances:
            priority = instance.get("priority", 100)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(instance)
            
        # Sort priorities (lowest first)
        sorted_priorities = sorted(priority_groups.keys())
        
        # Get the highest priority group and randomly select an instance
        highest_priority_group = priority_groups[sorted_priorities[0]]
        
        # Randomly select an instance from the highest priority group
        selected_instance = random.choice(highest_priority_group)
        logger.debug(f"Selected instance {selected_instance.get('name')} for model '{model_name}' from {len(highest_priority_group)} eligible instances with priority {sorted_priorities[0]}")
        
        # Add original model name for consistent instance selection in fallbacks
        selected_instance["_original_model_for_selection"] = model_name
        
        return selected_instance
        
    def _check_instance_capacity(self, instance: Dict[str, Any], required_tokens: int) -> bool:
        """
        Check if an instance has capacity for the requested tokens.
        
        Args:
            instance: Instance configuration and state
            required_tokens: Required tokens for the request
            
        Returns:
            True if instance has capacity, False otherwise
        """
        instance_name = instance.get("name", "")
        
        # Get status, defaulting to healthy if missing or None
        status = instance.get("status")
        if status is None:
            logger.warning(f"Instance {instance_name} has None status, defaulting to healthy")
            instance["status"] = "healthy"
            status = "healthy"
        
        # Proactively check if rate limit has expired
        if status == "rate_limited":
            rate_limited_until = instance.get("rate_limited_until")
            if rate_limited_until and time.time() >= rate_limited_until:
                logger.info(f"Rate limit for instance {instance_name} has expired in selector, marking as healthy")
                instance["status"] = "healthy"
                status = "healthy"
                # Also update in the actual instance manager
                instance_manager.mark_healthy(instance_name)
            else:
                remaining = int(rate_limited_until - time.time()) if rate_limited_until else 0
                logger.debug(f"Instance {instance_name} still rate limited for {remaining} more seconds")
            
        # Skip if not healthy
        if status != "healthy":
            logger.debug(f"Instance {instance_name} skipped: status is {status}")
            return False
            
        # Check TPM limit
        current_tpm = instance.get("current_tpm", 0)
        max_tpm = instance.get("max_tpm", 0)
        if max_tpm > 0 and current_tpm >= max_tpm * 0.95:  # Only check if max_tpm is set
            logger.debug(f"Instance {instance_name} skipped: approaching TPM limit ({current_tpm}/{max_tpm})")
            return False
            
        # Check rate limit
        allowed, retry_after = instance_manager.check_rate_limit(instance_name, required_tokens)
        if not allowed:
            logger.debug(f"Instance {instance_name} skipped: rate limited (retry after {retry_after}s)")
            # Mark as rate limited if not already
            if status != "rate_limited":
                instance_manager.mark_rate_limited(instance_name, int(retry_after) if retry_after else None)
            return False
            
        return True

# Create a singleton instance
instance_selector = InstanceSelector() 