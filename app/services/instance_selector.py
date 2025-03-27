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
        # Get configs first (no Redis call)
        configs = instance_manager.get_all_instance_configs()
        
        # Filter by provider type if needed
        if provider_type:
            configs = [config for config in configs if config.provider_type == provider_type]
        
        # Convert configs to dictionaries
        instances = []
        for config in configs:
            # Create instance dict from config
            instance = config.dict()
            
            # Add a default status to avoid None status issues
            # We'll load actual states later when needed
            instance["status"] = "healthy"
            instances.append(instance)
        
        return instances

    def _get_states_for_filtered_instances(self, instances: list) -> list:
        """
        Get states only for instances that have passed initial filtering.
        
        This optimizes Redis calls by only fetching states for instances
        that are already known to match basic filtering criteria.
        
        Args:
            instances: List of instance dictionaries (from configs only)
        
        Returns:
            List of instances with state data added
        """
        # Get state information for each instance
        instance_names = [instance["name"] for instance in instances]
        
        # Use more efficient batched state retrieval
        if instance_names:
            states = {}
            # Get states in batches
            for i in range(0, len(instance_names), 20):
                batch = instance_names[i:i+20]
                for name in batch:
                    state = instance_manager.get_instance_state(name)
                    if state:
                        states[name] = state
        else:
            states = {}
        
        # Update instances with state information
        for instance in instances:
            name = instance["name"]
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
        
        # Filter by model support (still using only config data, no Redis)
        return [
            instance for instance in instances
            if self._is_model_compatible(instance, model_name)
        ]
        
    def _is_model_compatible(self, instance: Dict[str, Any], model: str) -> bool:
        """
        Check if an instance supports a given model.
        
        Args:
            instance: The instance configuration to check
            model: The model name to check support for
            
        Returns:
            True if the instance supports the model, False otherwise
        """
        # Normalize the model name
        model_lower = model.lower()
        
        # If no supported_models list, assume all models are supported
        supported_models = instance.get("supported_models", [])
        if not supported_models:
            return True
            
        # Check if the model is in the supported_models list
        for supported_model in supported_models:
            if model_lower == supported_model.lower():
                return True
                
        return False

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
            
        # Get instances that support this model (config only - no Redis calls yet)
        instances = self.get_instances_for_model(model_name, provider_type)
        logger.debug(f"Found {len(instances)} instances supporting model '{model_name}'")
        
        # Filter out excluded instances
        instances = [
            instance for instance in instances 
            if instance.get("name") not in excluded
        ]
        
        if not instances:
            logger.debug(f"No instances available for model '{model_name}' after exclusions")
            return []
            
        # Now that we have filtered the instances by model and other criteria,
        # get the states only for these filtered instances
        instances = self._get_states_for_filtered_instances(instances)
        
        # Get all eligible instances based on capacity and health
        eligible_instances = self._get_eligible_instances(instances, required_tokens, model_name)
        
        # If no instances have capacity (including TPM and rate limits), return empty list
        if not eligible_instances:
            logger.debug(f"No instances available with capacity for {required_tokens} tokens for model '{model_name}'")
            return []
                
        return eligible_instances
        
    def _get_eligible_instances(self, instances: list, tokens: int, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all eligible instances based on capacity and health, ordered by priority.
        
        Args:
            instances: List of available instances that support the model
            tokens: Required tokens for the request
            model_name: The model name to use
            
        Returns:
            List of eligible instances ordered by priority and randomized within priority groups
        """
        # Ensure we have at least one instance
        if not instances:
            logger.debug(f"No instances provided to select from for model '{model_name}'")
            return []
            
        # Find eligible instances from the given list of pre-filtered instances
        eligible_instances = []
        
        # First filter out instances with non-healthy status
        # This avoids unnecessary capacity checks for instances we won't use anyway
        healthy_instances = []
        for instance in instances:
            status = instance.get("status")
            instance_name = instance.get("name", "unknown")
            
            # Proactively check if rate limit has expired for rate-limited instances
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
            
            if status is None:
                logger.warning(f"Instance {instance_name} has None status, defaulting to healthy")
                instance["status"] = "healthy"
                healthy_instances.append(instance)
            elif status == "healthy":
                healthy_instances.append(instance)
            else:
                logger.debug(f"Instance {instance_name} skipped: status is {status}")
                
        # If no healthy instances, return empty list early
        if not healthy_instances:
            logger.debug(f"No healthy instances available for model '{model_name}'")
            return []
            
        # For remaining healthy instances, sync rate limiter data once
        for instance in healthy_instances:
            instance_name = instance.get("name", "")
            # Sync rate limiter data to ensure current_tpm is accurate
            instance_manager.sync_rate_limiter_to_state(instance_name)
            
            # Update the instance state in our local dictionary with the fresh data
            current_state = instance_manager.get_instance_state(instance_name)
            if current_state:
                instance["current_tpm"] = current_state.current_tpm
        
        # Now check capacity for the healthy instances
        for instance in healthy_instances:
            # Check if instance has capacity for the requested tokens
            if self._check_instance_capacity(instance, tokens):
                # Add original model name for consistent instance selection in fallbacks
                instance["_original_model_for_selection"] = model_name
                eligible_instances.append(instance)
                
        if not eligible_instances:
            logger.debug(f"No eligible instances with capacity for {tokens} tokens for model '{model_name}'")
            return []
            
        # Group instances by priority (lower is better)
        priority_groups = {}
        for instance in eligible_instances:
            priority = instance.get("priority", 100)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(instance)
            
        # Sort priorities (lowest first)
        sorted_priorities = sorted(priority_groups.keys())
        
        # Build final list of instances ordered by priority
        ordered_instances = []
        for priority in sorted_priorities:
            # Randomly shuffle instances within the same priority group
            group = priority_groups[priority]
            random.shuffle(group)
            ordered_instances.extend(group)
        
        logger.debug(f"Returning {len(ordered_instances)} eligible instances for model '{model_name}' in priority order")
        return ordered_instances
        
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
        # Get eligible instances ordered by priority and randomized within priority groups
        ordered_instances = self._get_eligible_instances(instances, tokens, model_name)
        
        # Return the first instance (highest priority) or None if no eligible instances
        return ordered_instances[0] if ordered_instances else None
        
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
        
        # We no longer need to check status here since we've already filtered by status
        
        # Check TPM limit
        current_tpm = instance.get("current_tpm", 0)
        max_tpm = instance.get("max_tpm", 0)
        if max_tpm > 0 and current_tpm >= max_tpm * 0.95:  # Only check if max_tpm is set
            tpm_usage_percent = (current_tpm / max_tpm) * 100
            logger.debug(f"Instance {instance_name} skipped: approaching TPM limit ({current_tpm}/{max_tpm}, {tpm_usage_percent:.1f}%)")
            return False
            
        # Check rate limit
        allowed = instance_manager.check_rate_limit(instance_name, required_tokens)
        if not allowed:
            logger.debug(f"Instance {instance_name} skipped: rate limited")
            # Mark as rate limited - just a single call, no redundant state fetching
            # Get current token usage for better debugging
            current_usage = instance.get("current_tpm", 0)
            max_usage = instance.get("max_tpm", 0)
            logger.debug(f"Marking instance {instance_name} as rate limited. Current usage: {current_usage}/{max_usage} tokens")
            instance_manager.mark_rate_limited(instance_name, None)
            return False
            
        logger.debug(f"Instance {instance_name} has capacity for {required_tokens} tokens (TPM: {current_tpm}/{max_tpm})")
        return True

    def select_instance_for_request(self, model: str, tokens: int, provider_type: str = None) -> Optional[str]:
        """
        Select the best instance for a request based on model compatibility and capacity.
        
        Args:
            model: The model name to use
            tokens: Required tokens for the request
            provider_type: Optional provider type filter
            
        Returns:
            Name of the selected instance or None if no suitable instance is found
        """
        logger.debug(f"Selecting instance for model {model}, tokens {tokens}, provider_type {provider_type}")
        
        # Get instances that support this model (config only - no Redis calls yet)
        instances = self.get_instances_for_model(model, provider_type)
        if not instances:
            logger.warning(f"No instances found that support model '{model}'")
            return None
            
        # Now get state information only for filtered instances
        instances = self._get_states_for_filtered_instances(instances)
        
        # Get eligible instances ordered by priority
        eligible_instances = self._get_eligible_instances(instances, tokens, model)
        
        if not eligible_instances:
            logger.warning(f"No instances with capacity found for model '{model}' and {tokens} tokens")
            return None
        
        # Return the name of the first instance (highest priority)
        selected_instance = eligible_instances[0]
        logger.info(f"Selected instance {selected_instance['name']} for model {model}, tokens {tokens}")
        return selected_instance['name']

# Create a singleton instance
instance_selector = InstanceSelector() 