import random
import logging
import time
from typing import Dict, List, Optional

from .api_instance import APIInstance
from .routing_strategy import RoutingStrategy

logger = logging.getLogger(__name__)

class InstanceRouter:
    """Selects API instances based on different routing strategies."""
    
    def __init__(self, routing_strategy: RoutingStrategy = RoutingStrategy.FAILOVER):
        """
        Initialize the instance router.
        
        Args:
            routing_strategy: Strategy for selecting instances
        """
        self.routing_strategy = routing_strategy
        self.round_robin_index = 0
    
    def select_instance(self, 
                       instances: Dict[str, APIInstance], 
                       required_tokens: int, 
                       model_name: Optional[str] = None) -> Optional[APIInstance]:
        """
        Select an API instance based on the configured routing strategy and model support.
        
        Args:
            instances: Dictionary of available API instances
            required_tokens: Estimated number of tokens required for the request
            model_name: The model name requested (optional)
            
        Returns:
            Selected instance or None if no suitable instance is available
        """
        # Filter healthy instances with enough capacity
        available_instances = [
            instance for instance in instances.values()
            if (instance.status == "healthy" or 
                (instance.status == "rate_limited" and time.time() >= (instance.rate_limited_until or 0)))
            and (instance.current_tpm + required_tokens) <= instance.max_tpm
            and (instance.max_input_tokens == 0 or required_tokens <= instance.max_input_tokens)
        ]
        
        # If a model name is provided, filter instances that support this model
        if model_name:
            # Normalize the model name
            normalized_model = model_name.lower().split(':')[0]
            
            # Filter instances that explicitly support this model
            model_instances = [
                instance for instance in available_instances
                if (
                    # Instance has this model in its supported_models list
                    normalized_model in [m.lower() for m in instance.supported_models] or
                    # Instance has a deployment mapping for this model
                    normalized_model in instance.model_deployments or
                    # Instance has no model restrictions (empty supported_models means it supports all models)
                    not instance.supported_models
                )
            ]
            
            # If we found instances that support this model, use them
            if model_instances:
                available_instances = model_instances
                logger.debug(f"Found {len(model_instances)} instances supporting model '{model_name}'")
            else:
                logger.warning(f"No instances explicitly support model '{model_name}', falling back to all available instances")
        
        if not available_instances:
            logger.warning(f"No healthy instances available with enough TPM capacity for {required_tokens} tokens")
            # Try to find any instance that's not in error state but still respects token limits
            available_instances = [
                instance for instance in instances.values()
                if instance.status != "error"
                and (instance.current_tpm + required_tokens) <= instance.max_tpm
                and (instance.max_input_tokens == 0 or required_tokens <= instance.max_input_tokens)
            ]
            
            if not available_instances:
                logger.error("No available instances found")
                return None
        
        # If we only have one instance, return it
        if len(available_instances) == 1:
            return available_instances[0]
        
        # Select instance based on routing strategy
        if self.routing_strategy == RoutingStrategy.PRIORITY:
            # Sort by priority (lower is higher priority)
            available_instances.sort(key=lambda x: x.priority)
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin
            self.round_robin_index = (self.round_robin_index + 1) % len(available_instances)
            return available_instances[self.round_robin_index]
            
        elif self.routing_strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection
            total_weight = sum(instance.weight for instance in available_instances)
            if total_weight == 0:
                # If all weights are 0, use simple round-robin
                self.round_robin_index = (self.round_robin_index + 1) % len(available_instances)
                return available_instances[self.round_robin_index]
            
            # Weighted random selection
            r = random.uniform(0, total_weight)
            upto = 0
            for instance in available_instances:
                upto += instance.weight
                if upto >= r:
                    return instance
            
            # Fallback to first instance
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            # Sort by current TPM usage (lower is better)
            available_instances.sort(key=lambda x: x.current_tpm)
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.FAILOVER:
            # Sort by priority (lower is higher priority) for failover
            available_instances.sort(key=lambda x: x.priority)
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.MODEL_SPECIFIC:
            # For model-specific routing, we've already filtered instances by model support above
            # Now sort by priority within those instances
            available_instances.sort(key=lambda x: x.priority)
            
            # If we have a model name, log which instance we're using for it
            if model_name:
                logger.info(f"Using model-specific routing: selected instance {available_instances[0].name} for model {model_name}")
            
            return available_instances[0]
            
        # Default to first instance
        return available_instances[0]
    
    def get_available_instances_for_model(self,
                                         instances: Dict[str, APIInstance],
                                         model_name: str,
                                         provider_type: Optional[str] = None) -> List[APIInstance]:
        """
        Get all available instances that support a specific model and provider type.
        
        Args:
            instances: Dictionary of available API instances
            model_name: The model name to check for
            provider_type: Optional provider type to filter instances
            
        Returns:
            List of instances supporting the model
        """
        # Filter out instances in error state and by provider type if specified
        available_instances = [
            instance for instance in instances.values() 
            if instance.status != "error" and
               (provider_type is None or instance.provider_type == provider_type)
        ]
        
        if not model_name:
            return available_instances
            
        # Normalize the model name
        normalized_model = model_name.lower().split(':')[0]
        
        # Filter instances that support this model
        model_instances = [
            instance for instance in available_instances
            if (
                # Instance has this model in its supported_models list
                normalized_model in [m.lower() for m in instance.supported_models] or
                # Instance has a deployment mapping for this model
                normalized_model in instance.model_deployments or
                # Instance has no model restrictions (empty supported_models means it supports all models)
                not instance.supported_models
            )
        ]
        
        # Sort by priority (lower is higher priority)
        model_instances.sort(key=lambda x: x.priority)
        
        return model_instances 