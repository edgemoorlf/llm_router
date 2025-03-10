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
            and (instance.instance_stats.current_tpm + required_tokens) <= instance.max_tpm
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
                and (instance.instance_stats.current_tpm + required_tokens) <= instance.max_tpm
                and (instance.max_input_tokens == 0 or required_tokens <= instance.max_input_tokens)
            ]
            
            if not available_instances:
                logger.error("No available instances found")
                return None
        
        # If we only have one instance, return it
        if len(available_instances) == 1:
            return available_instances[0]
        
        # Apply routing strategy to select an instance
        logger.debug(f"Selecting instance using {self.routing_strategy} strategy from {len(available_instances)} available instances")
        
        # Select instance based on routing strategy
        if self.routing_strategy == RoutingStrategy.PRIORITY:
            # Sort by priority (lower is higher priority)
            sorted_instances = sorted(available_instances, key=lambda x: x.priority)
            selected = sorted_instances[0]
            logger.debug(f"PRIORITY strategy selected instance {selected.name} with priority {selected.priority}")
            return selected
            
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            # Ensure index is valid after instances may have been added/removed
            if self.round_robin_index >= len(available_instances):
                self.round_robin_index = 0
                
            # Simple round-robin
            selected = available_instances[self.round_robin_index]
            self.round_robin_index = (self.round_robin_index + 1) % len(available_instances)
            logger.debug(f"ROUND_ROBIN strategy selected instance {selected.name}, next index: {self.round_robin_index}")
            return selected
            
        elif self.routing_strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection
            total_weight = sum(instance.weight for instance in available_instances)
            
            if total_weight == 0:
                # If all weights are 0, use simple round-robin
                if self.round_robin_index >= len(available_instances):
                    self.round_robin_index = 0
                selected = available_instances[self.round_robin_index]
                self.round_robin_index = (self.round_robin_index + 1) % len(available_instances)
                logger.debug(f"WEIGHTED strategy using round-robin fallback (all weights 0) selected instance {selected.name}")
                return selected
            
            # Weighted random selection
            r = random.uniform(0, total_weight)
            upto = 0
            for instance in available_instances:
                upto += instance.weight
                if upto >= r:
                    logger.debug(f"WEIGHTED strategy selected instance {instance.name} with weight {instance.weight}/{total_weight}")
                    return instance
            
            # Should never reach here if weights are positive
            logger.warning(f"WEIGHTED strategy selection error, falling back to first instance")
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            # Sort by current TPM usage (lower is better)
            sorted_instances = sorted(available_instances, key=lambda x: x.instance_stats.current_tpm)
            selected = sorted_instances[0]
            logger.debug(f"LEAST_LOADED strategy selected instance {selected.name} with {selected.instance_stats.current_tpm} TPM")
            return selected
            
        elif self.routing_strategy == RoutingStrategy.FAILOVER:
            # Sort by priority (lower is higher priority) for failover
            sorted_instances = sorted(available_instances, key=lambda x: x.priority)
            selected = sorted_instances[0]
            logger.debug(f"FAILOVER strategy selected instance {selected.name} with priority {selected.priority}")
            return selected
            
        elif self.routing_strategy == RoutingStrategy.MODEL_SPECIFIC:
            # For model-specific routing, we've already filtered instances by model support above
            # Now sort by priority within those instances
            sorted_instances = sorted(available_instances, key=lambda x: x.priority)
            selected = sorted_instances[0]
            
            # If we have a model name, log which instance we're using for it
            if model_name:
                logger.info(f"MODEL_SPECIFIC routing: selected instance {selected.name} for model {model_name}")
            
            return selected
        
        # If we reach here, the routing strategy wasn't recognized
        logger.warning(f"Unrecognized routing strategy: {self.routing_strategy}, using FAILOVER as default")
        # Default to failover strategy (sort by priority)
        sorted_instances = sorted(available_instances, key=lambda x: x.priority)
        return sorted_instances[0]
    
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
