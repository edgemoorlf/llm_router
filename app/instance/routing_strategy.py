from enum import Enum
import random
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from .api_instance import APIInstance

logger = logging.getLogger(__name__)

class RoutingStrategy(str, Enum):
    """Strategy used to select an API instance."""
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin" 
    WEIGHTED = "weighted"
    LEAST_LOADED = "least_loaded"
    FAILOVER = "failover"
    MODEL_SPECIFIC = "model_specific"

class RoutingStrategyBase(ABC):
    """Base class for routing strategy implementations."""
    
    @abstractmethod
    def select_instance(self, 
                        instances: List[APIInstance], 
                        required_tokens: int,
                        model_name: Optional[str] = None) -> Optional[APIInstance]:
        """
        Select an API instance based on the specific strategy.
        
        Args:
            instances: List of available API instances
            required_tokens: Estimated number of tokens required for the request
            model_name: The model name requested (optional)
            
        Returns:
            Selected instance or None if no suitable instance is available
        """
        pass

class PriorityStrategy(RoutingStrategyBase):
    """Selects instances based on priority (lower is higher priority)."""
    
    def select_instance(self, instances: List[APIInstance], required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        if not instances:
            return None
            
        # Sort by priority (lower is higher priority)
        sorted_instances = sorted(instances, key=lambda x: x.priority)
        selected = sorted_instances[0]
        logger.debug(f"PRIORITY strategy selected instance {selected.name} with priority {selected.priority}")
        return selected

class RoundRobinStrategy(RoutingStrategyBase):
    """Selects instances in a round-robin fashion."""
    
    def __init__(self):
        self.index = 0
    
    def select_instance(self, instances: List[APIInstance], required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        if not instances:
            return None
            
        # Ensure index is valid
        if self.index >= len(instances):
            self.index = 0
            
        # Simple round-robin
        selected = instances[self.index]
        self.index = (self.index + 1) % len(instances)
        logger.debug(f"ROUND_ROBIN strategy selected instance {selected.name}, next index: {self.index}")
        return selected

class WeightedStrategy(RoutingStrategyBase):
    """Selects instances based on their weight."""
    
    def select_instance(self, instances: List[APIInstance], required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        if not instances:
            return None
            
        # Weighted random selection
        total_weight = sum(instance.weight for instance in instances)
        
        if total_weight == 0:
            # If all weights are 0, use simple round-robin
            round_robin = RoundRobinStrategy()
            return round_robin.select_instance(instances, required_tokens, model_name)
        
        # Weighted random selection
        r = random.uniform(0, total_weight)
        upto = 0
        for instance in instances:
            upto += instance.weight
            if upto >= r:
                logger.debug(f"WEIGHTED strategy selected instance {instance.name} with weight {instance.weight}/{total_weight}")
                return instance
        
        # Should never reach here if weights are positive
        logger.warning("WEIGHTED strategy selection error, falling back to first instance")
        return instances[0]

class LeastLoadedStrategy(RoutingStrategyBase):
    """Selects the instance with the lowest current TPM usage."""
    
    def select_instance(self, instances: List[APIInstance], required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        if not instances:
            return None
            
        # Sort by current TPM usage (lower is better)
        sorted_instances = sorted(instances, key=lambda x: x.instance_stats.current_tpm)
        selected = sorted_instances[0]
        logger.debug(f"LEAST_LOADED strategy selected instance {selected.name} with {selected.instance_stats.current_tpm} TPM")
        return selected

class FailoverStrategy(RoutingStrategyBase):
    """Failover strategy that selects instances based on priority."""
    
    def select_instance(self, instances: List[APIInstance], required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        if not instances:
            return None
            
        # Sort by priority (lower is higher priority) for failover
        sorted_instances = sorted(instances, key=lambda x: x.priority)
        selected = sorted_instances[0]
        logger.debug(f"FAILOVER strategy selected instance {selected.name} with priority {selected.priority}")
        return selected

class ModelSpecificStrategy(RoutingStrategyBase):
    """Model-specific strategy that selects instances based on model support and priority."""
    
    def select_instance(self, instances: List[APIInstance], required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        if not instances:
            return None
            
        # We assume instances are already filtered by model support
        # Sort by priority within those instances
        sorted_instances = sorted(instances, key=lambda x: x.priority)
        selected = sorted_instances[0]
        
        # If we have a model name, log which instance we're using for it
        if model_name:
            logger.info(f"MODEL_SPECIFIC routing: selected instance {selected.name} for model {model_name}")
        
        return selected

class RoutingStrategyFactory:
    """Factory for creating routing strategy instances."""
    
    @staticmethod
    def create_strategy(strategy_type: RoutingStrategy) -> RoutingStrategyBase:
        """
        Create a routing strategy instance based on the strategy type.
        
        Args:
            strategy_type: The type of routing strategy to create
            
        Returns:
            A routing strategy instance
            
        Raises:
            ValueError: If an unknown strategy type is provided
        """
        strategies = {
            RoutingStrategy.PRIORITY: PriorityStrategy(),
            RoutingStrategy.ROUND_ROBIN: RoundRobinStrategy(),
            RoutingStrategy.WEIGHTED: WeightedStrategy(),
            RoutingStrategy.LEAST_LOADED: LeastLoadedStrategy(),
            RoutingStrategy.FAILOVER: FailoverStrategy(),
            RoutingStrategy.MODEL_SPECIFIC: ModelSpecificStrategy(),
        }
        
        if strategy_type not in strategies:
            logger.warning(f"Unknown routing strategy: {strategy_type}, using FAILOVER as default")
            return strategies[RoutingStrategy.FAILOVER]
            
        return strategies[strategy_type]
