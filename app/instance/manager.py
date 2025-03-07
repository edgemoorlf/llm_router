"""Instance manager for multiple OpenAI-compatible service instances."""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

from .routing_strategy import RoutingStrategy
from .api_instance import APIInstance
from .config_loader import InstanceConfigLoader
from .router import InstanceRouter
from .forwarder import RequestForwarder
from .monitor import InstanceMonitor

class InstanceManager:
    """Manager for multiple API service instances with load balancing and failover."""
    
    def __init__(self, routing_strategy: RoutingStrategy = RoutingStrategy.FAILOVER):
        """
        Initialize the instance manager.
        
        Args:
            routing_strategy: Strategy for selecting instances
        """
        self.instances: Dict[str, APIInstance] = {}
        self.router = InstanceRouter(routing_strategy)
        self.forwarder = RequestForwarder()
        self.config_loader = InstanceConfigLoader()
        self.monitor = InstanceMonitor()
        
        # Load instances from environment variables
        self.instances = self.config_loader.load_from_env()
        
        # If no instances found, try loading legacy configuration
        if not self.instances:
            logger.warning("No API instances configured. Falling back to legacy single instance configuration.")
            self.instances = self.config_loader.load_legacy_instance()
        
        logger.info(f"Initialized {len(self.instances)} API instances with {routing_strategy} routing strategy")
    
    def load_from_csv(self, csv_path: str) -> None:
        """
        Load instances from a CSV file, replacing existing configuration.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.instances.clear()
        instances = self.config_loader.load_from_csv(csv_path)
        self.instances.update(instances)
        
        if not self.instances:
            logger.warning("No instances loaded from CSV. Falling back to environment variables.")
            self.instances = self.config_loader.load_from_env()
            
        logger.info(f"Loaded {len(self.instances)} instances from CSV")
    
    def select_instance(self, required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        """
        Select an API instance based on the configured routing strategy and model support.
        
        Args:
            required_tokens: Estimated number of tokens required for the request
            model_name: The model name requested (optional)
            
        Returns:
            Selected instance or None if no suitable instance is available
        """
        return self.router.select_instance(self.instances, required_tokens, model_name)
    
    async def try_instances(self, 
                           endpoint: str, 
                           deployment: str, 
                           payload: Dict[str, Any], 
                           required_tokens: int,
                           method: str = "POST",
                           provider_type: Optional[str] = None) -> Tuple[Dict[str, Any], APIInstance]:
        """
        Try instances until one succeeds or all fail.
        
        Args:
            endpoint: The API endpoint
            deployment: The deployment name
            payload: The request payload
            required_tokens: Estimated tokens required for the request
            method: HTTP method
            provider_type: Optional provider type to filter instances (e.g., "azure" or "generic")
            
        Returns:
            Tuple of (response, instance used)
            
        Raises:
            HTTPException: If all instances fail
        """
        return await self.forwarder.try_instances(
            self.instances,
            endpoint,
            deployment,
            payload,
            required_tokens,
            self.router,
            method,
            provider_type
        )
    
    def get_instance_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all instances.
        
        Returns:
            List of instance statistics dictionaries
        """
        return self.monitor.get_instance_stats(self.instances)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status summary for all instances.
        
        Returns:
            Summary of instance health status
        """
        return self.monitor.get_health_status(self.instances)


# Create a singleton instance with the default routing strategy
instance_manager = InstanceManager(
    routing_strategy=RoutingStrategy(os.getenv("API_ROUTING_STRATEGY", RoutingStrategy.FAILOVER))
) 